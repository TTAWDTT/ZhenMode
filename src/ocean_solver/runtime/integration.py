"""Accept or reject each proposed step before updating records and checkpoints."""

from __future__ import annotations

import time

from ocean_solver.audit.monitor import classify_state
from ocean_solver.audit.schema import accumulate_budget
from ocean_solver.diagnostics.state import diagnostics_to_arrays
from ocean_solver.io.output import RunOutcome, write_final_records
from ocean_solver.io.records import capture_snapshot, restore_records
from ocean_solver.io.restart import save_restart
from ocean_solver.numerics.backend import jnp, np
from ocean_solver.runtime.cli import AMPLITUDE_CAP_C, DRIFT_TOL_C, ETA_BLOWUP_M, MAX_U_BOUND
from ocean_solver.runtime.identity import _same_state_bytes
from ocean_solver.runtime.reporting import print_run_header


def run_integration(args, requested_steps, context, paths, recovery):
    print_run_header(args, context, recovery)
    history, counters, ledger = restore_records(args, recovery)
    state = recovery.state
    rejected = None
    audited = None
    t0 = time.time()
    if recovery.restored is None:
        maxu, maxT, maxeta, nan = capture_snapshot(
            args, context, paths, recovery, history, counters, ledger, state, 0
        )
    else:
        maxu, maxT, maxeta, nan = (
            history.snap_maxu[-1],
            history.snap_maxT[-1],
            history.snap_maxeta[-1],
            0,
        )
    counters.cur = recovery.start_step
    counters.attempted_steps = recovery.start_step
    counters.first_rejected_step = -1
    counters.rejected_path = ""
    initial_metrics = classify_state(state, MAX_U_BOUND, ETA_BLOWUP_M)
    counters.max_u_peak = float(jnp.maximum(counters.max_u_peak, initial_metrics.max_u))
    counters.max_velocity_peak = float(
        jnp.maximum(counters.max_velocity_peak, initial_metrics.max_velocity)
    )
    counters.max_eta_peak = float(jnp.maximum(counters.max_eta_peak, initial_metrics.max_eta))
    counters.failure_code = int(initial_metrics.failure)
    if counters.failure_code:
        counters.diverged_at = counters.cur * args.dt / 86400.0
        counters.first_rejected_step = counters.cur
        counters.diverge_reason = f"invalid incoming state (monitor code {counters.failure_code})"
        rejected = state
    while counters.cur < recovery.n_total and (not counters.failure_code):
        take = min(recovery.n_snap, recovery.n_total - counters.cur)
        for _ in range(take):
            proposed = context.solver.do_step(state, counters.cur * args.dt / 86400.0)
            interval_ledger = {}
            if args.budget_audit:
                proposed, audited, interval_ledger = proposed
            metrics = classify_state(proposed, MAX_U_BOUND, ETA_BLOWUP_M)
            counters.attempted_steps += 1
            counters.max_u_peak = float(jnp.maximum(counters.max_u_peak, metrics.max_u))
            counters.max_velocity_peak = float(
                jnp.maximum(counters.max_velocity_peak, metrics.max_velocity)
            )
            counters.max_eta_peak = float(jnp.maximum(counters.max_eta_peak, metrics.max_eta))
            counters.failure_code = int(metrics.failure)
            if args.budget_audit:
                ledger.mismatch_fields = [
                    name
                    for name in proposed._fields
                    if not _same_state_bytes((getattr(proposed, name),), (getattr(audited, name),))
                ]
                pending_totals = accumulate_budget(ledger.totals, interval_ledger)
                if not counters.failure_code and ledger.mismatch_fields:
                    counters.failure_code = 7
                if not counters.failure_code and (
                    not all(
                        (
                            np.isfinite(np.asarray(value)).all()
                            for value in (*interval_ledger.values(), *pending_totals.values())
                        )
                    )
                ):
                    counters.failure_code = 2
            if counters.failure_code:
                rejected = proposed
                ledger.rejected = interval_ledger
                counters.first_rejected_step = counters.attempted_steps
                counters.diverged_at = counters.attempted_steps * args.dt / 86400.0
                counters.diverge_reason = f"first rejected step {counters.attempted_steps} (monitor code {counters.failure_code})"
                break
            state = proposed
            if args.budget_audit:
                ledger.totals = pending_totals
            counters.cur += 1
        if counters.cur * args.dt / 86400.0 != history.snap_days[-1]:
            maxu, maxT, maxeta, nan = capture_snapshot(
                args, context, paths, recovery, history, counters, ledger, state, counters.cur
            )
        if counters.failure_code:
            break
        if (
            args.checkpoint_days > 0
            and counters.cur % recovery.n_ckpt == 0
            and (counters.cur < recovery.n_total)
            and (counters.cur > recovery.start_step)
        ):
            checkpoint_history = {
                "days": history.snap_days,
                "max_u": history.snap_maxu,
                "max_velocity": history.snap_maxvelocity,
                "max_T": history.snap_maxT,
                "max_eta": history.snap_maxeta,
                "ssh_std": history.snap_sshstd,
                "ke": history.snap_ke,
                "eta": history.snap_eta,
                "T_top": history.snap_T_top,
                "ice_top": history.snap_ice_top,
                "ice_fraction": history.snap_ice_fraction,
                **diagnostics_to_arrays(history.snap_budget),
                **ledger.history,
            }
            save_restart(
                recovery.ckpt_path,
                state,
                recovery.checkpoint_contract,
                step=counters.cur,
                counters={
                    "n_3d_snaps": counters.n_3d_snaps,
                    "accepted_steps": counters.cur,
                    "attempted_steps": counters.attempted_steps,
                },
                cumulative={
                    "max_u_peak": counters.max_u_peak,
                    "max_velocity_peak": counters.max_velocity_peak,
                    "max_eta_peak": counters.max_eta_peak,
                    **{"ledger_" + name: value for name, value in ledger.totals.items()},
                },
                history=checkpoint_history,
                outputs=recovery.output_manifest.files,
            )
            print(
                f"  [ckpt] saved verified restart {recovery.ckpt_path} at step {counters.cur}",
                flush=True,
            )
    wall = time.time() - t0
    monotonic_drift = False
    amplitude_bounded = True
    if len(history.maxT_history) >= 4 and counters.diverged_at is None:
        half = len(history.maxT_history) // 2
        first_half_max = max(history.maxT_history[:half])
        final_quarter = history.maxT_history[-max(1, len(history.maxT_history) // 4) :]
        if final_quarter and max(final_quarter) > first_half_max + DRIFT_TOL_C:
            monotonic_drift = True
        if final_quarter and max(final_quarter) > context.inputs.T_init_max + AMPLITUDE_CAP_C:
            amplitude_bounded = False
    if counters.diverged_at is not None:
        verdict = (
            "FAIL_AUDIT_IDENTITY"
            if counters.failure_code == 7
            else "FAIL_LEDGER"
            if counters.failure_code == 2
            else "FAIL_BLOWUP"
        )
    elif monotonic_drift or not amplitude_bounded:
        verdict = "FAIL_DRIFT"
    elif counters.cur < requested_steps:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"
    print("=" * 78)
    print(
        f"VERDICT: {verdict}  (wall {wall / 60:.1f} min, {counters.cur} steps, max_u_peak={counters.max_u_peak:.3f})"
    )
    if counters.diverged_at is not None:
        print(f"  diverged at day {counters.diverged_at:.1f}: {counters.diverge_reason}")
    print(f"  monotonic_drift={monotonic_drift}  amplitude_bounded={amplitude_bounded}")
    print(f"  final max|u|={maxu:.3f}  max|T|={maxT:.3f}  max|eta|={maxeta:.3f}")
    outcome = RunOutcome(state, rejected, audited, verdict, monotonic_drift, amplitude_bounded)
    return write_final_records(
        args, requested_steps, context, paths, recovery, history, counters, ledger, outcome
    )
