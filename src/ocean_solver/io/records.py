"""Own diagnostic history, accepted ledgers, counters, and exclusive snapshots."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ocean_solver.audit.schema import empty_budget
from ocean_solver.diagnostics.runtime import total_kinetic_energy
from ocean_solver.diagnostics.state import BudgetDiagnostics, compute_budget_diagnostics
from ocean_solver.io.restart import file_sha256
from ocean_solver.numerics.backend import jax, jnp, np


@dataclass
class SnapshotHistory:
    snap_days: list[float] = field(default_factory=list)
    snap_maxu: list[float] = field(default_factory=list)
    snap_maxvelocity: list[float] = field(default_factory=list)
    snap_maxT: list[float] = field(default_factory=list)
    snap_maxeta: list[float] = field(default_factory=list)
    snap_sshstd: list[float] = field(default_factory=list)
    snap_ke: list[float] = field(default_factory=list)
    snap_eta: list[np.ndarray] = field(default_factory=list)
    snap_T_top: list[np.ndarray] = field(default_factory=list)
    snap_ice_top: list[np.ndarray] = field(default_factory=list)
    snap_ice_fraction: list[float] = field(default_factory=list)
    snap_budget: list[BudgetDiagnostics] = field(default_factory=list)
    maxT_history: list[float] = field(default_factory=list)


@dataclass
class RunCounters:
    cur: int = 0
    attempted_steps: int = 0
    first_rejected_step: int = -1
    rejected_path: str = ""
    max_eta_peak: float = 0.0
    max_u_peak: float = 0.0
    max_velocity_peak: float = 0.0
    n_3d_snaps: int = 0
    failure_code: int = 0
    diverged_at: float | None = None
    diverge_reason: str = ""


@dataclass
class AcceptedLedger:
    totals: dict[str, jax.Array]
    history: dict[str, list[np.ndarray]]
    rejected: dict[str, jax.Array] = field(default_factory=dict)
    mismatch_fields: list[str] = field(default_factory=list)


def restore_records(args, recovery):
    history = SnapshotHistory()
    counters = RunCounters(n_3d_snaps=recovery.n_3d_snaps)
    totals = empty_budget() if args.budget_audit else {}
    ledger = AcceptedLedger(totals, {"ledger_" + name: [] for name in totals})
    if recovery.restored is not None:
        history.snap_days = list(recovery.restored.history["days"])
        history.snap_maxu = list(recovery.restored.history["max_u"])
        history.snap_maxvelocity = list(recovery.restored.history["max_velocity"])
        history.snap_maxT = list(recovery.restored.history["max_T"])
        history.snap_maxeta = list(recovery.restored.history["max_eta"])
        history.snap_sshstd = list(recovery.restored.history["ssh_std"])
        history.snap_ke = list(recovery.restored.history["ke"])
        history.snap_eta = list(recovery.restored.history["eta"])
        history.snap_T_top = list(recovery.restored.history["T_top"])
        history.snap_ice_top = list(recovery.restored.history["ice_top"])
        history.snap_ice_fraction = list(recovery.restored.history["ice_fraction"])
        history.snap_budget = [
            BudgetDiagnostics(
                **{
                    name: recovery.restored.history[name][index]
                    for name in BudgetDiagnostics.__dataclass_fields__
                }
            )
            for index in range(len(history.snap_days))
        ]
        history.maxT_history = list(history.snap_maxT)
        counters.max_eta_peak = float(recovery.restored.cumulative["max_eta_peak"])
        ledger.totals = {
            name: jnp.asarray(recovery.restored.cumulative["ledger_" + name])
            for name in ledger.totals
        }
        ledger.history = {name: list(recovery.restored.history[name]) for name in ledger.history}
        counters.max_u_peak = float(recovery.restored.cumulative["max_u_peak"])
        counters.max_velocity_peak = float(recovery.restored.cumulative["max_velocity_peak"])
    return history, counters, ledger


def _save_snapshot_file(path, array, outputs, relative_path):
    with open(path, "xb") as stream:
        np.save(stream, array)
    if outputs is not None:
        outputs[relative_path] = file_sha256(path)


def capture_snapshot(args, context, paths, recovery, history, counters, ledger, state, cur_step):
    day = cur_step * args.dt / 86400.0
    f64 = (lambda a: np.asarray(a, dtype=np.float64)) if args.dtype == "float32" else np.asarray
    eta = f64(state.eta)
    maxu = float(np.max(np.abs(f64(state.u))))
    maxvelocity = max(maxu, float(np.max(np.abs(f64(state.v)))))
    maxT = float(np.max(np.abs(f64(state.T))))
    maxeta = float(np.nanmax(np.abs(eta))) if np.isfinite(eta).any() else float("nan")
    sshstd = (
        float(np.std(eta[context.inputs.ocean])) if context.inputs.ocean.any() else float("nan")
    )
    ke = total_kinetic_energy(state, context.inputs.ocean)
    nan = int(np.sum(~np.isfinite(f64(state.u))))
    history.snap_days.append(day)
    history.snap_maxu.append(maxu)
    history.snap_maxvelocity.append(maxvelocity)
    history.snap_maxT.append(maxT)
    history.snap_maxeta.append(maxeta)
    history.snap_sshstd.append(sshstd)
    history.snap_ke.append(ke)
    history.snap_eta.append(eta.copy())
    history.snap_T_top.append(f64(state.T[:, :, 0]).copy())
    ice_top = (
        f64(state.ice)
        if np.asarray(state.ice).shape != ()
        else np.zeros_like(f64(state.T[:, :, 0]))
    )
    history.snap_ice_top.append(ice_top.copy())
    history.snap_ice_fraction.append(
        float(np.mean(ice_top[context.inputs.ocean] > 0.0)) if context.inputs.ocean.any() else 0.0
    )
    history.snap_budget.append(compute_budget_diagnostics(state, context.inputs.grid))
    for name, value in ledger.totals.items():
        ledger.history["ledger_" + name].append(np.asarray(value).copy())
    if args.save_3d:
        snap3d = np.stack(
            [f64(state.T).copy(), f64(state.u).copy(), f64(state.v).copy(), f64(state.S).copy()],
            axis=0,
        )
        filename = f"snap_{counters.n_3d_snaps:05d}.npy"
        _save_snapshot_file(
            os.path.join(paths.three_d_dir, filename),
            snap3d,
            recovery.output_manifest.files if recovery.checkpoint_contract is not None else None,
            f"3d/{filename}",
        )
        del snap3d
        if args.save_3d_terms:
            tstack = (
                np.asarray(context.solver.terms_fn(state), dtype=np.float64)
                if args.dtype == "float32"
                else np.asarray(context.solver.terms_fn(state))
            )
            filename = f"terms_{counters.n_3d_snaps:05d}.npy"
            _save_snapshot_file(
                os.path.join(paths.three_d_terms_dir, filename),
                tstack,
                recovery.output_manifest.files
                if recovery.checkpoint_contract is not None
                else None,
                f"terms/{filename}",
            )
            del tstack
        counters.n_3d_snaps += 1
        import gc

        gc.collect()
    history.maxT_history.append(maxT)
    print(
        f"{day:7.1f} {cur_step:8d} {maxu:9.3f} {maxT:8.3f} {maxeta:9.3f} {sshstd:9.4f} {ke:12.4e} {nan:6d}",
        flush=True,
    )
    return (maxu, maxT, maxeta, nan)
