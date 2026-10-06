"""Diagnostic history, exclusive snapshots and independent restart history validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from zhenmode.model.diagnostics.budgets import empty_budget
from zhenmode.model.diagnostics.snapshot import (
    BudgetDiagnostics,
    compute_budget_diagnostics,
    total_kinetic_energy,
)
from zhenmode.model.runtime.cli import ETA_BLOWUP_M, MAX_U_BOUND
from zhenmode.model.solver.numerics.backend import jax, jnp, np
from zhenmode.provenance.sources import sha256_file as file_sha256

HISTORY_ATTRIBUTES = {
    "days": "snap_days",
    "max_u": "snap_maxu",
    "max_velocity": "snap_maxvelocity",
    "max_T": "snap_maxT",
    "max_eta": "snap_maxeta",
    "ssh_std": "snap_sshstd",
    "ke": "snap_ke",
    "eta": "snap_eta",
    "T_top": "snap_T_top",
    "ice_top": "snap_ice_top",
    "ice_fraction": "snap_ice_fraction",
}


def history_arrays(history):
    """Stored diagnostic names shared by output and continuation."""
    return {name: np.array(getattr(history, attr)) for name, attr in HISTORY_ATTRIBUTES.items()}


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
        for name, attribute in HISTORY_ATTRIBUTES.items():
            setattr(history, attribute, list(recovery.restored.history[name]))
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
    history.snap_budget.append(compute_budget_diagnostics(state, context.inputs.grid,
        thermodynamics=context.inputs.physics.thermodynamics))
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

SNAPSHOT_SCALARS = (
    "days",
    "max_u",
    "max_velocity",
    "max_T",
    "max_eta",
    "ssh_std",
    "ke",
    "ice_fraction",
)

SNAPSHOT_FIELDS = ("eta", "T_top", "ice_top")

def _validate_restart_history(
    record, grid, n_snap, dt, save_3d, save_terms, output_dirs, budget_audit=False
):
    """A production continuation restores all diagnostics and verifies retained files."""
    if record.step % n_snap:
        raise ValueError("restart must land on a snapshot boundary")
    count = record.step // n_snap + 1
    budget_names = tuple(BudgetDiagnostics.__dataclass_fields__)
    expected = set(SNAPSHOT_SCALARS + SNAPSHOT_FIELDS + budget_names)
    ledger_shapes = (
        {"ledger_" + name: np.shape(value) for name, value in empty_budget().items()}
        if budget_audit
        else {}
    )
    expected.update(ledger_shapes)
    if set(record.history) != expected:
        raise ValueError("restart snapshot/diagnostic history fields mismatch")
    for name, values in record.history.items():
        shape = (
            (count,) + ledger_shapes[name]
            if name in ledger_shapes
            else (count, grid.nx, grid.ny)
            if name in SNAPSHOT_FIELDS
            else (count,)
        )
        if values.shape != shape:
            raise ValueError(f"restart history {name} shape differs from the snapshot timeline")
    days = np.arange(count) * n_snap * dt / 86400.0
    if not np.array_equal(record.history["days"], days):
        raise ValueError("restart history days differ from the absolute snapshot timeline")
    expected_count = count if save_3d else 0
    if record.counters != {
        "n_3d_snaps": expected_count,
        "accepted_steps": record.step,
        "attempted_steps": record.step,
    }:
        raise ValueError("restart actual 3D output counter mismatch")
    stat_names = {"max_u_peak", "max_velocity_peak", "max_eta_peak"}
    if (
        set(record.cumulative) != stat_names | set(ledger_shapes)
        or any(record.cumulative[name].shape != () for name in stat_names)
        or any(record.cumulative[name].shape != shape for name, shape in ledger_shapes.items())
    ):
        raise ValueError("restart production statistics mismatch")
    for name in ledger_shapes:
        if record.cumulative[name].tobytes() != record.history[name][-1].tobytes():
            raise ValueError("restart cumulative ledger differs from final history")
    eta_peak = float(record.cumulative["max_eta_peak"])
    if eta_peak < max(record.history["max_eta"]) or eta_peak > ETA_BLOWUP_M:
        raise ValueError("restart historical eta peak mismatch")
    peak = float(record.cumulative["max_u_peak"])
    velocity_peak = float(record.cumulative["max_velocity_peak"])
    if (
        peak < max(record.history["max_u"])
        or velocity_peak < max(record.history["max_velocity"])
        or velocity_peak < peak
        or velocity_peak > MAX_U_BOUND
    ):
        raise ValueError("restart historical velocity peak mismatch")
    for snapshot_field, state_field in (("eta", "eta"), ("T_top", "T"), ("ice_top", "ice")):
        current = record.state[state_field]
        if state_field == "T":
            current = current[..., 0]
        if not np.array_equal(record.history[snapshot_field][-1], current):
            raise ValueError(f"restart final snapshot {snapshot_field} differs from state")
    expected_outputs = {f"3d/snap_{index:05d}.npy" for index in range(expected_count)}
    if save_terms:
        expected_outputs.update(f"terms/terms_{index:05d}.npy" for index in range(expected_count))
    if set(record.outputs) != expected_outputs:
        raise ValueError("restart output manifest differs from the actual counters")
    for relative_path, digest in record.outputs.items():
        category, filename = relative_path.split("/")
        path = Path(output_dirs[category]) / filename
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"restart retained snapshot is missing or changed: {path}")
