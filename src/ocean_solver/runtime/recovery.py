"""Validate historical records and restore a source-bound checkpoint."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

from diagnostics import BudgetDiagnostics
from ocean_solver.audit.schema import empty_budget
from ocean_solver.fd.backend import jnp, np
from ocean_solver.fd.types import JaxStateG
from ocean_solver.runtime.cli import ETA_BLOWUP_M, MAX_U_BOUND
from ocean_solver.runtime.identity import SOURCE_MODULES
from restart_contract import RestartRecord, file_sha256, load_restart

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
    for field, state_field in (("eta", "eta"), ("T_top", "T"), ("ice_top", "ice")):
        current = record.state[state_field]
        if state_field == "T":
            current = current[..., 0]
        if not np.array_equal(record.history[field][-1], current):
            raise ValueError(f"restart final snapshot {field} differs from state")
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


@dataclass
class OutputManifest:
    files: dict = dataclass_field(default_factory=dict)


@dataclass
class RecoveryContext:
    state: JaxStateG
    n_total: int
    n_snap: int
    ckpt_path: str
    checkpoint_contract: dict | None
    restored: RestartRecord | None
    output_manifest: OutputManifest
    start_step: int
    n_3d_snaps: int
    n_ckpt: int | None


def prepare_recovery(args, requested_steps, context, paths, services, source_directory):
    state = context.solver.state
    n_ckpt = None
    n_total = requested_steps
    if args.max_steps > 0:
        n_total = min(n_total, args.max_steps)
    n_snap = max(1, int(round(args.snap_days * 86400.0 / args.dt)))
    ckpt_path = os.path.join(args.out_dir, f"ckpt_{paths.tag}.npz")
    checkpoint_contract = None
    restored = None
    output_manifest = OutputManifest()
    if args.restart_from or args.checkpoint_days > 0:
        source_dir = Path(source_directory)
        source_names = SOURCE_MODULES
        checkpoint_contract = services.make_restart_contract(
            context.inputs.grid,
            context.solver.params,
            dtype=args.dtype,
            forcing={
                "initial_T": context.inputs.T_init,
                "initial_S": context.inputs.S_init,
                "baked": context.forcing.forcing_baked,
                "wind_months": context.forcing.wind_months,
                "air_months": context.forcing.T_atm_months,
            },
            controls={
                "calendar": "repeating_360_day_30_day_months",
                "seasonal": context.forcing.seasonal,
                "wind_blend_days": args.wind_blend_days,
                "wind_jit": args.wind_jit,
                "n_snap": n_snap,
                "save_3d": args.save_3d,
                "save_3d_terms": args.save_3d_terms,
                "budget_kind": "strict_shadow_accepted_stage_ledger_v1"
                if args.budget_audit
                else "snapshot_inventory_only_no_flux_ledger",
                "forcing_provenance": context.forcing.forcing_provenance,
                "production_schema_version": 4,
                "monitor_schema_version": 1,
                "monitor_comparison": ">",
                "velocity_limit": MAX_U_BOUND,
                "eta_limit": ETA_BLOWUP_M,
            },
            code_paths={name: source_dir / f"{name}.py" for name in source_names},
            execution=context.forcing.execution_identity,
        )
    start_step = 0
    n_3d_snaps = 0
    if args.restart_from:
        restored = load_restart(args.restart_from, checkpoint_contract)
        _validate_restart_history(
            restored,
            context.inputs.grid,
            n_snap,
            args.dt,
            args.save_3d,
            args.save_3d_terms,
            {"3d": paths.three_d_dir, "terms": paths.three_d_terms_dir},
            args.budget_audit,
        )
        state = JaxStateG(**{name: jnp.asarray(value) for name, value in restored.state.items()})
        start_step = restored.step
        if start_step > n_total:
            raise ValueError("restart step exceeds the requested final step")
        n_3d_snaps = restored.counters["n_3d_snaps"]
        output_manifest = OutputManifest(dict(restored.outputs))
        print(
            f"RESUME from {args.restart_from}: step {start_step} (day {start_step * args.dt / 86400.0:.1f}), {len(restored.history['days'])} diagnostic rows restored, {n_3d_snaps} verified 3D snapshots kept"
        )
    if args.checkpoint_days > 0:
        n_ckpt = max(1, int(round(args.checkpoint_days * 86400.0 / args.dt)))
        if n_ckpt % n_snap:
            raise ValueError("checkpoint cadence must be an integer multiple of snapshot cadence")
    return RecoveryContext(
        state=state,
        n_total=n_total,
        n_snap=n_snap,
        ckpt_path=ckpt_path,
        checkpoint_contract=checkpoint_contract,
        restored=restored,
        output_manifest=output_manifest,
        start_step=start_step,
        n_3d_snaps=n_3d_snaps,
        n_ckpt=n_ckpt,
    )
