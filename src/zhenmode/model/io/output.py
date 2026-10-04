"""Output locations, effective run metadata and final records."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass

from zhenmode.model.diagnostics.budgets import (
    METRIC_NAMES,
    NONLINEAR_PROCESS_NAMES,
    SOURCE_NAMES,
    STAGE_NAMES,
)
from zhenmode.model.diagnostics.snapshot import diagnostics_to_arrays
from zhenmode.model.io.records import history_arrays
from zhenmode.model.io.restart import fingerprint
from zhenmode.model.runtime.reporting import _Tee
from zhenmode.model.solver.dynamics.projection import projection_config
from zhenmode.model.solver.numerics.backend import np
from zhenmode.model.solver.state import JaxStateG, _state_identity


@dataclass
class RunPaths:
    tag: str
    out_npz: str
    out_log: str
    three_d_dir: str | None
    three_d_terms_dir: str | None

def prepare_output_paths(args):
    tag = args.tag or f"g{int(args.days)}d"
    out_npz = os.path.join(args.out_dir, f"global_{tag}.npz")
    if os.path.exists(out_npz):
        raise FileExistsError(f"result already exists: {out_npz}; select a new --tag or --out-dir")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    out_log = os.path.join(args.log_dir, f"global_{tag}.log")
    sys.stdout = _Tee(out_log)
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  {sys.executable}")
    print(f"# {' '.join(sys.argv)}")
    three_d_dir = None
    if args.save_3d:
        three_d_dir = os.path.join(args.out_dir, f"global_{tag}_3d")
        os.makedirs(three_d_dir, exist_ok=True)
    three_d_terms_dir = None
    if args.save_3d_terms:
        assert args.save_3d, "--save-3d-terms requires --save-3d"
        three_d_terms_dir = os.path.join(args.out_dir, f"global_{tag}_terms")
        os.makedirs(three_d_terms_dir, exist_ok=True)
    return RunPaths(tag, out_npz, out_log, three_d_dir, three_d_terms_dir)

@dataclass
class RunOutcome:
    state: JaxStateG
    rejected: JaxStateG | None
    audited: JaxStateG | None
    verdict: str
    monotonic_drift: bool
    amplitude_bounded: bool

def effective_config(args, context):
    return {
        "days": args.days,
        "snap_days": args.snap_days,
        "diagnostics_schema_version": 3,
        "monitor_schema_version": 1,
        "restart_schema_version": 1,
        "lat_max": args.lat_max,
        "ny": context.inputs.grid.ny,
        "nx": context.inputs.grid.nx,
        "nz": context.inputs.grid.nz,
        "resolution": float(context.inputs.gcfg.resolution),
        "resolution_remap": args.resolution_remap,
        "z_levels": args.z_levels or "",
        "dt": args.dt,
        "dtype": args.dtype,
        "nu_h": context.inputs.physics.nu_h,
        "nu_bi": context.inputs.physics.nu_bi,
        "kappa_v": context.inputs.physics.kappa_v,
        "kappa_conv": context.inputs.physics.kappa_conv,
        "kappa_gm": context.inputs.physics.kappa_gm,
        "kappa_redi": context.inputs.physics.kappa_redi,
        "gm_slope_max": context.inputs.physics.gm_slope_max,
        "mode_split": bool(args.mode_split),
        "dt_bt": float(context.solver.params.dt_bt),
        "n_subcyc": int(context.solver.params.n_subcyc),
        "nu_nsub": context.solver.params.nu_nsub,
        "use_scan": bool(args.use_scan),
        "freeze_adv_vel": args.freeze_adv_vel,
        "conservative_kv": args.conservative_kv,
        "project_adv_vel": args.project_adv_vel,
        "column_projection": projection_config(context.solver.params),
        "localize_conv": args.localize_conv,
        "monotone_adv": args.monotone_adv,
        "fct_adv": args.fct_adv,
        "mixed_layer_depth_m": args.mixed_layer_depth,
        "mixed_layer_mode": args.mixed_layer_mode,
        "mixed_layer_lat_band": args.mixed_layer_lat_band,
        "mld_density_delta": args.mld_density_delta,
        "mixed_layer_depth_min": args.mixed_layer_depth_min,
        "mixed_layer_depth_max": args.mixed_layer_depth_max,
        "ice_freeze_temp": args.ice_freeze_temp,
        "ice_salt_flux": args.ice_salt_flux,
        "dynamic_ice": bool(args.dynamic_ice),
        "ice_insulation_scale_m": args.ice_insulation_scale_m,
        "lambda_bulk": context.forcing.lambda_bulk,
        "bulk_lambda_mult": args.bulk_lambda_mult,
        "real_air_temp": bool(args.real_air_temp),
        "real_air_temp_monthly": bool(args.real_air_temp_monthly),
        "seasonal_wind": context.forcing.seasonal,
        "wind_year": args.wind_year,
        "wind_month": args.month,
        "wind_jit": bool(args.wind_jit),
        "wind_blend_days": args.wind_blend_days,
        "sss_restore_days": args.sss_restore_days,
        "sss_restore_zonal": bool(args.sss_restore_zonal),
        "sponge_days": args.sponge_days,
        "sponge_cells": args.sponge_cells,
        "polar_cap_rows": args.polar_cap_rows,
        "polar_cap_taper": args.polar_cap_taper,
        "eta_relax_days": args.eta_relax_days,
        "eta_relax_box": args.eta_relax_box,
        "eta_relax_buffer": args.eta_relax_buffer,
        "smooth_passes": args.smooth_passes,
        "min_depth": args.min_depth,
        "coastal_restore_days": args.coastal_restore_days,
        "global_sst_restore_days": args.global_sst_restore_days,
        "coastal_restore_cells": args.coastal_restore_cells,
        "coastal_restore_taper": args.coastal_restore_taper,
        "coastal_bulk_lambda": args.coastal_bulk_lambda,
        "coastal_kappa_h": args.coastal_kappa_h,
        "coastal_kappa_v": args.coastal_kappa_v,
        "init_from": args.init_from or "",
    }

def write_final_records(
    args, requested_steps, context, paths, recovery, history, counters, ledger, outcome
):
    config_dict = effective_config(args, context)
    if counters.failure_code:
        counters.rejected_path = os.path.join(args.out_dir, f"rejected_{paths.tag}.npz")
        with open(counters.rejected_path, "xb") as stream:
            np.savez_compressed(
                stream,
                **{
                    name: np.asarray(getattr(outcome.rejected, name))
                    for name in outcome.rejected._fields
                },
                **{"ledger_" + name: np.asarray(value) for name, value in ledger.rejected.items()},
                **{
                    "audited_" + name: np.asarray(getattr(outcome.audited, name))
                    for name in outcome.audited._fields
                }
                if counters.failure_code == 7
                else {},
                audit_mismatch_fields=np.asarray(ledger.mismatch_fields, dtype=str),
                resumable=False,
                failure_code=counters.failure_code,
                attempted_step=counters.first_rejected_step,
                accepted_step=counters.cur,
                elapsed_seconds=counters.diverged_at * 86400.0,
            )
    with open(paths.out_npz, "xb") as stream:
        np.savez_compressed(
            stream,
            **history_arrays(history),
            **diagnostics_to_arrays(history.snap_budget),
            **{name: np.asarray(value) for name, value in ledger.history.items()},
            budget_audit_enabled=args.budget_audit,
            ledger_metric_names=np.asarray(METRIC_NAMES),
            ledger_stage_names=np.asarray(STAGE_NAMES),
            ledger_source_names=np.asarray(SOURCE_NAMES),
            ledger_process_names=np.asarray(NONLINEAR_PROCESS_NAMES),
            ledger_inventory_kind="fixed_reference_node_water_minus_ice_latent; eta_displacement_separate",
            ledger_boundary_kind="internal_advection_reference_boundary_diagnostic_not_external_netboundaryflux",
            physical_budget_closed=False,
            external_netboundaryflux_status="not_measured_do_not_substitute_numerical_residual",
            actual_moving_volume_inventory_status="not_implemented_in_this_legacy_driver",
            effective_config_identity_json=json.dumps(
                fingerprint(context.solver.params), sort_keys=True
            ),
            grid_identity_json=json.dumps(fingerprint(context.inputs.grid), sort_keys=True),
            ledger_heat_residual_W_m2=float(ledger.totals["budget_residual"][0])
            / (
                float(
                    np.sum(
                        np.asarray(context.inputs.grid.dx_2d)
                        * context.inputs.grid.dy
                        * context.inputs.ocean
                    )
                )
                * counters.cur
                * args.dt
            )
            if args.budget_audit and counters.cur
            else np.nan,
            ledger_absolute_heat_residual_W_m2=float(ledger.totals["absolute_budget_residual"][0])
            / (
                float(
                    np.sum(
                        np.asarray(context.inputs.grid.dx_2d)
                        * context.inputs.grid.dy
                        * context.inputs.ocean
                    )
                )
                * counters.cur
                * args.dt
            )
            if args.budget_audit and counters.cur
            else np.nan,
            final_state_identity_json=json.dumps(_state_identity(outcome.state), sort_keys=True),
            forcing_provenance_json=json.dumps(context.forcing.forcing_provenance, sort_keys=True),
            source_identity_json=json.dumps(context.forcing.source_identity, sort_keys=True),
            execution_identity_json=json.dumps(context.forcing.execution_identity, sort_keys=True),
            forcing_phase_seconds=counters.cur * args.dt % (360 * 86400),
            elapsed_seconds=counters.cur * args.dt,
            diagnostics_schema_version=4,
            monitor_schema_version=1,
            salt_content_units="kg",
            T_init=context.inputs.T_init,
            S_init=context.inputs.S_init,
            wet_mask=np.asarray(context.inputs.grid.wet_mask),
            lat=np.asarray(context.inputs.grid.lat),
            lon=np.asarray(context.inputs.grid.lon),
            z=np.asarray(context.inputs.grid.z),
            verdict=outcome.verdict,
            diverged_at=counters.diverged_at if counters.diverged_at is not None else -1.0,
            monotonic_drift=outcome.monotonic_drift,
            amplitude_bounded=outcome.amplitude_bounded,
            max_u_peak=counters.max_u_peak,
            max_velocity_peak=counters.max_velocity_peak,
            max_eta_peak=counters.max_eta_peak,
            requested_steps=requested_steps,
            accepted_steps=counters.cur,
            attempted_steps=counters.attempted_steps,
            first_rejected_step=counters.first_rejected_step,
            failure_code=counters.failure_code,
            rejected_state_path=counters.rejected_path,
            duration_complete=counters.cur == requested_steps and (not counters.failure_code),
            n_3d_snaps=counters.n_3d_snaps,
            three_d_dir=paths.three_d_dir or "",
            config=str(config_dict),
        )
    print(f"  saved {paths.out_npz}")
    if paths.three_d_dir:
        print(f"  3D snapshots: {counters.n_3d_snaps} files in {paths.three_d_dir}")
    return 0 if outcome.verdict == "PASS" else 3 if outcome.verdict == "INCOMPLETE" else 1
