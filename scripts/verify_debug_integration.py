"""Real-ETOPO numerical smoke; input kinds are explicit, not climate qualification."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG, GlobalGridConfig, PhysicsConfig
from diagnostics import compute_budget_diagnostics
from grid import global_grid_dims, land_distance_from_land_mask, make_global_grid
from jax_solver_global import make_solver_global, projection_config
from run_long_integration_global import ETA_BLOWUP_M, MAX_U_BOUND
from stage_budgets import (
    MAXIMUM_BUDGET_FIELDS,
    METRIC_NAMES,
    NONLINEAR_PROCESS_NAMES,
    SOURCE_NAMES,
    STAGE_NAMES,
    TRANSPORT_METRIC_NAMES,
    accumulate_budget,
    empty_budget,
    make_budget_step,
)


def make_smoke_fixture(resolution, bathymetry, kappa_bi):
    """Shared real-grid, explicitly synthetic-forcing numerical fixture."""
    nx, ny = global_grid_dims(resolution, 65., remap="area")
    config = replace(GlobalGridConfig(), resolution=resolution, lat_max=65., nx=nx, ny=ny)
    grid = make_global_grid(config, bathymetry, smooth_passes=80, min_depth=500., remap="area")
    latitude = np.radians(grid.lat)[None, :]
    sst = np.broadcast_to(np.maximum(-1.8, 26. - 40. * np.sin(latitude) ** 2), (nx, ny))
    initial_temperature = 2. + (sst[:, :, None] - 2.) * np.exp(grid.z[None, None, :] / 500.)
    initial_salinity = np.full_like(initial_temperature, 35.)
    atmosphere = sst - 5.
    forcing = (np.broadcast_to(0.05 * np.cos(3. * latitude), (nx, ny)),
               np.zeros((nx, ny)), np.zeros((nx, ny)))
    physics = replace(PhysicsConfig(), nu_h=2e6, nu_bi=0., kappa_bi=kappa_bi,
                      kappa_v=1e-6, kappa_conv=0.01, kappa_gm=0., kappa_redi=0.)
    return grid, physics, initial_temperature, initial_salinity, atmosphere, forcing


def make_monitored_advance(step, zero_budget, audited=False, transport_tolerances=None):
    """Stop on the first rejected step; never book its state or ledger as accepted.

    Failure codes: 1 non-finite state, 2 non-finite budget, 3 velocity, 4 eta,
    5 continuity, 6 tracer/fast transport. Optional tolerances require auditing.
    An invalid incoming state has zero attempts. Peaks include rejected values.
    Returned state and totals contain only accepted steps; rejected data is separate.
    """
    if transport_tolerances is not None:
        if not audited or len(transport_tolerances) != 2 or not all(
                np.isfinite(value) and value > 0. for value in transport_tolerances):
            raise ValueError("transport_tolerances require auditing and two finite positive limits")

    def classify(state, ledger):
        velocity = jnp.maximum(jnp.max(jnp.abs(state.u)), jnp.max(jnp.abs(state.v)))
        eta = jnp.max(jnp.abs(state.eta))
        finite_state = jnp.all(jnp.stack([jnp.all(jnp.isfinite(field)) for field in state]))
        finite_ledger = jnp.all(jnp.stack([jnp.all(jnp.isfinite(values)) for values in ledger.values()]))
        failure = jnp.where(~finite_state, 1, jnp.where(~finite_ledger, 2,
                            jnp.where(velocity >= MAX_U_BOUND, 3, jnp.where(eta >= ETA_BLOWUP_M, 4, 0))))
        if transport_tolerances is not None:
            continuity, matching = transport_tolerances
            metrics = ledger["transport_consistency_max"]
            transport_failure = jnp.where(jnp.abs(metrics[0]) > continuity, 5,
                                          jnp.where(jnp.abs(metrics[2]) > matching, 6, 0))
            failure = jnp.where(failure == 0, transport_failure, failure)
        return velocity, eta, finite_state & finite_ledger, failure

    @jax.jit
    def advance(current, count):
        initial_velocity, initial_eta, initial_finite, initial_failure = classify(current, zero_budget)
        initial = (current, initial_velocity, initial_eta, initial_finite, zero_budget,
                   jnp.asarray(0), jnp.asarray(0), initial_failure, current, zero_budget)

        def attempt(carry):
            previous, peak_velocity, peak_eta, finite, totals, accepted, attempted, _, rejected, rejected_ledger = carry
            if audited:
                updated, ledger = step(previous)
            else:
                updated, ledger = step(previous), zero_budget
            new_totals = accumulate_budget(totals, ledger) if audited else totals
            velocity, eta, step_finite, failure = classify(updated, ledger)
            total_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(values)) for values in new_totals.values()]))
            failure = jnp.where((failure == 0) & ~total_finite, 2, failure)
            step_finite = step_finite & total_finite
            valid = failure == 0
            accepted_state = jax.tree.map(lambda new, old: jnp.where(valid, new, old), updated, previous)
            accepted_totals = jax.tree.map(lambda new, old: jnp.where(valid, new, old), new_totals, totals)
            rejected = jax.tree.map(lambda new, old: jnp.where(valid, old, new), updated, rejected)
            rejected_ledger = jax.tree.map(lambda new, old: jnp.where(valid, old, new), ledger, rejected_ledger)
            return (accepted_state, jnp.maximum(peak_velocity, velocity), jnp.maximum(peak_eta, eta),
                    finite & step_finite, accepted_totals, accepted + valid.astype(accepted.dtype),
                    attempted + 1, failure, rejected, rejected_ledger)

        def monitored_step(index, carry):
            return jax.lax.cond(carry[7] == 0, attempt, lambda unchanged: unchanged, carry)

        return jax.lax.fori_loop(0, count, monitored_step, initial)

    return advance


def load_initial_fixture(path, grid):
    """Reject mismatched geometry or invalid real initial arrays before stepping."""
    with np.load(path, allow_pickle=False) as saved:
        for name, expected in (("lon", grid.lon), ("lat", grid.lat), ("z", grid.z),
                               ("wet_mask_z", grid.wet_mask_3d)):
            if name not in saved or not np.array_equal(saved[name], expected):
                raise ValueError(f"initial fixture {name} differs from the actual smoke grid")
        shape = (grid.nx, grid.ny, grid.nz)
        fields = []
        for name in ("T_initial", "S_initial"):
            if name not in saved or saved[name].shape != shape or not np.isfinite(saved[name]).all():
                raise ValueError(f"initial fixture {name} must be finite with shape {shape}")
            fields.append(np.asarray(saved[name], dtype=float))
    return tuple(fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=1.)
    parser.add_argument("--resolution", type=float, default=2.)
    parser.add_argument("--dt", type=float, default=600.)
    parser.add_argument("--kappa-bi", type=float, default=0.)
    parser.add_argument("--audit-budget", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--column-geometry", choices=["legacy", "nodal_dual_v1"], default="legacy")
    parser.add_argument("--match-barotropic-transport", action="store_true")
    parser.add_argument("--process-time-scheme", choices=["legacy", "consistent_split_v1", "subcycled_rk2_v2", "symmetric_fast_v3"], default="legacy")
    parser.add_argument("--initial-from", default=None,
                        help="validated T_initial/S_initial with exact lon/lat/z/wet_mask_z; otherwise synthetic")
    parser.add_argument("--ncep-month", type=int, default=None,
                        help="fixed NCEP monthly air/wind index from January 1948; not evolving climate forcing")
    parser.add_argument("--projection-niter", type=int, default=None)
    parser.add_argument("--projection-rtol", type=float, default=None)
    parser.add_argument("--projection-preconditioner", choices=["none", "jacobi"], default="none")
    parser.add_argument("--projection-max-refinements", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--cases", nargs="+", choices=["baseline", "mixed", "ice", "coastal"],
                        default=["baseline", "mixed", "ice", "coastal"])
    parser.add_argument("--bathy", default=DEFAULT_CONFIG.bathymetry_file)
    parser.add_argument("--out", default="results/debug_verification/smoke.json")
    args = parser.parse_args()
    if not all(np.isfinite(value) and value > 0. for value in (args.days, args.dt, args.resolution)):
        parser.error("days, dt and resolution must be finite and positive")
    if not np.isfinite(args.kappa_bi) or args.kappa_bi < 0.:
        parser.error("kappa-bi must be finite and nonnegative")
    if args.match_barotropic_transport and args.column_geometry != "nodal_dual_v1":
        parser.error("--match-barotropic-transport requires --column-geometry nodal_dual_v1")
    if args.process_time_scheme != "legacy" and not args.match_barotropic_transport:
        parser.error(f"--process-time-scheme {args.process_time_scheme} requires --match-barotropic-transport")
    if args.ncep_month is not None and args.ncep_month < 0:
        parser.error("--ncep-month must be an explicit nonnegative month index")
    grid, physics, initial_temperature, initial_salinity, atmosphere, forcing = make_smoke_fixture(
        args.resolution, args.bathy, args.kappa_bi)
    additional_inputs = {}
    if args.initial_from:
        initial_temperature, initial_salinity = load_initial_fixture(args.initial_from, grid)
        additional_inputs[str(Path(args.initial_from).resolve())] = hashlib.sha256(Path(args.initial_from).read_bytes()).hexdigest()
    if args.ncep_month is not None:
        from air_reanalysis import CACHE_DIR as AIR_CACHE
        from air_reanalysis import load_monthly_mean_air_temp
        from wind_reanalysis import CACHE_DIR as WIND_CACHE
        from wind_reanalysis import real_wind_forcing

        year = 1948 + args.ncep_month // 12
        atmosphere = load_monthly_mean_air_temp(grid, year=year)[args.ncep_month % 12]
        stress_x, stress_y = real_wind_forcing(grid, month_idx=args.ncep_month)
        forcing = (stress_x, stress_y, np.zeros_like(stress_x))
        for path in (Path(AIR_CACHE) / f"air_2m_monthly_{year}.npz", Path(WIND_CACHE) / f"monthly_mean_{args.ncep_month}.npz"):
            additional_inputs[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    nx, ny = grid.nx, grid.ny
    total_steps = int(np.ceil(args.days * 86400. / args.dt))
    batch_steps = max(1, int(21600. / args.dt))
    results = []
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src").glob("*.py")) + [Path(__file__).resolve()]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "numerical_smoke_not_climate_or_forecast_skill", "status": "running",
        "arguments": vars(args), "grid": [nx, ny, grid.nz], "physics": asdict(physics),
        "input_kinds": {"initial": "external_validated_fixture" if args.initial_from else "synthetic_analytic",
                        "atmosphere_wind": "fixed_monthly_NCEP_snapshot" if args.ncep_month is not None else "synthetic_analytic",
                        "surface_heat": "simplified_bulk_lambda80_Q_prescribed_zero_not_full_flux_bundle"},
        "provenance": {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines(),
            "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources},
            "bathymetry_sha256": hashlib.sha256(Path(args.bathy).read_bytes()).hexdigest(),
            "additional_input_sha256": additional_inputs,
            "jax_version": jax.__version__, "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "cases": results,
    }

    def save_report():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save_report()
    for case in args.cases:
        coast_mask = None
        if case == "coastal":
            distance = land_distance_from_land_mask(grid.land_mask)
            coast_mask = grid.ocean_mask & (distance <= 3.)
        step, initialize, _, params, _ = make_solver_global(
            grid, physics, args.dt, forcing=forcing, T_atm=atmosphere, lambda_bulk=80.,
            T_init=initial_temperature, S_init=initial_salinity,
            mode_split=True, dt_bt=50., nu_nsub='cfl', use_scan=True, dtype=args.dtype,
            conservative_kv=True, localize_conv=True, project_adv_vel=True, fct_adv=True,
            projection_niter=args.projection_niter, projection_rtol=args.projection_rtol,
            projection_preconditioner=args.projection_preconditioner,
            projection_max_refinements=args.projection_max_refinements,
            mixed_layer_depth_m=20. if case in {"mixed", "ice"} else None,
            dynamic_ice=case == "ice", coastal_kappa_h_mask=coast_mask,
            coastal_kappa_h=500. if coast_mask is not None else 0., return_params=True,
            column_geometry=args.column_geometry, match_barotropic_transport=args.match_barotropic_transport,
            process_time_scheme=args.process_time_scheme)
        state = initialize(T_init=initial_temperature, S_init=initial_salinity)
        audited_step = make_budget_step(params) if args.audit_budget else None
        zero_budget = empty_budget()
        accumulated_budget = {name: np.zeros_like(np.asarray(values)) for name, values in zero_budget.items()}

        advance = make_monitored_advance(audited_step if args.audit_budget else step, zero_budget, audited=args.audit_budget)

        started = time.perf_counter()
        records = []
        completed = 0
        passed = True
        result = {"case": case, "status": "running", "records": records,
                  "column_projection": projection_config(params)}
        results.append(result)
        jax.block_until_ready(state)
        lower_started = time.perf_counter()
        lowered = advance.lower(state, jnp.asarray(batch_steps, dtype=jnp.int32))
        compile_started = time.perf_counter()
        compiled_advance = lowered.compile()
        integration_started = time.perf_counter()
        result["timing"] = {"trace_lower_seconds": compile_started - lower_started,
                            "compile_seconds": integration_started - compile_started,
                            "integration_seconds": 0., "scope": "audited_monitored_batches_no_report_IO"}
        result["effective_parameters"] = {}
        parameter_arrays = {}
        for name, value in params._asdict().items():
            if value is None or isinstance(value, (str, bool, int, float)):
                result["effective_parameters"][name] = value
            else:
                array = np.asarray(value)
                parameter_arrays[name] = array
                result["effective_parameters"][name] = {
                    "shape": list(array.shape), "dtype": str(array.dtype),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        parameter_path = output.with_name(f"{output.stem}_{case}_parameters.npz")
        np.savez_compressed(parameter_path, **parameter_arrays)
        result["parameter_arrays"] = str(parameter_path)
        initial_state_path = output.with_name(f"{output.stem}_{case}_initial_state.npz")
        np.savez_compressed(initial_state_path, **{name: np.asarray(value) for name, value in state._asdict().items()})
        result["initial_state_path"] = str(initial_state_path)
        while completed < total_steps:
            count = min(batch_steps, total_steps - completed)
            batch_started = time.perf_counter()
            monitored = compiled_advance(state, jnp.asarray(count, dtype=jnp.int32))
            jax.block_until_ready(monitored)
            state, peak_velocity, peak_eta, finite, ledger, accepted, attempted, failure, rejected, rejected_ledger = monitored
            result["timing"]["integration_seconds"] += time.perf_counter() - batch_started
            if args.audit_budget:
                for name, values in ledger.items():
                    if name in MAXIMUM_BUDGET_FIELDS:
                        accumulated_budget[name] = np.maximum(accumulated_budget[name], np.asarray(values))
                    else:
                        accumulated_budget[name] += np.asarray(values)
            batch_start = completed
            completed += int(accepted)
            maximum_velocity = float(jnp.maximum(jnp.max(jnp.abs(state.u)), jnp.max(jnp.abs(state.v))))
            maximum_eta = float(jnp.max(jnp.abs(state.eta)))
            peak_velocity_value, peak_eta_value = float(peak_velocity), float(peak_eta)
            passed = int(failure) == 0
            record = {"day": completed * args.dt / 86400., "max_velocity": maximum_velocity,
                      "max_eta": maximum_eta, "max_ice": float(jnp.max(state.ice)), "finite": bool(finite),
                      "batch_peak_velocity": peak_velocity_value if np.isfinite(peak_velocity_value) else None,
                      "batch_peak_eta": peak_eta_value if np.isfinite(peak_eta_value) else None,
                      "accepted_steps_in_batch": int(accepted), "attempted_steps_in_batch": int(attempted)}
            if not passed:
                reasons = {1: "nonfinite_state", 2: "nonfinite_budget", 3: "velocity_stop", 4: "eta_stop"}
                failure_step = batch_start + int(attempted)
                rejected_path = output.with_name(f"{output.stem}_{case}_first_rejected_state.npz")
                rejected_ledger_path = output.with_name(f"{output.stem}_{case}_first_rejected_ledger.npz")
                np.savez_compressed(rejected_path, **{name: np.asarray(value) for name, value in rejected._asdict().items()})
                np.savez_compressed(rejected_ledger_path, **{name: np.asarray(value) for name, value in rejected_ledger.items()})
                result["first_failure"] = {"step": failure_step, "day": failure_step * args.dt / 86400.,
                                           "reason": reasons[int(failure)], "incoming_state_failure": int(attempted) == 0,
                                           "last_accepted_step": completed, "rejected_state_path": str(rejected_path),
                                           "rejected_ledger_path": str(rejected_ledger_path)}
            if args.audit_budget:
                record["budget_residual"] = np.asarray(ledger["budget_residual"]).tolist()
                record["absolute_budget_residual"] = np.asarray(ledger["absolute_budget_residual"]).tolist()
                record["decomposition_residual"] = np.asarray(ledger["decomposition_residual"]).tolist()
                record["advection_boundary_residual"] = np.asarray(ledger["advection_boundary_residual"]).tolist()
                record["nonlinear_accounting_residual"] = np.asarray(ledger["nonlinear_accounting_residual"]).tolist()
                record["projection_transport_norm_squared"] = np.asarray(ledger["projection_transport_norm_squared"]).tolist()
                record["projection_relative_residual_max"] = float(ledger["projection_relative_residual_max"])
                record["transport_consistency_max"] = np.asarray(ledger["transport_consistency_max"]).tolist()
                record["transport_audited_steps"] = float(ledger["transport_audited_steps"])
            records.append(record)
            print(json.dumps({"case": case, **record}), flush=True)
            save_report()
            if not passed:
                break
        result.update(status="complete" if passed else "failed", stability_pass=bool(passed), steps=completed,
                      wall_seconds=time.perf_counter() - started,
                      final_budget=compute_budget_diagnostics(state, grid, column_geometry=args.column_geometry).as_dict(),
                      final_budget_geometry=args.column_geometry,
                      state_semantics="z_level_point_samples_not_moving_layer_inventory",
                      failure_detection_scope="first_rejected_step_stop_last_accepted_state_separate_rejected_ledger")
        final_state_path = output.with_name(f"{output.stem}_{case}_final_state.npz")
        np.savez_compressed(final_state_path, **{name: np.asarray(value) for name, value in state._asdict().items()})
        result["final_state_path"] = str(final_state_path)
        result["final_state_is_last_accepted"] = True
        if args.audit_budget:
            result["stage_budget"] = {"metrics": METRIC_NAMES, "stages": STAGE_NAMES, "sources": SOURCE_NAMES,
                                      "nonlinear_processes": NONLINEAR_PROCESS_NAMES,
                                      "transport_metrics": TRANSPORT_METRIC_NAMES,
                                      "projection_transport_norm_squared_units": "m4/s2; area-weighted before/after sums",
                                      "scope": ("fixed_node_proxy_not_complete_moving_volume_budget" if args.column_geometry == "legacy"
                                                else "static_nodal_reference_not_complete_moving_volume_budget"),
                                      "values": {name: values.tolist() for name, values in accumulated_budget.items()}}
        save_report()
    report["status"] = "complete" if all(result["stability_pass"] for result in results) else "failed"
    save_report()
    if not all(result["stability_pass"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
