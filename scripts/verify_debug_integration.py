"""Real-ETOPO numerical smoke runs with explicitly synthetic initial forcing."""
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
    METRIC_NAMES,
    NONLINEAR_PROCESS_NAMES,
    SOURCE_NAMES,
    STAGE_NAMES,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=1.)
    parser.add_argument("--resolution", type=float, default=2.)
    parser.add_argument("--dt", type=float, default=600.)
    parser.add_argument("--kappa-bi", type=float, default=0.)
    parser.add_argument("--audit-budget", action="store_true")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--projection-niter", type=int, default=None)
    parser.add_argument("--projection-rtol", type=float, default=None)
    parser.add_argument("--projection-preconditioner", choices=["none", "jacobi"], default="none")
    parser.add_argument("--cases", nargs="+", choices=["baseline", "mixed", "ice", "coastal"],
                        default=["baseline", "mixed", "ice", "coastal"])
    parser.add_argument("--bathy", default=DEFAULT_CONFIG.bathymetry_file)
    parser.add_argument("--out", default="results/debug_verification/smoke.json")
    args = parser.parse_args()
    if not all(np.isfinite(value) and value > 0. for value in (args.days, args.dt, args.resolution)):
        parser.error("days, dt and resolution must be finite and positive")
    if not np.isfinite(args.kappa_bi) or args.kappa_bi < 0.:
        parser.error("kappa-bi must be finite and nonnegative")
    grid, physics, initial_temperature, initial_salinity, atmosphere, forcing = make_smoke_fixture(
        args.resolution, args.bathy, args.kappa_bi)
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
        "provenance": {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines(),
            "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources},
            "bathymetry_sha256": hashlib.sha256(Path(args.bathy).read_bytes()).hexdigest(),
            "jax_version": jax.__version__, "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "cases": results,
    }

    def save_report():
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

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
            mixed_layer_depth_m=20. if case in {"mixed", "ice"} else None,
            dynamic_ice=case == "ice", coastal_kappa_h_mask=coast_mask,
            coastal_kappa_h=500. if coast_mask is not None else 0., return_params=True)
        state = initialize(T_init=initial_temperature, S_init=initial_salinity)
        audited_step = make_budget_step(params) if args.audit_budget else None
        zero_budget = empty_budget()
        accumulated_budget = {name: np.zeros_like(np.asarray(values)) for name, values in zero_budget.items()}

        @jax.jit
        def advance(current, count):
            def monitored_step(index, carry):
                previous, peak_velocity, peak_eta, finite, totals = carry
                if args.audit_budget:
                    updated, ledger = audited_step(previous)
                    totals = accumulate_budget(totals, ledger)
                else:
                    updated = step(previous)
                velocity = jnp.maximum(jnp.max(jnp.abs(updated.u)), jnp.max(jnp.abs(updated.v)))
                eta = jnp.max(jnp.abs(updated.eta))
                state_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(field)) for field in updated]))
                if args.audit_budget:
                    state_finite = state_finite & jnp.all(jnp.stack([jnp.all(jnp.isfinite(values)) for values in ledger.values()]))
                return updated, jnp.maximum(peak_velocity, velocity), jnp.maximum(peak_eta, eta), finite & state_finite, totals

            return jax.lax.fori_loop(0, count, monitored_step, (current, jnp.asarray(0., dtype=current.T.dtype),
                                                              jnp.asarray(0., dtype=current.T.dtype), jnp.asarray(True), zero_budget))

        started = time.perf_counter()
        records = []
        completed = 0
        passed = True
        result = {"case": case, "status": "running", "records": records,
                  "column_projection": projection_config(params)}
        results.append(result)
        while completed < total_steps:
            count = min(batch_steps, total_steps - completed)
            state, peak_velocity, peak_eta, finite, ledger = advance(state, count)
            state.T.block_until_ready()
            if args.audit_budget:
                for name, values in ledger.items():
                    if name == "projection_relative_residual_max":
                        accumulated_budget[name] = np.maximum(accumulated_budget[name], np.asarray(values))
                    else:
                        accumulated_budget[name] += np.asarray(values)
            completed += count
            maximum_velocity = float(jnp.maximum(jnp.max(jnp.abs(state.u)), jnp.max(jnp.abs(state.v))))
            maximum_eta = float(jnp.max(jnp.abs(state.eta)))
            passed = bool(finite) and float(peak_velocity) < MAX_U_BOUND and float(peak_eta) < ETA_BLOWUP_M
            record = {"day": completed * args.dt / 86400., "max_velocity": maximum_velocity,
                      "max_eta": maximum_eta, "max_ice": float(jnp.max(state.ice)), "finite": bool(finite),
                      "batch_peak_velocity": float(peak_velocity), "batch_peak_eta": float(peak_eta)}
            if args.audit_budget:
                record["budget_residual"] = np.asarray(ledger["budget_residual"]).tolist()
                record["absolute_budget_residual"] = np.asarray(ledger["absolute_budget_residual"]).tolist()
                record["decomposition_residual"] = np.asarray(ledger["decomposition_residual"]).tolist()
                record["advection_boundary_residual"] = np.asarray(ledger["advection_boundary_residual"]).tolist()
                record["nonlinear_accounting_residual"] = np.asarray(ledger["nonlinear_accounting_residual"]).tolist()
                record["projection_transport_norm_squared"] = np.asarray(ledger["projection_transport_norm_squared"]).tolist()
                record["projection_relative_residual_max"] = float(ledger["projection_relative_residual_max"])
            records.append(record)
            print(json.dumps({"case": case, **record}), flush=True)
            save_report()
            if not passed:
                break
        result.update(status="complete" if passed else "failed", stability_pass=bool(passed), steps=completed,
                      wall_seconds=time.perf_counter() - started,
                      final_budget=compute_budget_diagnostics(state, grid).as_dict())
        if args.audit_budget:
            result["stage_budget"] = {"metrics": METRIC_NAMES, "stages": STAGE_NAMES, "sources": SOURCE_NAMES,
                                      "nonlinear_processes": NONLINEAR_PROCESS_NAMES,
                                      "projection_transport_norm_squared_units": "m4/s2; area-weighted before/after sums",
                                      "scope": "fixed_node_proxy_not_complete_moving_volume_budget",
                                      "values": {name: values.tolist() for name, values in accumulated_budget.items()}}
        save_report()
    report["status"] = "complete" if all(result["stability_pass"] for result in results) else "failed"
    save_report()
    if not all(result["stability_pass"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
