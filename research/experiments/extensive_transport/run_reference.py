"""Real-ETOPO cell-content/coupling qualification, not full ocean integration."""
import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from ocean_solver.provenance.archives import current_source_files

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from zhenmode_research.candidates.fv.barotropic import coupled_surface_step
from ocean_solver.config.definitions import DEFAULT_CONFIG, GlobalGridConfig
from zhenmode_research.candidates.fv.geometry import ExtensiveState, build_geometry, surface_height, surface_volume
from ocean_solver.io.grid import global_grid_dims, make_global_grid

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bathy", default=DEFAULT_CONFIG.bathymetry_file)
    parser.add_argument("--out", default="results/industrial_alignment/extensive_transport_current.json")
    parser.add_argument("--inventory-policy", choices=("same", "float64"), default="same")
    parser.add_argument("--transport-scheme", choices=("donor", "centered_fct"), default="donor")
    args = parser.parse_args()
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "src" / name for name in ("finite_volume.py", "bounded_transport.py", "barotropic_transport.py", "grid.py", "config.py")]
    sources += [Path(__file__).resolve(), Path(__file__).with_name("protocol.md"), ROOT / "tests/test_extensive_transport.py"]
    sources += [Path(__file__).with_name("precision_protocol.md")]
    sources += [Path(__file__).parent.parent / "bounded_extensive_transport" / name
                for name in ("protocol.md", "selection.md")]
    sources += [ROOT / "tests/test_bounded_extensive_transport.py"]
    report = {"scope": "physical_transport_and_linear_wave_reference_not_full_ocean_or_climate",
              "status": "running", "inventory_policy": args.inventory_policy,
              "transport_scheme": args.transport_scheme, "provenance": {
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
                  "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in current_source_files(ROOT, sources).items()},
                  "bathymetry_sha256": hashlib.sha256(Path(args.bathy).read_bytes()).hexdigest(),
                  "jax_version": jax.__version__, "backend": jax.default_backend(),
                  "devices": [str(device) for device in jax.devices()]}, "geometries": [], "runs": []}

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    nx, ny = global_grid_dims(2., 65., remap="area")
    config = replace(GlobalGridConfig(), resolution=2., lat_max=65., nx=nx, ny=ny)
    for label, smooth, floor in (("prior_smoothed", 80, 500.), ("unsmoothed", 0, 10.)):
        grid = make_global_grid(config, args.bathy, smooth_passes=smooth, min_depth=floor, remap="area")
        lon_edges = np.r_[grid.lon - 1., grid.lon[-1] + 1.]
        lat_edges = np.r_[grid.lat - 1., grid.lat[-1] + 1.]
        interfaces = np.r_[-grid.z, 8000.]
        geometry = build_geometry(lon_edges, lat_edges, interfaces, grid.depth)
        depth_error = float(np.max(np.abs(np.sum(geometry.thickness, axis=-1) - grid.depth)))
        if depth_error > 1e-13 * max(1., float(np.max(grid.depth))):
            raise ValueError(f"physical depth identity failed: {depth_error}")
        volume = geometry.thickness * geometry.area[..., None]
        report["geometries"].append({"label": label, "shape": list(volume.shape),
            "longitude_edges": [float(lon_edges[0]), float(lon_edges[-1])],
            "latitude_edges": [float(lat_edges[0]), float(lat_edges[-1])],
            "interfaces_m": interfaces.tolist(), "smooth_passes": smooth,
            "minimum_depth_land_threshold_m": floor, "maximum_depth_m": float(np.max(grid.depth)),
            "wet_columns": int(np.sum(grid.depth > 0.)),
            "physical_volume_m3": float(np.sum(volume)), "maximum_column_depth_error_m": depth_error,
            "input_sha256": hashlib.sha256(grid.depth.tobytes() + lon_edges.tobytes() + lat_edges.tobytes() + interfaces.tobytes()).hexdigest()})
        phase = 2. * np.pi * np.arange(nx)[:, None] / nx
        for dtype_name in ("float64", "float32"):
            dtype = getattr(jnp, dtype_name)
            inventory_dtype = jnp.float64 if args.inventory_policy == "float64" else dtype
            device_geometry = jax.tree_util.tree_map(lambda value: jnp.asarray(value, dtype), geometry)
            tolerance = 1e-12 if dtype_name == "float64" else 2e-6
            for constant in (True, False):
                for dt in (60., 10., 1.):
                    eta_requested = jnp.asarray(np.broadcast_to(.2 * np.cos(phase), (nx, ny)) * grid.ocean_mask, dtype)
                    initial_volume = surface_volume(device_geometry, eta_requested.astype(inventory_dtype))
                    eta = surface_height(device_geometry, initial_volume).astype(dtype)
                    concentration = np.ones(volume.shape + (2,)) * [12., 35.]
                    if not constant:
                        concentration[..., 0] += np.sin(phase)[..., None] * np.exp(-geometry.center_depth / 500.)
                        concentration[..., 1] += .1 * np.sin(np.radians(grid.lat))[None, :, None]
                    state = ExtensiveState(initial_volume, initial_volume[..., None] * jnp.asarray(concentration, dtype).astype(inventory_dtype))
                    initial_content = np.asarray(state.content, dtype=np.float64)
                    wet = volume > 0.
                    initial_concentration = initial_content[wet] / np.asarray(state.volume, dtype=np.float64)[wet, None]
                    east = jnp.asarray(np.broadcast_to(.02 * np.sin(phase), (nx, ny)), dtype)
                    east = jnp.where(jnp.sum(device_geometry.east_area, axis=-1) > 0., east, 0.)
                    north = jnp.zeros_like(east)
                    layer_east = jnp.zeros(state.volume.shape, dtype)
                    layer_north = jnp.zeros(state.volume.shape, dtype)

                    @jax.jit
                    def step(current, height, velocity_east, velocity_north):
                        return coupled_surface_step(device_geometry, current, height, velocity_east,
                                                    velocity_north, layer_east, layer_north, dt / 4., 4,
                                                    inventory_precision=None if args.inventory_policy == "same" else "float64",
                                                    transport_scheme=args.transport_scheme)

                    run = {"geometry": label, "dtype": dtype_name, "constant": constant,
                           "inventory_dtype": str(initial_volume.dtype),
                           "inventory_initialization": "momentum_quantized_eta_and_concentration_promoted_before_V_N_construction",
                           "dt_seconds": dt, "barotropic_substeps": 4, "requested_steps": 100,
                           "status": "running", "maximum_surface_identity_error": 0.,
                           "maximum_outflow_fraction": 0., "maximum_gravity_cfl_bound": 0.,
                           "surface_diagnosis_arithmetic": "float64_division_subtraction_returning_state_dtype",
                           "eta_representation": "derived_from_primary_top_volume_not_independent_redundant_state"}
                    report["runs"].append(run)
                    for index in range(100):
                        result = step(state, eta, east, north)
                        run["completed_steps"] = index
                        run["maximum_surface_identity_error"] = max(run["maximum_surface_identity_error"], float(result.surface_error))
                        run["maximum_outflow_fraction"] = max(run["maximum_outflow_fraction"], float(result.transport.max_outflow_fraction))
                        run["maximum_gravity_cfl_bound"] = max(run["maximum_gravity_cfl_bound"], float(result.barotropic.gravity_cfl_bound))
                        if not bool(result.valid):
                            run.update(status="FAIL", failed_step=index + 1,
                                       barotropic_valid=bool(result.barotropic.valid),
                                       transport_valid=bool(result.transport.valid))
                            break
                        state = result.transport.state
                        eta = surface_height(device_geometry, state.volume).astype(dtype)
                        east, north = result.barotropic.east_velocity, result.barotropic.north_velocity
                        run["completed_steps"] = index + 1
                    final_content = np.asarray(state.content, dtype=np.float64)
                    final_volume = np.asarray(state.volume, dtype=np.float64)
                    volume_delta = final_volume - np.asarray(initial_volume, dtype=np.float64)
                    snapshot = output.with_name(output.stem + f"_{label}_{dtype_name}_{constant}_dt{dt:g}.npz")
                    np.savez(snapshot, initial_volume=np.asarray(initial_volume), initial_content=initial_content,
                             final_volume=final_volume, final_content=final_content,
                             physical_depth=grid.depth, area=np.asarray(device_geometry.area),
                             thickness=np.asarray(device_geometry.thickness), eta=np.asarray(eta))
                    run["snapshot_path"] = str(snapshot.relative_to(ROOT))
                    run["snapshot_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
                    run["volume_budget_delta_m3"] = float(np.sum(volume_delta))
                    run["maximum_column_volume_change_m3"] = float(np.max(np.abs(np.sum(volume_delta, axis=-1))))
                    run["volume_budget_equivalent_mean_surface_m"] = float(np.sum(volume_delta) / np.sum(geometry.area[grid.depth > 0.]))
                    run["initial_tracer_extrema"] = [initial_concentration.min(axis=0).tolist(), initial_concentration.max(axis=0).tolist()]
                    totals_before = np.sum(initial_content, axis=(0, 1, 2))
                    totals_after = np.sum(final_content, axis=(0, 1, 2))
                    denominator = np.sum(np.abs(initial_content), axis=(0, 1, 2))
                    budget_error = np.abs(totals_after - totals_before) / denominator
                    final_concentration = final_content[wet] / final_volume[wet, None]
                    constant_error = float(np.max(np.abs(final_concentration - initial_concentration) / np.maximum(np.abs(initial_concentration), 1.)))
                    bounded = bool(np.all(final_concentration.min(axis=0) >= initial_concentration.min(axis=0) - tolerance)
                                   and np.all(final_concentration.max(axis=0) <= initial_concentration.max(axis=0) + tolerance))
                    passed = (run["completed_steps"] == 100 and np.all(budget_error <= tolerance)
                              and (constant_error <= tolerance if constant else bounded))
                    run.update(status="PASS" if passed else "FAIL", content_budget_relative_error=budget_error.tolist(),
                               constant_relative_error=constant_error if constant else None, bounded=bounded,
                               maximum_eta_m=float(np.max(np.abs(eta))), minimum_wet_volume_m3=float(np.min(final_volume[wet])),
                               tracer_extrema=[final_concentration.min(axis=0).tolist(), final_concentration.max(axis=0).tolist()],
                               model_seconds=run["completed_steps"] * dt)
                    save()
                    print(f"{label}/{dtype_name}/constant={constant}/dt={dt}: {run['status']}, steps={run['completed_steps']}, budget={budget_error.tolist()}, surface={run['maximum_surface_identity_error']:.3e}", flush=True)
                    timestep_rejected = (run["maximum_outflow_fraction"] > 1.
                                         or run["maximum_gravity_cfl_bound"] > 2.)
                    if passed or not timestep_rejected:
                        break
    groups = {(run["geometry"], run["dtype"], run["constant"]) for run in report["runs"] if run["status"] == "PASS"}
    report["status"] = "PASS" if len(groups) == 8 else "FAIL"
    report["qualification"] = "component_only_legacy_production_not_migrated"
    save()
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
