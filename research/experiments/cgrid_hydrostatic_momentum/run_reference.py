"""Eight active density/momentum references, not production or climate qualification."""
import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from cgrid_momentum import LayerState, linear_momentum_surface_step
from config import DEFAULT_CONFIG, GlobalGridConfig
from finite_volume import ExtensiveState, _physical_surface_height, build_geometry, surface_volume
from grid import global_grid_dims, make_global_grid
from wet_fluxes import evaluate_wet_flux
from physical_velocity import evaluate_physical_velocity

jax.config.update("jax_enable_x64", True)

METRICS = ("pressure_max_m_s2", "rotation_energy_relative", "rotation_solve_relative",
           "surface_error_m", "surface_gate_m", "outflow_fraction", "gravity_cfl_bound",
           "speed_max_m_s", "layer_shear_max_m_s", "constant_error", "bound_excursion",
        "lower_height_relative_error", "eta_max_m", "wet_trace_bottom_relative", "dual_commutation_relative",
        "physical_bulk_continuity_relative")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bathy", default=DEFAULT_CONFIG.bathymetry_file)
    parser.add_argument("--out", default="results/industrial_alignment/cgrid_physical_velocity_reference.json")
    args = parser.parse_args()
    output = ROOT / args.out
    if output.exists():
        raise FileExistsError("retain previous evidence; choose a new --out")
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "src" / name for name in ("finite_volume.py", "bounded_transport.py", "barotropic_transport.py",
                                               "cgrid_momentum.py", "wet_fluxes.py", "physical_velocity.py", "grid.py", "config.py")]
    sources += [Path(__file__), Path(__file__).with_name("protocol.md"), Path(__file__).with_name("selection.md"),
                Path(__file__).with_name("overlap_protocol.md"),
                Path(__file__).with_name("pressure_work_protocol.md"),
                Path(__file__).with_name("dual_mass_protocol.md"),
                Path(__file__).with_name("wet_trace_protocol.md"),
                Path(__file__).with_name("metric_dual_protocol.md"),
                Path(__file__).with_name("physical_velocity_protocol.md"),
                ROOT / "tests/test_cgrid_hydrostatic_momentum.py", ROOT / "tests/test_cgrid_pressure_work.py",
                ROOT / "tests/test_momentum_shared_flux.py", ROOT / "tests/test_wet_flux_reconstruction.py",
                ROOT / "tests/test_wet_flux_metrics.py", ROOT / "tests/test_physical_velocity.py"]
    report = {"scope": "frozen_pressure_linear_3d_momentum_active_density_fct_not_full_ocean",
              "rotation_formulation": "physical_wet_dual_rectangles_and_common_overlap",
              "pressure_formulation": "shared_face_force_over_physical_dual_mass",
              "transport_flux_contract": "actual_matched_layer_fluxes_from_fast_substep_mean",
              "wet_trace_contract": "latitude_arc_wet_traces_paired_interior_flux_frozen_geometry_not_velocity",
              "dual_transport_contract": "all_wet_half_prisms_metric_integrated_q_not_force_mass_or_nonlinear_momentum",
              "physical_velocity_contract": "mean_q_frozen_eulerian_lift_signed_top_boundary_source_not_full_ale",
              "status": "running", "reference_coefficients": [1., 1e-3, 1e-7],
              "provenance": {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                             "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
                             "source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                             "bathymetry_sha256": hashlib.sha256(Path(args.bathy).read_bytes()).hexdigest(),
                             "jax_version": jax.__version__, "backend": jax.default_backend(),
                             "devices": [str(device) for device in jax.devices()]}, "geometries": [], "runs": []}

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    nx, ny = global_grid_dims(2., 65., remap="area")
    config = replace(GlobalGridConfig(), resolution=2., lat_max=65., nx=nx, ny=ny)
    coefficients = jnp.asarray(report["reference_coefficients"])
    for label, smooth, floor in (("prior_smoothed", 80, 500.), ("unsmoothed", 0, 10.)):
        grid = make_global_grid(config, args.bathy, smooth_passes=smooth, min_depth=floor, remap="area")
        lon_edges = np.r_[grid.lon - 1., grid.lon[-1] + 1.]
        lat_edges = np.r_[grid.lat - 1., grid.lat[-1] + 1.]
        interfaces = np.r_[-grid.z, 8000.]
        geometry = build_geometry(lon_edges, lat_edges, interfaces, grid.depth)
        device = jax.tree_util.tree_map(jnp.asarray, geometry)
        volume = surface_volume(device, jnp.zeros((nx, ny), jnp.float64))
        wet = volume > 0.
        depth_error = float(np.max(np.abs(np.sum(geometry.thickness, axis=-1) - grid.depth)))
        if depth_error > 1e-13 * max(1., float(np.max(grid.depth))):
            raise ValueError("physical layer depth identity failed")
        report["geometries"].append({"label": label, "shape": list(volume.shape), "smooth_passes": smooth,
                                    "minimum_depth_land_threshold_m": floor, "latitude_edges": [float(lat_edges[0]), float(lat_edges[-1])],
                                    "maximum_column_depth_error_m": depth_error,
                                    "input_sha256": hashlib.sha256(grid.depth.tobytes() + lon_edges.tobytes() + lat_edges.tobytes() + interfaces.tobytes()).hexdigest()})
        center, height = device.center_depth, device.thickness
        background = coefficients[0] + coefficients[1] * center + coefficients[2] * (center ** 2 + height ** 2 / 12.)
        phase = jnp.radians(jnp.asarray(grid.lon))[:, None, None]
        for dtype_name in ("float64", "float32"):
            dtype = getattr(jnp, dtype_name)
            for disturbed in (False, True):
                density = background + (.1 * jnp.sin(phase) if disturbed else 0.)
                concentration = jnp.stack((density, jnp.full_like(density, 35.)), axis=-1)
                initial = LayerState(ExtensiveState(volume, volume[..., None] * concentration),
                                     jnp.zeros(volume.shape, dtype), jnp.zeros(volume.shape, dtype))
                low = jnp.min(jnp.where(wet[..., None], concentration, jnp.inf), axis=(0, 1, 2))
                high = jnp.max(jnp.where(wet[..., None], concentration, -jnp.inf), axis=(0, 1, 2))

                @jax.jit
                def step(state):
                    density_now = state.inventory.content[..., 0] / jnp.where(state.inventory.volume > 0., state.inventory.volume, 1.)
                    result = linear_momentum_surface_step(device, state, density_now, 60., 4, reference=coefficients)
                    final = result.state
                    concentration_now = final.inventory.content / jnp.where(final.inventory.volume > 0., final.inventory.volume, 1.)[..., None]
                    eta_before = _physical_surface_height(device, state.inventory.volume)
                    eta_after = _physical_surface_height(device, final.inventory.volume)
                    eta_scale = jnp.maximum(jnp.max(jnp.abs(eta_before)), jnp.max(jnp.abs(result.barotropic.eta)))
                    pressure_max = jnp.maximum(jnp.max(jnp.abs(result.pressure.east)), jnp.max(jnp.abs(result.pressure.north)))
                    speed = jnp.maximum(jnp.max(jnp.abs(final.east_velocity)), jnp.max(jnp.abs(final.north_velocity)))
                    shear = jnp.maximum(jnp.max(jnp.ptp(final.east_velocity, axis=-1)), jnp.max(jnp.ptp(final.north_velocity, axis=-1)))
                    constant = jnp.max(jnp.where(wet, jnp.abs(concentration_now[..., 1] - 35.), 0.))
                    excursion = jnp.max(jnp.where(wet[..., None], jnp.maximum(low - concentration_now, concentration_now - high), 0.))
                    lower_error = jnp.max(jnp.abs(final.inventory.volume[..., 1:] / device.area[..., None] - device.thickness[..., 1:])
                                          / jnp.maximum(device.thickness[..., 1:], 1.))
                    metrics = jnp.array([pressure_max, result.rotation_error, result.rotation_residual, result.surface_error,
                                         1e-12 + 1e-12 * eta_scale, result.transport.max_outflow_fraction,
                                         result.barotropic.gravity_cfl_bound, speed, shear, constant, jnp.maximum(excursion, 0.), lower_error,
                                         jnp.max(jnp.abs(eta_after)), result.flux_reconstruction.bottom_closure_relative,
                                         result.dual_transport.commutation_relative, result.physical_velocity.bulk_continuity_relative])
                    flags = jnp.array([result.valid, result.pressure.valid, result.barotropic.valid, result.transport.valid])
                    reconstruction = result.flux_reconstruction
                    top_point = evaluate_wet_flux(reconstruction, .37, .61, reconstruction.top)
                    bottom_point = evaluate_wet_flux(reconstruction, .37, .61, reconstruction.top + reconstruction.height)
                    middle_point = evaluate_wet_flux(reconstruction, .37, .61, reconstruction.top + .38123 * reconstruction.height)
                    trace_points = (top_point.vertical, bottom_point.vertical, middle_point.east_per_depth,
                                    middle_point.north_per_depth, middle_point.vertical)
                    physical_points = tuple(evaluate_physical_velocity(result.physical_velocity, .37, .61, depth)
                                            for depth in (reconstruction.top, reconstruction.top + reconstruction.height,
                                                          reconstruction.top + .38123 * reconstruction.height))
                    return final, metrics, flags, result.fluxes, result.barotropic.mean_east, result.barotropic.mean_north, reconstruction, trace_points, result.dual_transport, result.physical_velocity, physical_points

                run = {"geometry": label, "velocity_dtype": dtype_name, "disturbed": disturbed,
                       "inventory_dtype": "float64", "dt_seconds": 60., "barotropic_substeps": 4,
                       "completed_steps": 0, "status": "running", "history": []}
                report["runs"].append(run)
                save()
                state = initial
                for iteration in range(100):
                    previous_volume = state.inventory.volume
                    candidate, metric_array, flags, actual_fluxes, mean_east, mean_north, reconstruction, trace_points, dual, physical, physical_points = step(state)
                    values, accepted = np.asarray(metric_array), np.asarray(flags)
                    row = {name: float(value) if np.isfinite(value) else None for name, value in zip(METRICS, values)}
                    row["stage_valid"] = accepted.tolist()
                    row["step"] = iteration + 1
                    row["wet_trace_valid"] = bool(reconstruction.valid)
                    row["dual_transport_valid"] = bool(dual.valid)
                    row["physical_velocity_valid"] = bool(physical.valid)
                    energy_gate = 2e-6 if dtype_name == "float32" else 1e-12
                    passed = (np.all(np.isfinite(values)) and np.all(accepted)
                              and row["surface_error_m"] <= row["surface_gate_m"]
                              and row["rotation_energy_relative"] <= energy_gate and row["rotation_solve_relative"] <= 1e-12
                              and row["constant_error"] <= 1e-12 and row["bound_excursion"] <= 1e-12
                              and row["lower_height_relative_error"] <= 1e-12
                              and row["wet_trace_valid"] and row["wet_trace_bottom_relative"] <= 1e-12 + 64. * np.finfo(np.float64).eps
                              and row["dual_transport_valid"] and row["dual_commutation_relative"] <= 1e-12 + 64. * np.finfo(np.float64).eps
                              and row["physical_velocity_valid"] and row["physical_bulk_continuity_relative"] <= 1e-12 + 64. * np.finfo(np.float64).eps
                              and (disturbed or row["pressure_max_m_s2"] <= 1e-12))
                    run["history"].append(row)
                    if not passed:
                        run["rejected_step"] = iteration + 1
                        rejected_path = output.with_name(f"{output.stem}_{label}_{dtype_name}_{int(disturbed)}_rejected.npz")
                        np.savez(rejected_path, volume=np.asarray(candidate.inventory.volume), content=np.asarray(candidate.inventory.content),
                                 east_velocity=np.asarray(candidate.east_velocity), north_velocity=np.asarray(candidate.north_velocity))
                        run["rejected_snapshot"] = rejected_path.relative_to(ROOT).as_posix()
                        run["rejected_snapshot_sha256"] = hashlib.sha256(rejected_path.read_bytes()).hexdigest()
                        break
                    state = candidate
                    run["completed_steps"] = iteration + 1
                initial_content, final_content = np.asarray(initial.inventory.content), np.asarray(state.inventory.content)
                budget = np.abs(np.sum(final_content - initial_content, axis=(0, 1, 2))) / np.sum(np.abs(initial_content), axis=(0, 1, 2))
                final_volume = np.asarray(state.inventory.volume)
                delta_volume = float(np.sum(final_volume - np.asarray(volume)))
                activity = max(row["speed_max_m_s"] or 0. for row in run["history"])
                shear = max(row["layer_shear_max_m_s"] or 0. for row in run["history"])
                passed = run["completed_steps"] == 100 and np.all(budget <= 1e-12) and (not disturbed or (activity > 1e-9 and shear > 0.))
                snapshot = output.with_name(f"{output.stem}_{label}_{dtype_name}_{int(disturbed)}.npz")
                physical_fields = {f"last_physical_{location}_{name}": np.asarray(field)
                                   for location, point in zip(("top", "bottom", "middle"), physical_points)
                                   for name, field in zip(point._fields, point)}
                np.savez(snapshot, initial_volume=np.asarray(volume), initial_content=initial_content,
                         final_volume=final_volume, final_content=final_content,
                         initial_east_velocity=np.asarray(initial.east_velocity), initial_north_velocity=np.asarray(initial.north_velocity),
                         final_east_velocity=np.asarray(state.east_velocity), final_north_velocity=np.asarray(state.north_velocity),
                         last_previous_volume=np.asarray(previous_volume), last_east_flux=np.asarray(actual_fluxes.east),
                         last_north_flux=np.asarray(actual_fluxes.north), last_vertical_flux=np.asarray(actual_fluxes.vertical),
                         last_fast_mean_east=np.asarray(mean_east), last_fast_mean_north=np.asarray(mean_north),
                         last_trace_cell_top=np.asarray(reconstruction.top), last_trace_cell_height=np.asarray(reconstruction.height),
                         last_trace_side_top=np.asarray(reconstruction.side_top), last_trace_side_bottom=np.asarray(reconstruction.side_bottom),
                         last_trace_side_density=np.asarray(reconstruction.side_density), last_trace_net_flux=np.asarray(reconstruction.net_flux),
                         last_trace_evaluated_top=np.asarray(trace_points[0]), last_trace_evaluated_bottom=np.asarray(trace_points[1]),
                         last_trace_evaluated_middle_east=np.asarray(trace_points[2]), last_trace_evaluated_middle_north=np.asarray(trace_points[3]),
                         last_trace_evaluated_middle_vertical=np.asarray(trace_points[4]),
                         last_trace_south_latitude=np.asarray(reconstruction.south_latitude), last_trace_north_latitude=np.asarray(reconstruction.north_latitude),
                         last_dual_east_volume=np.asarray(dual.east_volume), last_dual_north_volume=np.asarray(dual.north_volume),
                         last_dual_east_east_flux=np.asarray(dual.east_fluxes.east), last_dual_east_north_flux=np.asarray(dual.east_fluxes.north),
                         last_dual_east_vertical_flux=np.asarray(dual.east_fluxes.vertical), last_dual_north_east_flux=np.asarray(dual.north_fluxes.east),
                         last_dual_north_north_flux=np.asarray(dual.north_fluxes.north), last_dual_north_vertical_flux=np.asarray(dual.north_fluxes.vertical),
                         last_physical_area=np.asarray(physical.area), last_physical_meridional_width=np.asarray(physical.meridional_area_width),
                         last_physical_zonal_arc=np.asarray(physical.zonal_arc), last_physical_surface_downward=np.asarray(physical.surface_downward),
                         last_physical_source_speed=np.asarray(physical.source_speed), last_physical_absolute_lift=np.asarray(physical.absolute_lift),
                         last_volume_source=np.zeros_like(np.asarray(previous_volume)),
                         physical_depth=grid.depth, area=geometry.area, thickness=geometry.thickness, interfaces=interfaces,
                         longitude_edges=lon_edges, latitude_edges=lat_edges, reference_coefficients=np.asarray(coefficients), **physical_fields)
                run.update(status="PASS" if passed else "FAIL", content_budget_relative=budget.tolist(),
                           volume_budget_delta_m3=delta_volume, volume_budget_equivalent_surface_m=delta_volume / np.sum(geometry.area[np.asarray(wet[..., 0])]),
                           maximum_speed_m_s=activity, maximum_shear_m_s=shear,
                           snapshot_path=snapshot.relative_to(ROOT).as_posix(), snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
                save()
                print(f"{label}/{dtype_name}/disturbed={disturbed}: {run['status']} steps={run['completed_steps']} budget={budget.tolist()} speed={activity:.3e}", flush=True)
    report["status"] = "PASS" if len(report["runs"]) == 8 and all(run["status"] == "PASS" for run in report["runs"]) else "FAIL"
    report["qualification"] = "linear_component_only_production_not_migrated"
    save()
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
