"""Actual paired dynamics on the eight immutable qualified real-grid inputs."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from ocean_solver.provenance.archives import current_source_files

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from zhenmode_research.candidates.fv.momentum import LayerState
from zhenmode_research.candidates.fv.geometry import ExtensiveState, build_geometry
from zhenmode_research.candidates.fv.paired import paired_momentum_surface_step

jax.config.update("jax_enable_x64", True)

METRICS = ("solve_relative", "frozen_energy_work_relative", "cast_work_per_rho0",
           "moving_mass_energy_per_rho0", "surface_error_m", "outflow_fraction",
           "wet_bottom_relative", "dual_commutation_relative", "bulk_continuity_relative",
           "constant_error", "bound_excursion", "speed_max_m_s", "eta_max_m")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="results/industrial_alignment/cgrid_physical_velocity_coordinate_reference.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    input_path, output = ROOT / args.inputs, ROOT / args.out
    if output.exists():
        raise FileExistsError("retain prior evidence; choose a new output")
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    if inputs["status"] != "PASS" or len(inputs["runs"]) != 8:
        raise ValueError("requires eight qualified immutable input groups")
    sources = [ROOT / "src" / name for name in ("finite_volume.py", "bounded_transport.py", "cgrid_momentum.py",
                                               "paired_dynamics.py", "wet_fluxes.py", "physical_velocity.py", "config.py")]
    sources.extend((Path(__file__), Path(__file__).with_name("paired_dynamics_protocol.md"),
                    Path(__file__).with_name("paired_increment_addendum.md"),
                    Path(__file__).with_name("paired_initial_eta_addendum.md"),
                    Path(__file__).with_name("verify_paired_reference.py"),
                    Path(__file__).with_name("analyze_physical_kinetic.py"),
                    ROOT / "tests/test_paired_dynamics.py"))
    files = current_source_files(ROOT, sources)
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    report = {"status": "running", "scope": "actual_frozen_paired_layer_surface_inventory_not_nonlinear_ocean",
              "kinetic_norm": "physical_wet_contact_horizontal_field_L2",
              "time_scheme": "simultaneous_implicit_midpoint_rotation_pressure_surface_mean_q",
              "source_policy": "zero_in_real_reference_signed_sources_in_direct_tests",
              "provenance": {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                             "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
                             "source_sha256": hashes, "input_report": args.inputs,
                             "input_report_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                             "jax_version": jax.__version__, "backend": jax.default_backend(),
                             "devices": [str(device) for device in jax.devices()]}, "runs": []}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    kernels = {}
    for original in inputs["runs"]:
        snapshot_path = ROOT / original["snapshot_path"]
        if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != original["snapshot_sha256"]:
            raise ValueError("immutable real input snapshot hash mismatch")
        with np.load(snapshot_path, allow_pickle=False) as saved:
            depth = saved["physical_depth"]
            geometry = build_geometry(saved["longitude_edges"], saved["latitude_edges"], saved["interfaces"], depth)
            np.testing.assert_allclose(geometry.area, saved["area"], rtol=1e-14)
            np.testing.assert_allclose(geometry.thickness, saved["thickness"], rtol=1e-14)
            device = jax.tree_util.tree_map(jnp.asarray, geometry)
            coefficients = jnp.asarray(saved["reference_coefficients"])
            initial = LayerState(ExtensiveState(jnp.asarray(saved["initial_volume"]), jnp.asarray(saved["initial_content"])),
                                 jnp.asarray(saved["initial_east_velocity"]), jnp.asarray(saved["initial_north_velocity"]))
            longitude_edges, latitude_edges, interfaces = (saved[name].copy() for name in ("longitude_edges", "latitude_edges", "interfaces"))
        wet = geometry.thickness > 0.
        initial_concentration = np.asarray(initial.inventory.content) / np.where(np.asarray(initial.inventory.volume) > 0., initial.inventory.volume, 1.)[..., None]
        low = np.min(np.where(wet[..., None], initial_concentration, np.inf), axis=(0, 1, 2))
        high = np.max(np.where(wet[..., None], initial_concentration, -np.inf), axis=(0, 1, 2))
        key = original["geometry"]
        if key not in kernels:
            def create_step(active_geometry, reference_coefficients):
                @jax.jit
                def step(state):
                    density = state.inventory.content[..., 0] / jnp.where(state.inventory.volume > 0., state.inventory.volume, 1.)
                    return paired_momentum_surface_step(active_geometry, state, density, 60., 4, reference=reference_coefficients)
                return step
            kernels[key] = create_step(device, coefficients)
        run = {"geometry": key, "velocity_dtype": original["velocity_dtype"], "disturbed": original["disturbed"],
               "shape": list(initial.inventory.volume.shape), "dt_seconds": 60., "fast_substeps": 4,
               "input_snapshot": original["snapshot_path"], "input_snapshot_sha256": original["snapshot_sha256"],
               "requested_steps": args.steps, "completed_steps": 0, "status": "running", "history": []}
        report["runs"].append(run)
        save()
        state = initial
        for iteration in range(args.steps):
            previous = state
            result = kernels[key](previous)
            result.valid.block_until_ready()
            final = result.state
            concentration = np.asarray(final.inventory.content) / np.where(np.asarray(final.inventory.volume) > 0., final.inventory.volume, 1.)[..., None]
            values = [result.surface.solve_relative_residual, result.surface.energy_work_relative, result.surface.cast_work,
                      result.moving_mass_energy_change, result.surface_error, result.transport.max_outflow_fraction,
                      result.flux_reconstruction.bottom_closure_relative, result.dual_transport.commutation_relative,
                      result.physical_velocity.bulk_continuity_relative,
                      np.max(np.abs(concentration[..., 1][wet] - 35.)),
                      max(0., float(np.max(np.maximum(low - concentration[wet], concentration[wet] - high)))),
                      max(float(jnp.max(jnp.abs(final.east_velocity))), float(jnp.max(jnp.abs(final.north_velocity)))),
                      float(jnp.max(jnp.abs(result.surface.eta)))]
            finite = bool(np.all(np.isfinite(values)))
            row = {name: float(value) if np.isfinite(value) else None for name, value in zip(METRICS, values)}
            passed = (bool(result.valid) and finite and row["solve_relative"] <= 1e-12
                      and row["frozen_energy_work_relative"] <= 1e-11
                      and row["constant_error"] <= 1e-12 and row["bound_excursion"] <= 1e-12)
            row.update(step=iteration + 1, accepted=bool(passed))
            run["history"].append(row)
            state = final
            if not passed:
                run["rejected_step"] = iteration + 1
                break
            run["completed_steps"] += 1
            if iteration % 10 == 0:
                print(key, original["velocity_dtype"], original["disturbed"], iteration + 1, row, flush=True)
                save()
        arrays = {"area": geometry.area, "thickness": geometry.thickness, "longitude_edges": longitude_edges,
                  "latitude_edges": latitude_edges, "interfaces": interfaces, "depth": depth,
                  "initial_volume": initial.inventory.volume, "initial_content": initial.inventory.content,
                  "previous_volume": previous.inventory.volume, "previous_content": previous.inventory.content,
                  "previous_east_velocity": previous.east_velocity, "previous_north_velocity": previous.north_velocity,
                  "final_volume": state.inventory.volume, "final_content": state.inventory.content,
                  "final_east_velocity": state.east_velocity, "final_north_velocity": state.north_velocity,
                  "uncast_east_velocity": result.surface.uncast_velocity.east, "uncast_north_velocity": result.surface.uncast_velocity.north,
                  "mean_east_flux": result.fluxes.east, "mean_north_flux": result.fluxes.north, "vertical_flux": result.fluxes.vertical,
                  "mean_eta": result.surface.mean_eta, "surface_eta": result.surface.eta, "initial_eta": result.surface.initial_eta,
                  "force_east": result.held_force.east, "force_north": result.held_force.north,
                  "external_work": result.surface.external_work, "cast_work": result.surface.cast_work,
                  "moving_mass_energy_change": result.moving_mass_energy_change}
        endpoint = output.with_name(f"{output.stem}_{key}_{original['velocity_dtype']}_{int(original['disturbed'])}.npz")
        np.savez(endpoint, **{name: np.asarray(value) for name, value in arrays.items()})
        run["snapshot_path"] = endpoint.relative_to(ROOT).as_posix()
        run["snapshot_sha256"] = hashlib.sha256(endpoint.read_bytes()).hexdigest()
        content_delta = np.sum(np.asarray(state.inventory.content - initial.inventory.content), axis=(0, 1, 2))
        budget = np.abs(content_delta) / np.sum(np.abs(np.asarray(initial.inventory.content)), axis=(0, 1, 2))
        run["content_budget_relative"] = [float(value) if np.isfinite(value) else None for value in budget]
        volume_delta = float(jnp.sum(state.inventory.volume - initial.inventory.volume))
        run["volume_delta_m3"] = volume_delta if np.isfinite(volume_delta) else None
        active_motion = max(row["speed_max_m_s"] or 0. for row in run["history"])
        run["status"] = "PASS" if run["completed_steps"] == args.steps and np.all(budget <= 1e-12) and (not run["disturbed"] or active_motion > 1e-9) else "FAIL"
        print("FINISHED", key, original["velocity_dtype"], original["disturbed"], run["status"], flush=True)
        save()
    unchanged = all(hashlib.sha256(files[name].read_bytes()).hexdigest() == value for name, value in hashes.items())
    report["source_hashes_unchanged"] = unchanged
    report["status"] = "PASS" if unchanged and all(run["status"] == "PASS" for run in report["runs"]) else "FAIL"
    save()
    print("VERDICT", report["status"], flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
