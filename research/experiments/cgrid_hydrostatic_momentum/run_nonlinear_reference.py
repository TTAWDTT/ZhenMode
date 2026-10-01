"""New actual nonlinear trajectories from eight immutable real starting states."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from cgrid_momentum import LayerState
from finite_volume import ExtensiveState, build_geometry
from nonlinear_dynamics import nonlinear_momentum_surface_step

jax.config.update("jax_enable_x64", True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="results/industrial_alignment/cgrid_physical_velocity_coordinate_reference.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    output, input_path = ROOT / args.out, ROOT / args.inputs
    if output.exists() or args.steps < 1:
        raise ValueError("retain evidence; choose a new output and positive steps")
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    if git_status:
        raise ValueError("reference requires a clean committed revision")
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    if inputs["status"] != "PASS" or len(inputs["runs"]) != 8:
        raise ValueError("requires eight qualified immutable real starting states")
    source_paths = [ROOT / "src" / name for name in ("nonlinear_dynamics.py", "finite_volume.py", "bounded_transport.py", "cgrid_momentum.py", "wet_fluxes.py", "config.py")]
    source_paths.extend((Path(__file__), Path(__file__).with_name("verify_nonlinear_reference.py"),
                         Path(__file__).with_name("nonlinear_dual_protocol.md"), ROOT / "tests/test_nonlinear_dynamics.py"))
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in source_paths}
    report = {"status": "running", "scope": "actual_nonlinear_dual_momentum_surface_inventory_not_complete_thermodynamics_or_production",
              "kinetic_norm": "finite_volume_full_half_prism_dual_not_physical_field_L2", "curvature": True, "upwind": False,
              "source_policy": "zero_in_real_reference_explicit_signed_in_direct_tests", "dt_seconds": 60.,
              "provenance": {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                             "git_status": git_status, "source_sha256": hashes, "input_report": args.inputs,
                             "input_report_sha256": sha(input_path), "jax_version": jax.__version__,
                             "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()]}, "runs": []}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    for original in inputs["runs"]:
        input_snapshot = ROOT / original["snapshot_path"]
        if sha(input_snapshot) != original["snapshot_sha256"]:
            raise ValueError("immutable real starting-state hash mismatch")
        with np.load(input_snapshot, allow_pickle=False) as saved:
            coordinates = {name: saved[name].copy() for name in ("longitude_edges", "latitude_edges", "interfaces", "physical_depth")}
            geometry = build_geometry(coordinates["longitude_edges"], coordinates["latitude_edges"], coordinates["interfaces"], coordinates["physical_depth"])
            coefficients = jnp.asarray(saved["reference_coefficients"])
            initial = LayerState(ExtensiveState(jnp.asarray(saved["initial_volume"]), jnp.asarray(saved["initial_content"])),
                                 jnp.asarray(saved["initial_east_velocity"]), jnp.asarray(saved["initial_north_velocity"]))
        device_geometry = jax.tree_util.tree_map(jnp.asarray, geometry)
        wet = geometry.thickness > 0.
        initial_concentration = np.asarray(initial.inventory.content) / np.where(np.asarray(initial.inventory.volume) > 0., initial.inventory.volume, 1.)[..., None]
        low, high = np.min(initial_concentration[wet], axis=0), np.max(initial_concentration[wet], axis=0)
        run = {"geometry": original["geometry"], "velocity_dtype": original["velocity_dtype"], "disturbed": original["disturbed"],
               "shape": list(initial.inventory.volume.shape), "requested_steps": args.steps, "completed_steps": 0,
               "input_snapshot": original["snapshot_path"], "input_snapshot_sha256": original["snapshot_sha256"], "status": "running", "history": []}
        report["runs"].append(run)
        state = initial
        for iteration in range(args.steps):
            previous = state
            density = previous.inventory.content[..., 0] / jnp.where(previous.inventory.volume > 0., previous.inventory.volume, 1.)
            result = nonlinear_momentum_surface_step(device_geometry, previous, density, 60., reference=coefficients)
            result.valid.block_until_ready()
            concentration = np.asarray(result.state.inventory.content) / np.where(np.asarray(result.state.inventory.volume) > 0., result.state.inventory.volume, 1.)[..., None]
            row = {name: float(value) if np.isfinite(value) else None for name, value in result.diagnostics._asdict().items()}
            row.update(constant_error=float(np.max(abs(concentration[..., 1][wet] - 35.))),
                       bound_excursion=float(max(0., np.max(np.maximum(low - concentration[wet], concentration[wet] - high)))),
                       speed_max=float(max(np.max(abs(result.state.east_velocity)), np.max(abs(result.state.north_velocity)))))
            passed = bool(result.valid) and all(value is not None and np.isfinite(value) for value in row.values()) and row["constant_error"] <= 1e-12 and row["bound_excursion"] <= 1e-12
            row.update(step=iteration + 1, accepted=passed)
            run["history"].append(row)
            if not passed:
                print("FAIL", run["geometry"], run["velocity_dtype"], run["disturbed"], row, flush=True)
                break
            state = result.state
            run["completed_steps"] += 1
            if iteration == 0 or iteration + 1 == args.steps:
                print(run["geometry"], run["velocity_dtype"], run["disturbed"], iteration + 1, row, flush=True)
            save()
        fields = {"area": geometry.area, "thickness": geometry.thickness, **coordinates,
                  "previous_volume": previous.inventory.volume, "previous_content": previous.inventory.content,
                  "previous_east_velocity": previous.east_velocity, "previous_north_velocity": previous.north_velocity,
                  "final_volume": result.state.inventory.volume, "final_content": result.state.inventory.content,
                  "final_east_velocity": result.state.east_velocity, "final_north_velocity": result.state.north_velocity,
                  "uncast_east": result.uncast_velocity.east, "uncast_north": result.uncast_velocity.north,
                  "transport_east": result.transport_velocity.east, "transport_north": result.transport_velocity.north,
                  "initial_eta": result.initial_eta, "final_eta": result.eta, "mean_eta": result.mean_eta,
                  "flux_east": result.fluxes.east, "flux_north": result.fluxes.north, "flux_vertical": result.fluxes.vertical,
                  "held_east": result.held_force.east, "held_north": result.held_force.north,
                  "reaction_east": result.wall_reaction.east, "reaction_north": result.wall_reaction.north,
                  "reference_coefficients": coefficients}
        fields.update({"diagnostic_" + name: value for name, value in result.diagnostics._asdict().items()})
        snapshot = output.with_name(f"{output.stem}_{run['geometry']}_{run['velocity_dtype']}_{int(run['disturbed'])}.npz")
        if snapshot.exists():
            raise FileExistsError("retain prior nonlinear snapshot")
        np.savez_compressed(snapshot, **{name: np.asarray(value) for name, value in fields.items()})
        run.update(status="PASS" if run["completed_steps"] == args.steps else "FAIL",
                   snapshot_path=snapshot.relative_to(ROOT).as_posix(), snapshot_sha256=sha(snapshot))
        save()
    report["source_hashes_unchanged"] = all(sha(ROOT / name) == value for name, value in hashes.items())
    report["status"] = "PASS" if report["source_hashes_unchanged"] and all(run["status"] == "PASS" for run in report["runs"]) else "FAIL"
    save()
    print(report["status"], flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
