"""Full actual geometry prescribed-stage capacity; never an ocean integration."""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for folder in (ROOT, ROOT / "src"):
        sys.path.insert(0, str(folder))

    import jax
    import jax.numpy as jnp
    import numpy as np

    from config import C_P, RHO_0
    from research.experiments.material_rstar_coordinates.bed_completion import (
        complete_bed_reference,
    )
    from research.experiments.material_rstar_coordinates.kernel import rstar_geometry
    from research.experiments.material_rstar_coordinates.nodal_mass import (
        apply_nodal_mass,
        make_nodal_mass,
        solve_nodal_mass,
    )
    from research.experiments.material_rstar_coordinates.weak_bounded import bounded_weak_euler
    from research.experiments.material_rstar_coordinates.weak_sparse import make_weak_graph
    from research.experiments.material_rstar_coordinates.weak_transport import WeakParameters

    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--advance", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ["src/config.py", "src/jax_solver_global.py", "src/material_top.py"]
    names += ["research/experiments/material_rstar_coordinates/" + name for name in
              ("kernel.py", "bed_completion.py", "nodal_mass.py", "pressure_work.py", "weak_transport.py", "weak_sparse.py", "weak_bounded.py", "sparse_diffusion.py", "weak_bounded_protocol.json", "real_geometry_protocol.json")]
    names += [str(Path(__file__).relative_to(ROOT)).replace("\\", "/"), "research/experiments/material_rstar_coordinates/weak_actual_protocol.json"]
    hashes = {name: digest(ROOT / name) for name in names}
    for name in names:
        destination = output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    report = {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "backend": jax.default_backend(),
              "devices": [str(device) for device in jax.devices()], "x64_enabled": bool(jax.config.x64_enabled),
              "jax": jax.__version__, "numpy": np.__version__, "source_hashes": hashes, "cases": [],
              "environment": {name: os.environ.get(name) for name in ("JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE")},
              "scope": "original full actual grid with prescribed fresh analytic fields, not continuation or whole-ocean dynamics",
              "accepted_ocean_steps": 0, "production_promotion": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    protocol = json.loads((ROOT / "research/experiments/material_rstar_coordinates/real_geometry_protocol.json").read_text())
    advance = jax.jit(bounded_weak_euler)
    for index, directory in enumerate(protocol["inputs"]):
        grid_path = ROOT / directory / "grid_used.npz"
        parameter_path = ROOT / directory / "parameter_arrays_used.npz"
        state_path = ROOT / protocol["attempt_inputs"][index]
        inputs = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in (grid_path, parameter_path, state_path)}
        with np.load(grid_path) as packet:
            grid = {name: packet[name] for name in packet.files}
        with np.load(parameter_path) as packet:
            old = SimpleNamespace(**{name: packet[name] for name in packet.files}, dy=float(grid["dy"]))
        with np.load(state_path) as packet:
            surface = jnp.asarray(packet["eta"])
        reference = complete_bed_reference(-grid["z"], old)
        params = WeakParameters(reference.wet, reference.wet[..., 0], reference.widths,
                                jnp.asarray(grid["dx_2d"]), float(grid["dy"]), jnp.asarray(grid["cos_lat"]))
        mass = make_nodal_mass(reference.nodes, params)
        started = time.perf_counter()
        graph = make_weak_graph(params)
        jax.block_until_ready(graph)
        record = {"original_input": directory, "input_hashes": inputs, "shape": list(reference.wet.shape),
                  "wet_nodes": int(reference.wet.sum()), "added_bed_samples": int(reference.added.sum()),
                  "weak_pairs": int(graph.pair_left.size), "graph_build_seconds": time.perf_counter() - started,
                  "stored_graph_bytes": sum(value.size * value.dtype.itemsize for value in jax.tree.leaves(graph)),
                  "maximum_incidence_degree": max(group.edges.shape[1] for group in graph.incidence_groups),
                  "prescribed_tracer_stage_attempted": args.advance}
        print(json.dumps(record), flush=True)
        report["active_case"] = record
        report["phase"] = "compiling_and_executing" if args.advance else "static_graph"
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.advance:
            geometry = rstar_geometry(surface, reference.nodes, reference.widths, reference.wet)
            longitude = jnp.radians(jnp.asarray(grid["lon"]))[:, None, None]
            latitude = jnp.radians(jnp.asarray(grid["lat"]))[None, :, None]
            profile = jnp.exp(-geometry.node_depth / 2000.)
            concentration = (10. + 10. * profile + .2 * jnp.sin(longitude) * jnp.cos(latitude)) * mass.wet
            velocities = (.03 * jnp.sin(longitude) * jnp.cos(latitude) * profile * mass.wet,
                          .02 * jnp.cos(longitude) * jnp.sin(latitude) * profile * mass.wet)
            area = params.dx_2d * params.dy
            content = apply_nodal_mass(concentration, geometry, area, mass)
            source = jnp.zeros_like(content).at[..., 0].set(100. * area * params.wet_mask / (RHO_0 * C_P))
            jax.block_until_ready((content, velocities, source))
            started = time.perf_counter()
            result = advance(content, surface, velocities, source, 600., params, reference.nodes, mass, graph)
            jax.block_until_ready(result)
            elapsed = time.perf_counter() - started
            next_geometry = rstar_geometry(result.surface, reference.nodes, reference.widths, reference.wet)
            decoded = np.asarray(solve_nodal_mass(result.content, next_geometry, area, mass))
            wet = np.asarray(mass.wet)
            bounds_floor = 64. * np.finfo(float).eps * (1. + np.abs(decoded) + np.abs(np.asarray(concentration)))
            bound_passed = bool(np.all(decoded[wet] >= np.asarray(result.lower_bound)[wet] - bounds_floor[wet])
                                and np.all(decoded[wet] <= np.asarray(result.upper_bound)[wet] + bounds_floor[wet]))
            heat = RHO_0 * C_P * float(np.asarray(result.source).sum())
            exact_heat = 600. * float(np.sum(100. * np.asarray(area) * np.asarray(params.wet_mask)))
            heat_passed = abs(heat - exact_heat) <= 64. * np.finfo(float).eps * abs(exact_heat)
            record.update(valid=bool(result.valid), bounds_passed=bound_passed, physical_heat_passed=bool(heat_passed),
                          compile_and_first_execute_seconds=elapsed, fraction=float(result.fraction), minimum_limiter=float(result.minimum_limiter),
                          inventory_residual=float(result.inventory_residual), heat_joules=heat, expected_heat_joules=exact_heat,
                          min_actual_thickness_m=float(np.asarray(next_geometry.thickness)[wet].min()),
                          min_tracer=float(decoded[wet].min()), max_tracer=float(decoded[wet].max()), memory_stats=jax.devices()[0].memory_stats())
            witness = output / ("case" + str(index) + ".npz")
            np.savez_compressed(witness, content=np.asarray(result.content), original_content=np.asarray(content),
                                source=np.asarray(result.source), decoded=decoded, original_concentration=np.asarray(concentration),
                                surface=np.asarray(result.surface), original_surface=np.asarray(surface),
                                lower=np.asarray(result.lower_bound), upper=np.asarray(result.upper_bound), wet=wet,
                                thickness=np.asarray(next_geometry.thickness), area=np.asarray(area))
            record.update(witness=witness.name, witness_sha256=digest(witness))
            print(json.dumps(record), flush=True)
        record["inputs_unchanged"] = all(digest(ROOT / name) == value for name, value in inputs.items())
        report["cases"].append(record)
        report["source_hashes_unchanged"] = all(digest(ROOT / name) == value for name, value in hashes.items())
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["phase"] = "finished"
    report["accepted_prescribed_tracer_stages"] = sum(case.get("valid", False) for case in report["cases"])
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    assert report["source_hashes_unchanged"] and all(case["inputs_unchanged"] for case in report["cases"])
    if args.advance:
        assert all(case["valid"] and case["bounds_passed"] and case["physical_heat_passed"] for case in report["cases"])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv and not isinstance(error, FileExistsError):
            folder = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
            if folder.is_dir():
                (folder / "failure.json").write_text(json.dumps({"exception": type(error).__name__, "message": str(error),
                    "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "accepted_ocean_steps": 0}, indent=2) + "\n")
        raise
