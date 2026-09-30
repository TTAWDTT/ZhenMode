"""Retained actual rejected-state geometry and sparse topology, no time advance."""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-topology", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = Path(__file__).with_name("real_geometry_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = [Path(__file__), protocol_path, Path(__file__).with_name("kernel.py"),
             Path(__file__).with_name("sparse_diffusion.py")]
    for name in protocol["inputs"]:
        paths.extend(root / name / filename for filename in ("grid_used.npz", "parameter_arrays_used.npz"))
    paths.extend(root / name for name in (*protocol["attempt_inputs"], *protocol["rejection_reports"]))
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (output / "hashes_start.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    for path in paths[:4]:
        shutil.copyfile(path, output / path.name)
    for folder in (root, root / "src"):
        sys.path.insert(0, str(folder))
    import jax
    import jax.numpy as jnp
    import numpy as np

    from research.experiments.material_rstar_coordinates.kernel import (
        make_reference_stencil,
        rstar_geometry,
    )
    from research.experiments.material_rstar_coordinates.sparse_diffusion import (
        make_diffusion_graph,
    )

    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "cpu":
        raise ValueError("registered real geometry/topology audit uses CPU only, no full-step GPU compile")
    report = {"scope": protocol["scope"], "backend": jax.default_backend(), "cases": [], "production_promotion": False}
    topology_done = False
    for name, attempted_name, report_name in zip(protocol["inputs"], protocol["attempt_inputs"], protocol["rejection_reports"], strict=True):
        folder = root / name
        with np.load(folder / "grid_used.npz", allow_pickle=False) as grid:
            depths = -grid["z"].copy()
            dy = float(grid["dy"])
        with np.load(folder / "parameter_arrays_used.npz", allow_pickle=False) as saved:
            values = {key: saved[key].copy() for key in ("wet_mask_z", "dz_node", "dx_2d", "cos_lat", "inv_dx")}
        with np.load(root / attempted_name, allow_pickle=False) as saved:
            surface = saved["eta"].copy()
        original_report = json.loads((root / report_name).read_text(encoding="utf-8"))
        wet = values["wet_mask_z"] > 0.
        widths = values["dz_node"] * wet
        depth = widths.sum(axis=-1)
        column = wet.any(axis=-1)
        scale = 1. + np.where(column, surface, 0.) / np.where(depth > 0., depth, 1.)
        expected = widths * scale[..., None]
        interfaces = np.concatenate((np.zeros(depth.shape + (1,)), np.cumsum(widths, axis=-1)), axis=-1)
        expected_interfaces = -np.where(column, surface, 0.)[..., None] + scale[..., None] * interfaces
        geometry = rstar_geometry(jnp.asarray(surface), jnp.asarray(depths), jnp.asarray(values["dz_node"]), jnp.asarray(values["wet_mask_z"]))
        actual = np.asarray(geometry.thickness)
        thickness_floor = 64. * np.finfo(float).eps * np.max(np.abs(expected))
        total_error = float(np.max(np.abs(actual.sum(axis=-1)[column] - (depth + surface)[column])))
        bed_error = float(np.max(np.abs(np.asarray(geometry.interface_depth)[..., -1][column] - depth[column])))
        geometry_floor = 64. * np.finfo(float).eps * float(np.max(depth[column] + np.abs(surface[column])))
        interface_error = float(np.max(np.abs(np.asarray(geometry.interface_depth) - expected_interfaces)))
        old_top = np.where(column, values["dz_node"][..., 0] + surface, np.inf)
        case = {"geometry_input": name, "attempt_input": attempted_name, "original_status_retained": original_report["status"],
                "old_top_minimum_m": float(np.min(old_top)), "rstar_wet_minimum_m": float(np.min(actual[wet])),
                "rstar_total_depth_minimum_m": float(np.min((depth + surface)[column])),
                "thickness_numpy_difference_m": float(np.max(np.abs(actual - expected))), "thickness_64eps_floor_m": thickness_floor,
                "total_depth_difference_m": total_error, "bed_difference_m": bed_error, "geometry_64eps_floor_m": geometry_floor,
                "interface_numpy_difference_m": interface_error,
                "static_geometry_passed": bool(geometry.valid and np.min(actual[wet]) > 0. and np.min(old_top) < 0.
                                               and np.max(np.abs(actual - expected)) <= thickness_floor
                                               and total_error <= geometry_floor and bed_error <= geometry_floor and interface_error <= geometry_floor)}
        if arguments.build_topology and not topology_done:
            params = SimpleNamespace(**{key: jnp.asarray(value) for key, value in values.items()}, dy=dy, inv_dy=1. / dy)
            started = time.perf_counter()
            stencil = make_reference_stencil(depths, params.wet_mask_z)
            graph = make_diffusion_graph(params, stencil)
            leaves = [leaf for leaf in jax.tree.leaves(graph) if hasattr(leaf, "nbytes")]
            case["topology"] = {"shape": list(graph.shape), "full_nodes": int(graph.wet.size), "wet_nodes": int(np.sum(wet)),
                                "gradient_faces": int(graph.left.size), "coalesced_edges": int(graph.pair_left.size),
                                "contribution_terms": int(graph.term_faces.size), "maximum_incident_degree": max(group.edges.shape[1] for group in graph.incidence_groups),
                                "stored_graph_bytes": int(sum(leaf.nbytes for leaf in leaves)),
                                "construction_seconds_not_throughput_benchmark": time.perf_counter() - started,
                                "full_step_compiled": False, "accepted_integration_steps": 0}
            topology_done = True
        report["cases"].append(case)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report["source_inputs_unchanged"] = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    report["static_geometry_gates_passed"] = report["source_inputs_unchanged"] and all(case["static_geometry_passed"] for case in report["cases"])
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0 if report["static_geometry_gates_passed"] else 1)


if __name__ == "__main__":
    main()
