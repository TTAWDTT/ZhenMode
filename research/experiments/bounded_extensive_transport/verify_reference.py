"""Recompute inventory gates from hashed reference snapshots, not status alone."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ocean_solver.provenance.archives import verify_current_source_hashes

ROOT = Path(__file__).resolve().parents[3]


def _workspace_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"missing or outside-workspace evidence: {relative}")
    return path


def verify(report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["inventory_policy"] != "float64" or report["transport_scheme"] != "centered_fct":
        raise ValueError("this verifier requires the registered explicit64 centered-FCT reference")
    required_sources = {"src/finite_volume.py", "src/bounded_transport.py", "src/barotropic_transport.py",
                        "src/grid.py", "src/config.py", "tests/test_bounded_extensive_transport.py",
                        "tests/test_extensive_transport.py", "research/experiments/extensive_transport/run_reference.py",
                        "research/experiments/extensive_transport/protocol.md",
                        "research/experiments/extensive_transport/precision_protocol.md",
                        "research/experiments/bounded_extensive_transport/protocol.md",
                        "research/experiments/bounded_extensive_transport/selection.md"}
    manifest_paths = {relative.replace("\\", "/") for relative in report["provenance"]["source_sha256"]}
    if not required_sources.issubset(manifest_paths):
        raise ValueError("runtime source manifest is incomplete")
    verify_current_source_hashes(ROOT, report["provenance"]["source_sha256"])
    groups = set()
    rows = []
    for run in report["runs"]:
        if run["completed_steps"] != 100 or run["dt_seconds"] != 60. or run["barotropic_substeps"] != 4:
            raise ValueError("registered duration/timestep/substep gate not met")
        if run["status"] != "PASS" or run["inventory_dtype"] != "float64":
            raise ValueError("reference run did not qualify explicit64 inventories")
        if not all(np.isfinite(run[key]) for key in ("maximum_surface_identity_error", "maximum_outflow_fraction", "maximum_gravity_cfl_bound")):
            raise ValueError("nonfinite recorded intermediate gate")
        snapshot = _workspace_path(run["snapshot_path"])
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != run["snapshot_sha256"]:
            raise ValueError(f"snapshot hash mismatch: {snapshot}")
        with np.load(snapshot, allow_pickle=False) as data:
            initial_volume, final_volume = data["initial_volume"], data["final_volume"]
            initial_content, final_content = data["initial_content"], data["final_content"]
            depth, area, thickness = data["physical_depth"], data["area"], data["thickness"]
            eta = data["eta"]
        if (initial_volume.shape != (180, 66, 14) or final_volume.shape != initial_volume.shape
                or initial_content.shape != (180, 66, 14, 2) or final_content.shape != initial_content.shape
                or depth.shape != (180, 66) or area.shape != depth.shape
                or thickness.shape != initial_volume.shape or eta.shape != depth.shape):
            raise ValueError("registered real-reference shapes do not match")
        if any(field.dtype != np.float64 for field in (initial_volume, final_volume, initial_content, final_content)):
            raise ValueError("registered64 inventories were not stored")
        if any(not np.all(np.isfinite(field)) for field in (initial_volume, final_volume, initial_content, final_content, eta, depth, area, thickness)):
            raise ValueError("nonfinite inventory or surface")
        if np.any(area <= 0.) or np.any(depth < 0.) or np.any(thickness < 0.):
            raise ValueError("invalid physical geometry")
        wet = thickness > 0.
        if not (np.all(final_volume[wet] > 0.) and np.all(final_volume[~wet] == 0.)
                and np.all(final_content[~wet] == 0.)):
            raise ValueError("invalid wet/dry final inventory")
        dtype_tolerance = 1e-12 if run["dtype"] == "float64" else 2e-6
        geometry_tolerance = 1e-12 if run["dtype"] == "float64" else 2e-6
        if np.max(np.abs(thickness.sum(axis=-1) - depth) / np.maximum(depth, 1.)) > geometry_tolerance:
            raise ValueError("physical depth identity failed")
        before = initial_content[wet] / initial_volume[wet, None]
        after = final_content[wet] / final_volume[wet, None]
        under = np.maximum(before.min(axis=0) - after.min(axis=0), 0.)
        over = np.maximum(after.max(axis=0) - before.max(axis=0), 0.)
        bound_error = float(max(under.max(), over.max()))
        constant_error = float(np.max(np.abs(after - before) / np.maximum(np.abs(before), 1.)))
        budgets = np.abs(final_content.sum(axis=(0, 1, 2)) - initial_content.sum(axis=(0, 1, 2))) / np.abs(initial_content).sum(axis=(0, 1, 2))
        if np.any(budgets > 1e-12) or bound_error > dtype_tolerance:
            raise ValueError("recomputed inventory budget or unforced bound failed")
        if run["constant"] and constant_error > dtype_tolerance:
            raise ValueError("recomputed constant gate failed")
        if run["maximum_surface_identity_error"] > dtype_tolerance:
            raise ValueError("recorded intermediate surface gate failed")
        if run["maximum_outflow_fraction"] > 1. or run["maximum_gravity_cfl_bound"] > 2.:
            raise ValueError("recorded intermediate CFL gate failed")
        physical_eta = final_volume[..., 0] / area - thickness[..., 0]
        if not np.allclose(eta, physical_eta, rtol=dtype_tolerance, atol=1e-12 if run["dtype"] == "float64" else 1e-7):
            raise ValueError("stored surface is not derived from primary V")
        delta = final_volume - initial_volume
        volume_delta = float(delta.sum())
        if not np.isclose(volume_delta, run["volume_budget_delta_m3"], rtol=0., atol=1e-8):
            raise ValueError("reported water residual differs from snapshots")
        rows.append({"geometry": run["geometry"], "momentum_dtype": run["dtype"],
                     "constant": run["constant"], "content_budget": budgets.tolist(),
                     "absolute_bound_excursion": bound_error, "constant_error": constant_error if run["constant"] else None,
                     "volume_delta_m3": volume_delta,
                     "equivalent_mean_surface_m": volume_delta / float(area[depth > 0.].sum())})
        groups.add((run["geometry"], run["dtype"], run["constant"]))
    expected_groups = {(geometry, dtype, constant)
                       for geometry in ("prior_smoothed", "unsmoothed")
                       for dtype in ("float64", "float32") for constant in (True, False)}
    if len(rows) != 8 or groups != expected_groups or report["status"] != "PASS":
        raise ValueError("eight registered groups are not all qualified")
    return {"status": "PASS", "scope": "hashed_component_snapshots_not_production_century_climate_or_full_adjoint",
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="results/industrial_alignment/bounded_transport_reference.json")
    args = parser.parse_args()
    report_path = _workspace_path(args.report)
    result = verify(report_path)
    output = report_path.with_name(report_path.stem + "_verified.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
