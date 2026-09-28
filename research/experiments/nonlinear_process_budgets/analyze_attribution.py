"""Summarize registered attribution without converting internal transport into a source."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/industrial_alignment/nonlinear_process_budgets_1d.json")
    parser.add_argument("--out", default="results/industrial_alignment/nonlinear_attribution.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    input_path = root / args.input
    report = json.loads(input_path.read_text(encoding="utf-8"))
    if report["status"] != "complete" or not report["arguments"]["audit_budget"]:
        raise ValueError("a completed actual-stage audit is required")
    changed_sources = [name for name, digest in report["provenance"]["source_sha256"].items()
                       if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    if changed_sources:
        raise ValueError(f"runtime source hashes no longer match: {changed_sources}")
    summaries = []
    for case in report["cases"]:
        ledger = case["stage_budget"]
        if ledger["nonlinear_processes"] != ["advection", "convection", "gm", "redi"]:
            raise ValueError("unknown nonlinear process table")
        values = {name: np.asarray(value) for name, value in ledger["values"].items()}
        if not all(np.all(np.isfinite(value)) for value in values.values()):
            raise ValueError("nonfinite budget diagnostic")
        residual = values["budget_residual"]
        surface = values["surface_displacement_tracer_change"]
        boundary = values["advection_boundary_changes"]
        summaries.append({
            "case": case["case"], "stability_pass": case["stability_pass"],
            "metrics": ledger["metrics"],
            "internal_process_changes": dict(zip(ledger["nonlinear_processes"],
                                                  values["nonlinear_process_changes"].tolist(), strict=True)),
            "known_source_inputs": values["source_inputs"].sum(axis=0).tolist(),
            "original_budget_residual": residual.tolist(),
            "actual_top_transport": boundary.tolist(),
            "unexplained_after_top_attribution": (residual - boundary).tolist(),
            "advection_boundary_residual": values["advection_boundary_residual"].tolist(),
            "absolute_advection_boundary_residual": values["absolute_advection_boundary_residual"].tolist(),
            "nonlinear_accounting_residual": values["nonlinear_accounting_residual"].tolist(),
            "absolute_nonlinear_accounting_residual": values["absolute_nonlinear_accounting_residual"].tolist(),
            "surface_displacement_tracer_change": surface.tolist(),
            "linearized_surface_inventory_residual_not_physical_closure": (residual + surface).tolist(),
        })
        if "projection_transport_norm_squared" in values:
            norms = values["projection_transport_norm_squared"]
            summaries[-1]["projection_transport_norm_squared"] = norms.tolist()
            summaries[-1]["native_projection_relative_area_l2_residual"] = (
                float(np.sqrt(norms[1] / norms[0])) if norms[0] > 0. else None)
    output = {"scope": "process_attribution_not_moving_volume_conservation_or_climate_qualification",
              "input": str(input_path), "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "runtime_source_hashes_match": True, "cases": summaries}
    destination = root / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
