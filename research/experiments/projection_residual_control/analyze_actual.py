"""Verify all registered full trajectories and distinguish qualification scopes."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/industrial_alignment/projection_refined_actual_summary.json")
    args = parser.parse_args()
    specifications = [("float64", 2, [180, 66, 14]), ("float32", 2, [180, 66, 14]),
                      ("float32", 1, [360, 130, 14])]
    report = {"scope": "projection_and_short_stability_not_physical_budget_or_climate_qualification",
              "status": "running", "results": [], "inputs": [],
              "analysis_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for dtype, resolution, dimensions in specifications:
        path = ROOT / f"results/industrial_alignment/projection_refined_{dtype}_{resolution}deg_1d.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["status"] != "complete" or data["grid"] != dimensions:
            raise ValueError(f"completed actual registered geometry required: {path}")
        arguments = data["arguments"]
        if arguments["days"] != 1. or arguments["dt"] != 600. or not arguments["audit_budget"]:
            raise ValueError("registered duration/time step and actual ledger required")
        for name, expected in data["provenance"]["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"runtime source mismatch: {name}")
        if hashlib.sha256(Path(arguments["bathy"]).read_bytes()).hexdigest() != data["provenance"]["bathymetry_sha256"]:
            raise ValueError("runtime bathymetry mismatch")
        report["inputs"].append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "git_head": data["provenance"]["git_head"],
                                 "runtime_source_sha256_verified": True})
        if {case["case"] for case in data["cases"]} != {"baseline", "ice"}:
            raise ValueError("both registered actual cases required")
        for case in data["cases"]:
            settings = case["column_projection"]
            if settings["niter"] != 600 or settings["preconditioner"] != "jacobi" or settings["max_refinements"] != 2:
                raise ValueError("registered effective projection settings required")
            if case["steps"] != 144 or case["status"] != "complete":
                raise ValueError("complete 144-step actual trajectory required")
            values = case["stage_budget"]["values"]
            maximum = float(values["projection_relative_residual_max"])
            maximum_records = max(record["projection_relative_residual_max"] for record in case["records"])
            if maximum != maximum_records:
                raise ValueError("per-step maximum aggregation differs from record maxima")
            norms = values["projection_transport_norm_squared"]
            if norms[0] <= 0. or norms[1] < 0.:
                raise ValueError("nonzero actual predictor norm required")
            threshold = 1e-9 if dtype == "float64" else 5e-5
            finite = np.isfinite(maximum) and all(record["finite"] for record in case["records"])
            projection_pass = bool(finite and maximum <= threshold)
            result = {"dtype": dtype, "resolution_degrees": resolution, "grid": dimensions, "case": case["case"],
                      "stability_pass": case["stability_pass"], "projection_pass": projection_pass,
                      "maximum_native_relative_residual": maximum, "projection_gate": threshold,
                      "cumulative_native_relative_residual": float(np.sqrt(norms[1] / norms[0])),
                      "peak_velocity_m_per_s": max(record["batch_peak_velocity"] for record in case["records"]),
                      "peak_eta_m": max(record["batch_peak_eta"] for record in case["records"]),
                      "fixed_node_proxy_budget_residual_heat_J_salt_kg_eta_m3": values["budget_residual"],
                      "physical_conservation_qualified": False, "century_qualified": False, "climate_qualified": False}
            report["results"].append(result)
    report["all_short_stability_pass"] = all(result["stability_pass"] for result in report["results"])
    report["all_registered_projection_gates_pass"] = all(result["projection_pass"] for result in report["results"])
    report["status"] = "complete"
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if not report["all_short_stability_pass"] or not report["all_registered_projection_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
