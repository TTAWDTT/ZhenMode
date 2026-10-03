"""Read → validate → existing score → comparability → durable report."""
from __future__ import annotations

import json
import math
from pathlib import Path

import netCDF4
import numpy as np

from ocean_solver.evaluation.protocols import LEGACY, digest, file_digest, load_json, load_protocol
from ocean_solver.validation.benchmarks.external import _time_days, score_external_field
from ocean_solver.validation.benchmarks.metrics import score_npz
from ocean_solver.validation.benchmarks.table import load_metric, markdown_table


def _timeline(days, protocol):
    days = np.asarray(days, dtype=float)
    if days.ndim != 1 or not days.size or not np.all(np.isfinite(days)) or np.any(np.diff(days) <= 0):
        raise ValueError("output time must be finite, nonempty and strictly increasing")
    start, end = protocol["window"]["start_day"], protocol["window"]["end_day"]
    if not math.isclose(float(days[-1]), end, rel_tol=0, abs_tol=1e-10):
        raise ValueError("output endpoint differs from protocol scoring-window endpoint")
    selected = days >= start - 1e-12
    if not selected.any() or days[0] > start + 1e-10:
        raise ValueError("output does not cover the requested scoring window")
    return [float(days[selected][0]), float(days[-1])]


def _reference_arrays(path):
    with np.load(path, allow_pickle=False) as values:
        required = {"T_init", "wet_mask", "lat", "lon"}
        if not required.issubset(values.files):
            raise ValueError(f"reference missing arrays: {sorted(required - set(values.files))}")
        initial = np.asarray(values["T_init"], dtype=float)
        raw_wet = np.asarray(values["wet_mask"])
        if not np.all(np.isfinite(raw_wet)) or not np.all((raw_wet == 0) | (raw_wet == 1)):
            raise ValueError("wet mask must contain finite binary 0/1 or Boolean values")
        wet = raw_wet.astype(bool)
        lat, lon = np.asarray(values["lat"], dtype=float), np.asarray(values["lon"], dtype=float)
        if initial.ndim != 3 or initial.shape[:2] != wet.shape or wet.shape != (lon.size, lat.size):
            raise ValueError("reference must have x/y/layer ordering on the declared grid")
        if not wet.any() or not np.all(np.isfinite(initial[:, :, 0][wet])):
            raise ValueError("reference requires finite temperature on nonempty wet grid")
        return initial, wet, lat, lon


def _score(path, protocol, *, format="npz", reference=None, geometry=None,
           variable="temp", lat_var="lath", lon_var="lonh", wet_var="wet"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"result file missing: {path}")
    _reference_arrays(path if format == "npz" else reference)
    start, end = protocol["window"]["start_day"], protocol["window"]["end_day"]
    if format == "npz":
        with np.load(path, allow_pickle=False) as values:
            required = {"days", "T_top", "T_init", "wet_mask", "lat", "lon", "verdict",
                        "max_u_peak", "max_eta", "heat_content_J", "salt_content_kg"}
            if not required.issubset(values.files):
                raise ValueError(f"result missing arrays: {sorted(required - set(values.files))}")
            window = _timeline(values["days"], protocol)
            if values["T_top"].shape != (values["days"].size, *values["wet_mask"].shape):
                raise ValueError("SST timeline/shape differs from reference")
            peak = np.asarray(values["max_u_peak"], dtype=float)
            if peak.ndim != 0 or not np.isfinite(peak) or peak < 0:
                raise ValueError("max_u_peak must be a finite nonnegative scalar")
            for name in ("heat_content_J", "salt_content_kg", "max_eta"):
                if values[name].shape != values["days"].shape or not np.all(np.isfinite(values[name])):
                    raise ValueError(f"invalid diagnostic timeline: {name}")
        metrics = score_npz(path, steady_days=end-start)
    elif format == "mom6":
        if reference is None or geometry is None:
            raise ValueError("MOM6 SST evaluation requires reference NPZ and native geometry")
        with netCDF4.Dataset(path) as values, netCDF4.Dataset(geometry) as grid:
            field = values[variable]
            if getattr(field, "units", "") not in {"degC", "Celsius", "degrees_C", "degrees_Celsius"}:
                raise ValueError("external temperature units must explicitly be Celsius")
            window = _timeline(_time_days(values["time"]), protocol)
            _, wet, lat, lon = _reference_arrays(reference)
            for name, expected in ((lat_var, lat), (lon_var, lon)):
                actual = np.asarray(grid[name][:], dtype=float)
                if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-6):
                    raise ValueError("shared grid coordinates differ; no implicit regridding")
            native_wet = np.asarray(grid[wet_var][:])
            if not np.all(np.isfinite(native_wet)) or not np.all((native_wet == 0) | (native_wet == 1)):
                raise ValueError("native wet mask must be finite binary values")
            if not np.array_equal(native_wet.astype(bool).T, wet):
                raise ValueError("shared wet masks differ")
        metrics = score_external_field(path, variable=variable, reference_path=reference,
                                       geometry=geometry, lat_var=lat_var, lon_var=lon_var,
                                       wet_var=wet_var, steady_days=end-start)
    else:
        raise ValueError(f"unknown output format {format}")
    declared_hash = protocol["reference"]["sha256"]
    if declared_hash is not None and declared_hash != metrics["comparison_reference_sha256"]:
        raise ValueError("reference field does not match protocol sha256")
    metrics["requested_window_days"] = [start, end]
    metrics["actual_saved_record_window_days"] = window
    return metrics


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _verify_execution_artifact(path, manifest):
    path = Path(path).resolve()
    registered = manifest.get("result_path")
    expected = manifest.get("result_sha256")
    candidates = [item for item in manifest.get("outputs", [])
                  if isinstance(item, dict) and item.get("role") == "model_result"
                  and Path(item.get("path", "")).resolve() == path]
    if registered is None and len(candidates) == 1:
        registered, expected = candidates[0]["path"], candidates[0]["sha256"]
    if registered is None or Path(registered).resolve() != path or expected != file_digest(path):
        raise ValueError("result is not the actual hash-bound output registered by this completed run")
    missing = []
    full, executed = manifest.get("source_identity"), manifest.get("executed_source_files")
    if not isinstance(full, dict) or not isinstance(executed, dict) or len(executed) < 2:
        missing.append("actual_executed_source_file_identity")
    else:
        for name, sha in executed.items():
            if name not in full or full[name] != sha:
                raise ValueError(f"executed source differs from frozen full package: {name}")
        embedded_receipt = manifest.get("result_source_identity", {})
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as actual_output:
                if "source_identity_json" in actual_output.files:
                    actual_receipt = json.loads(str(actual_output["source_identity_json"]))
                    if embedded_receipt and actual_receipt != embedded_receipt:
                        raise ValueError("actual output embedded source receipt differs from worker manifest")
                    embedded_receipt = actual_receipt
                else:
                    missing.append("actual_output_embedded_source_receipt")
        embedded = embedded_receipt.get("all_package_files")
        if embedded is None and isinstance(embedded_receipt.get("source_sha256"), dict):
            embedded = {name.removeprefix("ocean_solver/"): sha
                        for name, sha in embedded_receipt["source_sha256"].items()
                        if name.startswith("ocean_solver/")}
        if embedded is None:
            missing.append("result_embedded_source_identity")
        elif embedded != full:
            raise ValueError("result's embedded source identity differs from frozen run source")
        if manifest["method"] in {"zhenmode", "zhenmode_global_fd", "ocean_solver", "ocean-solver"}:
            from ocean_solver.provenance.sources import PACKAGE_SOURCE_MODULES

            required_full = {name.removeprefix("ocean_solver/")+".py" for name in PACKAGE_SOURCE_MODULES}
            required_executed = {"runtime/application.py", "runtime/integration.py", "model/factory.py",
                                 "timestepping/integration.py", "dynamics/processes.py",
                                 "numerics/horizontal.py", "physics/vertical.py", "io/output.py"}
            if not required_full <= set(full):
                missing.append("producer_full_package_source_envelope")
            if not required_executed <= set(executed):
                missing.append("production_executed_core_source_envelope")
            registry = Path(__file__).resolve().parents[1]/"provenance/sources.py"
            if full.get("provenance/sources.py") != file_digest(registry):
                missing.append("known_producer_source_registry_version")
    return missing


def _acceptance(metrics, protocol):
    limits = protocol["acceptance"]
    inputs = {"max_raw_rmse_degC": metrics["global"]["raw_rmse"],
              "max_abs_heat_drift_percent": metrics.get("heat_drift_percent"),
              "max_abs_salt_drift_percent": metrics.get("salt_drift_percent")}
    checks = {name: bool(_finite(inputs[name]) and abs(inputs[name]) <= limit)
              for name, limit in limits.items() if limit is not None}
    valid = metrics["verdict"] == "PASS" and metrics["coverage_complete"]
    return {"status": "not_declared" if not checks else ("passed" if valid and all(checks.values()) else "failed"),
            "checks": checks, "scope": "declared_finite_window_diagnostics_only"}


def _json_safe(value):
    if isinstance(value, dict):
        return {name: _json_safe(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def scoring_source_identity():
    package = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(package)): file_digest(path)
            for path in sorted(package.rglob("*.py"))}


def write_report(report, out_dir):
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "metrics.json", "report.md"):
        if (destination / name).exists():
            raise FileExistsError(f"report already exists: {destination / name}")
    report = _json_safe(report)
    (destination / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                      allow_nan=False) + "\n", encoding="utf-8")
    (destination / "metrics.json").write_text(json.dumps(report["effect"], indent=2,
                                                       allow_nan=False) + "\n", encoding="utf-8")
    text = (f"# {report['run_id']}\n\nCase: `{report['case_id']}`; method: `{report['method']}`.\n\n"
            f"Execution: **{report['execution_status']}**; acceptance: **{report['acceptance']['status']}**.\n\n"
            f"Protocol: `{report['protocol']['id']}`. Numerical scope: {report['numerical']['scope']}.\n\n"
            "The reference is the saved initialization; these diagnostics do not certify independent climate accuracy.\n\n"
            + markdown_table([load_metric(destination / "metrics.json", report["run_id"])]) + "\n\n"
            + "Limitations:\n\n" + "\n".join("- " + item for item in report["limitations"]) + "\n")
    (destination / "report.md").write_text(text, encoding="utf-8")
    return report


def evaluate(input_path, protocol_path, manifest_path, out_dir, **adapter):
    protocol, manifest = load_protocol(protocol_path), load_json(manifest_path)
    for name in ("run_id", "case_id", "method", "config_hash"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"run manifest missing identity: {name}")
    if manifest["case_id"] != protocol["case_id"]:
        raise ValueError("run case differs from evaluation protocol")
    if manifest.get("evaluation_protocol", protocol["id"]) != protocol["id"]:
        raise ValueError("run evaluation protocol differs")
    frozen_bytes = manifest.get("protocol_file_sha256", manifest.get("protocol_sha256"))
    if frozen_bytes is not None and frozen_bytes != file_digest(protocol_path):
        raise ValueError("evaluation protocol content changed after run was frozen")
    if manifest.get("protocol_content_sha256") is not None and manifest["protocol_content_sha256"] != digest(protocol):
        raise ValueError("evaluation canonical protocol content differs from frozen run")
    if manifest.get("execution_status") != "completed":
        raise ValueError("only a completed run may be scored; failed/uncompleted stays in run index")
    provenance_missing = _verify_execution_artifact(input_path, manifest)
    if adapter.get("format", "npz") == "mom6":
        for argument, path_field, hash_field in (("geometry", "geometry_path", "geometry_sha256"),
                                                ("reference", "reference_npz", "reference_npz_sha256")):
            selected = adapter.get(argument)
            if selected is None or manifest.get(path_field) is None or Path(selected).resolve() != Path(manifest[path_field]).resolve() or file_digest(selected) != manifest.get(hash_field):
                raise ValueError(f"actual MOM6 {argument} is not hash-bound to this run")
    metrics = _score(input_path, protocol, **adapter)
    limitations = list(protocol["limitations"]) + [
        "Endpoint heat/salt changes are diagnostics, not independently closed budget residuals.",
        "Saved records have arithmetic time weights; no implicit time-bound weighting or regridding.",
        "No long-term stability, industrial qualification or equal-error speed advantage is inferred."]
    if metrics.get("area_source") == "inferred_center_edges":
        limitations.append("Cell areas inferred from centers; actual native geometry not certified.")
    comparison_identity = {"case_id": manifest["case_id"], "protocol_sha256": digest(protocol),
                           "physical_problem_sha256": manifest.get("physical_problem_sha256"),
                           "effective_physics_sha256": manifest.get("effective_physics_sha256"),
                           "physical_grid_sha256": manifest.get("physical_grid_sha256"),
                           "data_sha256": manifest.get("data_sha256"),
                           "window": metrics["actual_saved_record_window_days"],
                           "metric_definition": metrics["metric_definition"],
                           "domain_sha256": metrics["comparison_domain_sha256"],
                           "reference_sha256": metrics["comparison_reference_sha256"]}
    missing = [name for name, value in comparison_identity.items() if value is None] + provenance_missing
    report = {"schema_version": 1, "run_id": manifest["run_id"], "case_id": manifest["case_id"],
              "method": manifest["method"], "experiment_id": manifest.get("experiment_id"),
              "preset_id": manifest.get("preset_id"), "variant": manifest.get("variant"),
              "changes": manifest.get("changes", []), "config_hash": manifest["config_hash"],
              "change_categories": [{"option": change.get("option"),
                                     "category": ("numerical_discretization" if change.get("option") in {
                                         "dt", "dt_bt", "dtype", "n_subcyc", "mode_split", "process_time_scheme",
                                         "use_scan", "nu_nsub"} else "physical_or_parameterization_requires_identity_match")}
                                    for change in manifest.get("changes", []) if isinstance(change, dict)],
              "execution_status": "completed", "protocol": protocol,
              "comparison_identity": comparison_identity,
              "comparability": {"status": "limited" if missing else "eligible_for_contract_comparison",
                                "missing_identity": missing},
              "acceptance": _acceptance(metrics, protocol),
              "numerical": {"status": "passed" if metrics["verdict"] == "PASS" else "failed",
                            "scope": metrics["verdict_scope"], "long_term_stability": "not_run",
                            "independent_budget_closure": "not_run"},
              "effect": metrics, "cost": manifest.get("cost", {"status": "not_recorded"}),
              "provenance": {"result": {"path": str(Path(input_path).resolve()),
                                         "sha256": file_digest(input_path)},
                             "run_manifest_sha256": file_digest(manifest_path),
                             "execution_source_identity": manifest.get("source_identity"),
                             "scoring_source_identity": scoring_source_identity(),
                             "protocol_file_sha256": file_digest(protocol_path),
                             "protocol_content_sha256": digest(protocol)},
              "limitations": limitations, "industrial_qualified": False}
    if provenance_missing:
        limitations.append("Actual executed source receipt is incomplete; source verification remains limited.")
    written = write_report(report, out_dir)
    report_path = Path(out_dir).resolve() / "report.json"
    manifest.update(acceptance=written["acceptance"], comparability=written["comparability"],
                    evaluation_report={"path": str(report_path), "sha256": file_digest(report_path),
                                       "protocol_id": protocol["id"]})
    manifest_path = Path(manifest_path)
    temporary = manifest_path.with_suffix(".json.evaluation.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return written


def compare(reports):
    if len(reports) < 2:
        raise ValueError("comparison needs at least two reports")
    reasons = []
    base = reports[0].get("comparison_identity")
    if not base:
        reasons.append("missing protocol-bound comparison identity (historical results cannot be promoted)")
    for report in reports:
        identity = report.get("comparison_identity")
        if identity != base or not identity or any(v is None for v in identity.values()):
            reasons.append(f"{report.get('run_id', 'unknown')}: different or incomplete comparison identity")
        comparability = report.get("comparability", {})
        if not isinstance(comparability, dict) or comparability.get("status") != "eligible_for_contract_comparison":
            reasons.append(f"{report.get('run_id', 'unknown')}: verification/comparability is limited")
        if report.get("execution_status") != "completed" or report.get("numerical", {}).get("status") != "passed":
            reasons.append(f"{report.get('run_id', 'unknown')}: numerical completion not passed")
    cost_keys = {"hardware", "precision", "cpu_ranks", "timing_scope"}
    components = {"compile_s", "integration_s", "io_s", "end_to_end_s"}
    costs = [r.get("cost", {}) if isinstance(r.get("cost", {}), dict) else {} for r in reports]
    cost_ok = all(cost_keys | components <= set(c) and all(_finite(c[k]) and c[k] >= 0 for k in components)
                  for c in costs) and all({k: c[k] for k in cost_keys} == {k: costs[0][k] for k in cost_keys}
                                         for c in costs)
    return {"schema_version": 1, "comparable": not reasons, "reasons": sorted(set(reasons)),
            "cost_comparable": bool(cost_ok and not reasons), "ranking": "not_performed",
            "equal_error_speedup": "not_established_requires_measured_error_cost_curve",
            "runs": [{"run_id": r.get("run_id"), "method": r.get("method"),
                      "acceptance": r.get("acceptance"), "cost": r.get("cost")} for r in reports]}


def import_historical(path):
    value = load_json(path)
    return {"schema_version": 1, "kind": "historical_evidence", "source_path": str(Path(path)),
            "source_sha256": file_digest(path),
            "metric_definition": value.get("metric_definition", LEGACY),
            "metrics": value, "comparability": "historical_protocol_only_no_new_ranking",
            "independent_rerun": False}
