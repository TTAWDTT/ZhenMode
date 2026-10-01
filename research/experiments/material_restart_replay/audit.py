"""Independent NumPy/JSON audit; no imports from either restart or solver code."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _workspace_input_path(name, root):
    normalized = name.replace("\\", "/")
    workspace = root.as_posix().rstrip("/")
    prefixes = [workspace + "/"]
    if root.drive:
        prefixes.append(f"/mnt/{root.drive[0].lower()}{workspace[2:]}/")
    elif workspace.startswith("/mnt/") and len(workspace) > 6 and workspace[6] == "/":
        prefixes.append(workspace[5].upper() + ":" + workspace[6:] + "/")
    for prefix in prefixes:
        if normalized.casefold().startswith(prefix.casefold()):
            suffix = PurePosixPath(normalized[len(prefix):])
            if ".." in suffix.parts or suffix.is_absolute():
                raise ValueError("replay input path escapes the registered workspace")
            return root / suffix
    raise ValueError("replay input is not within this workspace or its WSL alias")


def _checkpoint(path):
    with np.load(path, allow_pickle=False) as saved:
        encoded = str(saved["metadata_json"])
        if hashlib.sha256(encoded.encode()).hexdigest() != str(saved["metadata_sha256"]):
            raise ValueError(f"metadata checksum failed: {path}")
        metadata = json.loads(encoded)
        if metadata["schema_version"] != 1 or set(metadata["arrays"]) != {"state", "cumulative", "history"}:
            raise ValueError("unsupported independent checkpoint schema")
        arrays = {}
        for group, descriptors in metadata["arrays"].items():
            arrays[group] = {}
            for name, descriptor in descriptors.items():
                values = np.array(saved[f"{group}__{name}"], copy=True)
                if (list(values.shape) != descriptor["shape"] or values.dtype.str != descriptor["dtype"]
                        or values.dtype.kind not in "biuf" or not np.isfinite(values).all()
                        or hashlib.sha256(values.tobytes()).hexdigest() != descriptor["sha256"]):
                    raise ValueError(f"raw checkpoint array identity failed: {group}/{name}")
                arrays[group][name] = values
        required = {"metadata_json", "metadata_sha256", *(f"{group}__{name}" for group in arrays for name in arrays[group])}
        if set(saved.files) != required or len(saved.files) != len(required):
            raise ValueError("unexpected or duplicate checkpoint arrays")
    if set(arrays["state"]) != {"u", "v", "T", "S", "eta", "ice"}:
        raise ValueError("independent audit requires all six fields")
    for name, values in arrays["state"].items():
        if (list(values.shape) != metadata["contract"]["state_shapes"][name]
                or values.dtype.str != metadata["contract"]["state_dtype"]):
            raise ValueError("state shape or dtype differs from its raw contract")
    if (metadata["elapsed_seconds"] != metadata["step"] * metadata["contract"]["dt_s"]
            or metadata["counters"] != {"accepted_steps": metadata["step"]}
            or arrays["history"]["minimum_thickness_m"].shape != (metadata["step"],)
            or np.any(arrays["history"]["minimum_thickness_m"] <= 0.)):
        raise ValueError("checkpoint clock, counter or positive-thickness history failed")
    return metadata, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-folder", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    folder = arguments.replay_folder.resolve()
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    baseline = root / protocol["baseline"]
    manifests = [json.loads((folder / segment / "manifest.json").read_text(encoding="utf-8"))
                 for segment in ("day7_to15", "day15_to30")]
    reports = [json.loads((folder / segment / "report.json").read_text(encoding="utf-8"))
               for segment in ("day7_to15", "day15_to30")]
    for manifest in manifests:
        for name, digest in manifest["original_sources_inputs"].items():
            if _hash_file(root / name) != digest:
                raise ValueError(f"current original source/input differs: {name}")
        for name, digest in manifest["replay_inputs"].items():
            if _hash_file(_workspace_input_path(name, root)) != digest:
                raise ValueError(f"replay source/input differs: {name}")
        for name, digest in manifest["contract"]["sources"].items():
            if _hash_file(root / "src" / f"{name}.py") != digest:
                raise ValueError(f"current contract source differs: {name}")
    paths = {"original_day7": baseline / "strict_restart_target7d.npz",
             "new_day15": folder / "day7_to15/strict_restart.npz",
             "new_day30": folder / "day15_to30/strict_restart.npz",
             "original_day30": baseline / "strict_restart_target30d.npz"}
    saved = {name: _checkpoint(path) for name, path in paths.items()}
    contracts_equal = all(metadata["contract"] == manifests[0]["contract"] for metadata, _ in saved.values())
    stages_match = (reports[0]["start_step"] == saved["original_day7"][0]["step"]
                    and reports[0]["stop_step"] == saved["new_day15"][0]["step"]
                    and reports[1]["start_step"] == saved["new_day15"][0]["step"]
                    and reports[1]["stop_step"] == saved["new_day30"][0]["step"])
    checkpoint_chain = (reports[0]["start_checkpoint_sha256"] == _hash_file(paths["original_day7"])
                        and reports[1]["start_checkpoint_sha256"] == _hash_file(paths["new_day15"]))
    original_report = json.loads((baseline / "report.json").read_text(encoding="utf-8"))["windows"]["30"]
    original_metadata, original_arrays = saved["original_day30"]
    with np.load(baseline / "last_authoritative_target30d.npz", allow_pickle=False) as original_state:
        baseline_state_matches = all(original_state[name].tobytes() == values.tobytes()
                                     for name, values in original_arrays["state"].items())
    baseline_budget_matches = all(np.asarray(original_report["cumulative_budget"][name], dtype=values.dtype).tobytes() == values.tobytes()
                                  for name, values in original_arrays["cumulative"].items())
    original_history = original_arrays["history"]["minimum_thickness_m"]
    prefix_identity = {name: arrays["history"]["minimum_thickness_m"].tobytes() == original_history[:metadata["step"]].tobytes()
                       for name, (metadata, arrays) in saved.items()}
    result = {"scope": protocol["scope"], "audit_has_no_solver_or_restart_imports": True,
              "contracts_equal": contracts_equal, "stages_match": stages_match,
              "start_checkpoint_hash_chain_matches": checkpoint_chain,
              "original_state_matches_authoritative_saved_state": baseline_state_matches,
              "original_budget_matches_continuous_report": baseline_budget_matches,
              "all_history_prefixes_match_continuous_trajectory": prefix_identity,
              "checkpoints": {name: {"sha256": _hash_file(path), "step": saved[name][0]["step"]} for name, path in paths.items()},
              "raw_array_byte_identity": {}, "segment_status": [report["status"] for report in reports]}
    reference_metadata, reference_arrays = saved["original_day30"]
    final_metadata, final_arrays = saved["new_day30"]
    for group in ("state", "cumulative", "history"):
        if set(final_arrays[group]) != set(reference_arrays[group]):
            raise ValueError("continuous and replayed arrays have different fields")
        result["raw_array_byte_identity"][group] = {
            name: (values.shape == reference_arrays[group][name].shape
                   and values.dtype == reference_arrays[group][name].dtype
                   and values.tobytes() == reference_arrays[group][name].tobytes())
            for name, values in final_arrays[group].items()}
    result["raw_clock_counter_output_identity"] = all(final_metadata[name] == reference_metadata[name]
                                                      for name in ("step", "elapsed_seconds", "counters", "outputs"))
    passed = (contracts_equal and stages_match and checkpoint_chain and result["raw_clock_counter_output_identity"]
              and baseline_state_matches and baseline_budget_matches and all(prefix_identity.values())
              and all(all(values.values()) for values in result["raw_array_byte_identity"].values())
              and all(report["status"] == "registered_segment_passed_not_industrial_qualification" for report in reports))
    result["audit_source_sha256"] = _hash_file(Path(__file__).resolve())
    result["protocol_sha256"] = _hash_file(protocol_path)
    result["status"] = "independent_cross_process_30day_restart_byte_identity_passed" if passed else "independent_audit_failed"
    with (folder / "validation_final.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
