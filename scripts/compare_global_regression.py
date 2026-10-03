"""Compare independently observed global runs, including independent byte checks.

Source metadata may differ between revisions. Numerical arrays, data checksums,
effective parameters, forcing, environment and accepted steps must match.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def same_array(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def compare(reference, candidate, receipt_file="completed.json"):
    before = json.loads((reference / receipt_file).read_text(encoding="utf-8"))
    after = json.loads((candidate / receipt_file).read_text(encoding="utf-8"))
    for receipt in (before, after):
        if receipt["execution_status"] != "completed" or not receipt["duration_complete"]:
            raise ValueError("cannot compare unfinished or failed global runs")
    for directory, receipt in ((reference, before), (candidate, after)):
        for relative, expected in receipt["artifacts"].items():
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("artifact path leaves run directory")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("frozen artifact changed: " + relative)
    def options(value):
        return {k: v for k, v in value.items() if k not in {"out_dir", "log_dir"}}
    for name in ("plan_sha256", "observer_sha256", "selected_data", "requested_steps", "accepted_steps"):
        if before[name] != after[name]:
            raise ValueError("comparison precondition differs: " + name)
    if options(before["parsed_options"]) != options(after["parsed_options"]):
        raise ValueError("effective CLI options differ")
    def trajectories(receipt):
        return [{k: v for k, v in record.items() if k != "step_seconds"}
                for record in receipt["observed_states"]]
    if trajectories(before) != trajectories(after):
        raise ValueError("observed full-state trajectory differs")
    compared = []
    source_metadata = {}
    for relative in ("final-state.npz", "model/global_probe.npz", "model/ckpt_probe.npz"):
        with (np.load(reference / relative, allow_pickle=False) as left,
              np.load(candidate / relative, allow_pickle=False) as right):
            if set(left.files) != set(right.files):
                raise ValueError("record keys differ: " + relative)
            for key in left.files:
                if key == "source_identity_json":
                    source_metadata[relative + "/" + key] = {
                        "reference": json.loads(str(left[key])), "candidate": json.loads(str(right[key]))}
                    continue
                if key == "metadata_sha256":
                    for result in (left, right):
                        actual = hashlib.sha256(str(result["metadata_json"]).encode()).hexdigest()
                        if actual != str(result[key]):
                            raise ValueError("invalid checkpoint metadata checksum")
                    continue
                lvalue, rvalue = left[key], right[key]
                if key == "metadata_json":
                    lmeta, rmeta = json.loads(str(lvalue)), json.loads(str(rvalue))
                    source_metadata[relative + "/contract.sources"] = {
                        "reference": lmeta["contract"].pop("sources"),
                        "candidate": rmeta["contract"].pop("sources")}
                    lvalue = np.asarray(json.dumps(lmeta, sort_keys=True))
                    rvalue = np.asarray(json.dumps(rmeta, sort_keys=True))
                if not same_array(lvalue, rvalue):
                    raise ValueError("record bytes differ: " + relative + "/" + key)
                compared.append(relative + "/" + key)
    return {
        "schema": "zhenmode.global_regression_comparison.v1", "status": "passed",
        "reference": str(reference), "candidate": str(candidate),
        "reference_revision": before["revision"], "candidate_revision": after["revision"],
        "accepted_steps": before["accepted_steps"],
        "full_state_snapshots_equal": len(before["observed_states"]),
        "byte_equal_records": compared, "source_metadata_preserved_separately": source_metadata,
        "normalization": "source_identity_json and checkpoint contract.sources only; metadata checksum independently verified; no data, residual, array, effective parameter or forcing normalization",
        "costs": {"reference": {k: before[k] for k in ("setup_seconds", "end_to_end_observer_seconds")},
                  "candidate": {k: after[k] for k in ("setup_seconds", "end_to_end_observer_seconds")}},
        "qualification": "bounded global trajectory regression; no long-term climate claim or fair speed ranking",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-file", default="completed.json",
                        help="explicit receipt filename, including separately finalized receipts")
    args = parser.parse_args()
    report = compare(args.reference.resolve(), args.candidate.resolve(), args.receipt_file)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    print(json.dumps({"status": report["status"], "byte_equal_records": len(report["byte_equal_records"]),
                      "full_state_snapshots_equal": report["full_state_snapshots_equal"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
