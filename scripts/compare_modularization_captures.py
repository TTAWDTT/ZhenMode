"""Compare complete payloads; normalize only declared input provenance locations."""

import argparse
import json
from pathlib import Path

import numpy as np


def normalized_provenance(value, source_root):
    provenance = json.loads(str(value))
    location = source_root.resolve().parent
    selected = (
        provenance["contract"]["controls"]["forcing_provenance"]
        if "contract" in provenance
        else provenance
    )
    for record in selected["selected_files"].values():
        path = Path(record["path"])
        try:
            relative = path.relative_to(location)
        except ValueError:
            continue
        record["path"] = "<source-parent>/" + relative.as_posix()
    return np.asarray(json.dumps(provenance, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    args = parser.parse_args()
    different = []
    with (
        np.load(args.reference, allow_pickle=False) as before,
        np.load(args.candidate, allow_pickle=False) as after,
    ):
        if set(before.files) != set(after.files):
            raise ValueError("capture record names differ")
        for name in before.files:
            left, right = before[name], after[name]
            if name.endswith(("/forcing_provenance_json", "/checkpoint/metadata_semantics_json")):
                left = normalized_provenance(left, args.reference_source)
                right = normalized_provenance(right, args.candidate_source)
            if (
                left.dtype != right.dtype
                or left.shape != right.shape
                or left.tobytes() != right.tobytes()
            ):
                different.append(name)
        print(
            json.dumps(
                {
                    "records": len(before.files),
                    "different_records": different,
                    "normalization": "selected_files.path under the declared source parent only; checksums and effective arrays retained",
                }
            )
        )
    return int(bool(different))


if __name__ == "__main__":
    raise SystemExit(main())
