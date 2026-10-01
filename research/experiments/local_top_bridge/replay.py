"""Canonical multi-column NPZ -> scalar-only static report; one byte snapshot."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from component import column, pair


def run(path):
    snapshot = Path(path).read_bytes()
    shared = {"Tref", "Sref", "alpha", "beta", "rho0", "gravity", "source_sha", "terrain_sha"}
    per = {
        "depth",
        "reference_weights",
        "wet_mask",
        "values",
        "eta",
        "terrain_depth",
        "discrete_bottom",
    }
    with np.load(io.BytesIO(snapshot), allow_pickle=False) as packet:
        if set(packet.files) != shared | per | {"distances_m"}:
            raise ValueError("exact canonical keys required")
        source_identity = str(packet["source_sha"].item())
        values = packet["values"]
        count = values.shape[0]
        if values.ndim != 3:
            raise ValueError("values C,N,4 required")
        if any(packet[k].shape != () for k in shared):
            raise ValueError("shared scalar required")
        if any(packet[k].shape[0] != count for k in per):
            raise ValueError("column shape")
        distances = packet["distances_m"]
        if (
            distances.shape != (count, count)
            or not np.isfinite(distances).all()
            or np.any(distances < 0)
        ):
            raise ValueError("nonnegative distance matrix required")
        results = []
        reports = []
        for i in range(count):
            p = {k: packet[k] for k in shared}
            p.update({k: packet[k][i] for k in per})
            p["control_interfaces"] = np.array([])
            result, ok, report = column(p)
            results.append(result if ok else None)
            reports.append(dict(column=i, accepted=ok, **report))
        pairs = []
        for i in range(count):
            for j in range(i + 1, count):
                if distances[i, j] > 0 and results[i] is not None and results[j] is not None:
                    pairs.append(
                        dict(columns=[i, j], **pair(results[i], results[j], distances[i, j]))
                    )
    return dict(
        input_sha256=hashlib.sha256(snapshot).hexdigest(),
        declared_source_sha=source_identity,
        historical_source_independently_verified=False,
        qualification_passed=False,
        columns=reports,
        pairs=pairs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    report = run(args.input)
    with open(args.output, "x") as f:
        json.dump(report, f, indent=2)
