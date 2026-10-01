"""Canonical multi-column NPZ -> scalar-only static report; one byte snapshot."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from component import column, pair, require_finite


def run(path, geometry_report_path):
    snapshot = Path(path).read_bytes()
    geometry_snapshot = Path(geometry_report_path).read_bytes()
    geometry = json.loads(geometry_snapshot)
    if geometry.get("schema_version") != 1 or geometry.get("full_geometry_passed") is not True:
        raise ValueError("external full geometry audit required")
    if geometry.get("input_sha256") != hashlib.sha256(snapshot).hexdigest():
        raise ValueError("geometry audit input identity mismatch")
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
        if geometry.get("source_sha") != source_identity or geometry.get("terrain_sha") != str(
            packet["terrain_sha"].item()
        ):
            raise ValueError("geometry audit source/terrain identity mismatch")
        if (
            geometry.get("reference_weights_sha256")
            != hashlib.sha256(packet["reference_weights"].tobytes()).hexdigest()
        ):
            raise ValueError("geometry audit weight identity mismatch")
        values = packet["values"]
        count = values.shape[0]
        if values.ndim != 3 or count == 0:
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
    report = dict(
        external_geometry_report_sha256=hashlib.sha256(geometry_snapshot).hexdigest(),
        external_geometry_assertion_bound=True,
        external_geometry_independently_reperformed=False,
        input_sha256=hashlib.sha256(snapshot).hexdigest(),
        declared_source_sha=source_identity,
        historical_source_independently_verified=False,
        qualification_passed=False,
        columns=reports,
        pairs=pairs,
    )

    require_finite(report)
    return report


def write_report(report, path):
    payload = json.dumps(report, indent=2, allow_nan=False)
    with open(path, "x") as f:
        f.write(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--geometry-report", required=True)
    args = parser.parse_args()
    report = run(args.input, args.geometry_report)
    write_report(report, args.output)
