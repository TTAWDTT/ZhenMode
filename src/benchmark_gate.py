"""Pre-registered pass/fail gates for standardized benchmark comparisons."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _metric(data: dict, name: str):
    if name == "global_a2_rmse":
        return data.get("global_a2_rmse_c")
    if name == "na_raw_rmse":
        return data.get("north_atlantic_40_60", {}).get("raw_rmse")
    if name == "near_wall_raw_bias":
        return data.get("near_wall_55_60", {}).get("raw_bias")
    if name in {"heat_drift", "salt_drift"}:
        return data.get(f"{name}_percent")
    return data.get(name)


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def evaluate_gate(control: dict, experiment: dict,
                  expected_days: float | None = None,
                  tolerance: float = 1e-6) -> dict:
    """Apply the pre-registered internal candidate gates.

    Missing or non-finite metrics fail closed. Bias is compared by magnitude;
    global A2 and regional RMSE remain the main climate gates.
    """
    checks: dict[str, bool] = {}
    details: dict[str, dict] = {}
    if not _finite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")

    checks["verdict_pass"] = (control.get("verdict") == "PASS"
                              and experiment.get("verdict") == "PASS")
    checks["coverage_complete"] = all(
        data.get("coverage_complete", True) is True
        for data in (control, experiment))

    for metric in ["global_a2_rmse", "na_raw_rmse", "near_wall_raw_bias"]:
        base = _metric(control, metric)
        candidate = _metric(experiment, metric)
        if not _finite(base) or not _finite(candidate):
            checks[f"{metric}_not_worse"] = False
        elif metric == "near_wall_raw_bias":
            checks[f"{metric}_not_worse"] = abs(float(candidate)) <= abs(float(base)) + tolerance
        else:
            checks[f"{metric}_not_worse"] = 0.0 <= float(candidate) <= float(base) + tolerance
        details[metric] = {"control": base, "experiment": candidate}

    for metric in ["heat_drift", "salt_drift"]:
        base = _metric(control, metric)
        candidate = _metric(experiment, metric)
        limit = max(abs(float(base)), 1e-6) * 2.0 if _finite(base) else None
        checks[f"{metric}_bounded"] = (
            limit is not None and _finite(candidate) and abs(float(candidate)) <= limit)
        details[metric] = {"control": base, "experiment": candidate,
                           "limit_percent": limit}

    if expected_days is not None:
        if not _finite(expected_days) or expected_days <= 0:
            raise ValueError("expected_days must be finite and positive")
        duration = experiment.get("days_end")
        checks["duration_complete"] = _finite(duration) and float(duration) >= expected_days
    else:
        checks["duration_complete"] = True

    return {"pass": all(checks.values()), "checks": checks,
            "details": details}


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply pre-registered benchmark acceptance gates.")
    parser.add_argument("--control", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--days", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = evaluate_gate(_load(args.control), _load(args.experiment),
                           expected_days=args.days, tolerance=args.tolerance)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
