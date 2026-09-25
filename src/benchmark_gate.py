"""Pre-registered pass/fail gates for standardized benchmark comparisons."""
from __future__ import annotations

import argparse
import json
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


def evaluate_gate(control: dict, experiment: dict,
                  expected_days: float | None = None,
                  tolerance: float = 1e-6) -> dict:
    """Apply the pre-registered internal candidate gates.

    A smaller negative heat/salt drift is better.  A near-wall negative bias
    closer to zero is better, but global A2 and regional RMSE remain the main
    climate gates.
    """
    checks: dict[str, bool] = {}
    details: dict[str, dict] = {}

    checks["verdict_pass"] = (control.get("verdict") == "PASS"
                              and experiment.get("verdict") == "PASS")

    for metric in ["global_a2_rmse", "na_raw_rmse", "near_wall_raw_bias"]:
        base = _metric(control, metric)
        candidate = _metric(experiment, metric)
        if base is None or candidate is None:
            checks[f"{metric}_not_worse"] = False
        elif metric == "near_wall_raw_bias":
            checks[f"{metric}_not_worse"] = candidate >= base - tolerance
        else:
            checks[f"{metric}_not_worse"] = candidate <= base + tolerance
        details[metric] = {"control": base, "experiment": candidate}

    for metric in ["heat_drift", "salt_drift"]:
        base = _metric(control, metric)
        candidate = _metric(experiment, metric)
        limit = max(abs(float(base or 0.0)), 1e-6) * 2.0
        checks[f"{metric}_bounded"] = abs(float(candidate or 0.0)) <= limit
        details[metric] = {"control": base, "experiment": candidate,
                           "limit_percent": limit}

    if expected_days is not None:
        checks["duration_complete"] = float(
            experiment.get("days_end") or 0.0) >= float(expected_days)
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
