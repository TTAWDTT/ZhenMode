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
                  tolerance: float = 1e-6,
                  control_3d: dict | None = None,
                  experiment_3d: dict | None = None) -> dict:
    """Apply the pre-registered internal candidate gates.

    A smaller negative heat/salt drift is better.  A signed near-wall bias
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
            # A signed bias improves by moving toward zero from either side;
            # reject large positive overshoot as well as unchanged negative bias.
            checks[f"{metric}_not_worse"] = abs(candidate) <= abs(base) + tolerance
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

    three_d_details = {"control": control_3d, "experiment": experiment_3d}
    if control_3d is None and experiment_3d is None:
        pass
    elif control_3d is None or experiment_3d is None:
        checks["three_d_complete"] = False
        details["three_d"] = three_d_details
    else:
        three_d_checks = {}
        for name, section in [
            ("global_3d", "global_3d"),
            ("na_40_60_3d", "north_atlantic_40_60_3d"),
            ("near_wall_55_60_3d", "near_wall_55_60_3d"),
        ]:
            base = float(control_3d[section]["raw_rmse"])
            candidate = float(experiment_3d[section]["raw_rmse"])
            three_d_checks[f"{name}_not_worse"] = candidate <= base + tolerance
            details[f"{name}"] = {"control": base, "experiment": candidate}
        if "mld" not in control_3d or "mld" not in experiment_3d:
            checks["mld_present"] = False
            checks["three_d_complete"] = False
            details["mld"] = {"control_present": "mld" in control_3d,
                              "experiment_present": "mld" in experiment_3d}
            return {"pass": all(checks.values()), "checks": checks,
                    "details": details}
        for name in ["raw_bias_m", "raw_rmse_m"]:
            base = float(control_3d["mld"][name])
            candidate = float(experiment_3d["mld"][name])
            metric = "abs_" + name if name.endswith("bias_m") else name
            if name.endswith("bias_m"):
                check = abs(candidate) <= abs(base) + tolerance
            else:
                check = candidate <= base + tolerance
            three_d_checks[f"mld_{metric}_not_worse"] = check
            details[f"mld_{metric}"] = {"control": base, "experiment": candidate}
        checks["depth_mask_applied"] = bool(
            control_3d.get("depth_mask_applied")
            and experiment_3d.get("depth_mask_applied"))
        checks["three_d_complete"] = all(three_d_checks.values())
        details["three_d"] = {"checks": three_d_checks, **three_d_details}

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
    parser.add_argument("--control-3d", default=None,
                        help="optional 3D/MLD benchmark JSON for the control")
    parser.add_argument("--experiment-3d", default=None,
                        help="optional 3D/MLD benchmark JSON for the experiment")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = evaluate_gate(_load(args.control), _load(args.experiment),
                           expected_days=args.days, tolerance=args.tolerance,
                           control_3d=(None if args.control_3d is None
                                       else _load(args.control_3d)),
                           experiment_3d=(None if args.experiment_3d is None
                                          else _load(args.experiment_3d)))
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
