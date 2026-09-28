"""Tests for the pre-registered benchmark gate."""
import pytest

from benchmark_gate import evaluate_gate


def _run(verdict="PASS", global_a2=1.0, na=0.9, wall=-1.0,
         heat=-0.1, salt=-0.001, days=365.0):
    return {
        "verdict": verdict,
        "days_end": days,
        "global_a2_rmse_c": global_a2,
        "north_atlantic_40_60": {"raw_rmse": na},
        "near_wall_55_60": {"raw_bias": wall},
        "heat_drift_percent": heat,
        "salt_drift_percent": salt,
    }


def test_gate_accepts_broad_improvement():
    result = evaluate_gate(_run(), _run(global_a2=0.8, na=0.8, wall=-0.5,
                                        heat=-0.05))
    assert result["pass"]


def test_gate_rejects_global_worse_and_verdict_failure():
    result = evaluate_gate(_run(), _run(global_a2=1.2))
    assert not result["pass"]
    result = evaluate_gate(_run(), _run(verdict="FAIL"))
    assert not result["pass"]


def test_gate_requires_expected_duration():
    result = evaluate_gate(_run(), _run(days=30.0), expected_days=365.0)
    assert not result["pass"]


@pytest.mark.parametrize("field", ["heat_drift_percent", "salt_drift_percent"])
def test_gate_rejects_missing_budget(field):
    candidate = _run()
    candidate.pop(field)
    assert not evaluate_gate(_run(), candidate)["pass"]
    assert not evaluate_gate(candidate, _run())["pass"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_gate_rejects_nonfinite_budget(value):
    candidate = _run(heat=value)
    assert not evaluate_gate(candidate, candidate)["pass"]


def test_gate_rejects_bias_overshooting_into_warm_error():
    assert not evaluate_gate(_run(wall=-0.5), _run(wall=5.0))["pass"]
    assert evaluate_gate(_run(wall=-0.5), _run(wall=0.2))["pass"]


def test_gate_rejects_incomplete_coverage():
    candidate = dict(_run(), coverage_complete=False)
    assert not evaluate_gate(_run(), candidate)["pass"]
