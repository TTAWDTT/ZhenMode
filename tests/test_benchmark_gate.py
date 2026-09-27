"""Tests for the pre-registered benchmark gate."""
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


def test_gate_rejects_positive_near_wall_overshoot():
    result = evaluate_gate(_run(), _run(global_a2=0.8, na=0.8, wall=1.2))
    assert not result["pass"]
    assert not result["checks"]["near_wall_raw_bias_not_worse"]


def test_gate_accepts_signed_near_wall_bias_toward_zero():
    result = evaluate_gate(_run(), _run(global_a2=0.8, na=0.8, wall=0.5))
    assert result["checks"]["near_wall_raw_bias_not_worse"]
