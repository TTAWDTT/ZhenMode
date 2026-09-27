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


def _run_3d(global_3d=1.5, na_3d=1.2, wall_3d=1.0, mld_bias=100.0, mld_rmse=500.0):
    return {
        "global_3d": {"raw_rmse": global_3d},
        "north_atlantic_40_60_3d": {"raw_rmse": na_3d},
        "near_wall_55_60_3d": {"raw_rmse": wall_3d},
        "mld": {"raw_bias_m": mld_bias, "raw_rmse_m": mld_rmse},
        "depth_mask_applied": True,
    }


def test_gate_ignores_3d_when_not_provided():
    result = evaluate_gate(_run(), _run())
    assert result["pass"]
    assert "three_d_complete" not in result["checks"]


def test_gate_accepts_3d_and_mld_improvement():
    result = evaluate_gate(_run(), _run(), control_3d=_run_3d(),
                           experiment_3d=_run_3d(global_3d=1.4, na_3d=1.1,
                                                  wall_3d=0.9, mld_bias=80.0,
                                                  mld_rmse=450.0))
    assert result["pass"]


def test_gate_rejects_partial_3d_input():
    result = evaluate_gate(_run(), _run(), experiment_3d=_run_3d())
    assert not result["pass"]
    assert not result["checks"]["three_d_complete"]


def test_gate_rejects_3d_and_mld_regression():
    result = evaluate_gate(_run(), _run(), control_3d=_run_3d(),
                           experiment_3d=_run_3d(global_3d=1.6, na_3d=1.3,
                                                  wall_3d=1.1, mld_bias=120.0,
                                                  mld_rmse=550.0))
    assert not result["pass"]
    assert not result["checks"]["three_d_complete"]


def test_gate_requires_depth_mask_when_3d_is_checked():
    control = _run_3d()
    experiment = _run_3d()
    result = evaluate_gate(_run(), _run(), control_3d=control,
                           experiment_3d=experiment)
    assert result["checks"]["depth_mask_applied"]

    experiment["depth_mask_applied"] = False
    result = evaluate_gate(_run(), _run(), control_3d=control,
                           experiment_3d=experiment)
    assert not result["checks"]["depth_mask_applied"]
    assert not result["pass"]
