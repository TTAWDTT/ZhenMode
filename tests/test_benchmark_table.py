"""Benchmark comparison table tests."""
import json

from benchmark_table import load_metric, markdown_table


def test_load_metric_flattens_fields(tmp_path):
    path = tmp_path / "run.json"
    data = {
        "verdict": "PASS",
        "days_end": 365.0,
        "global": {"raw_bias": -0.1, "raw_rmse": 0.5},
        "global_a2_rmse_c": 1.1,
        "north_atlantic_40_60": {"raw_rmse": 0.9},
        "near_wall_55_60": {"raw_rmse": 1.0},
        "ice": {"n_cells": 0},
        "heat_drift_percent": -0.1,
        "salt_drift_percent": -0.001,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    row = load_metric(path, "baseline")
    assert row["label"] == "baseline"
    assert row["global_a2_rmse"] == 1.1
    assert row["near_wall_raw_rmse"] == 1.0
    assert markdown_table([row]).startswith("| run |")


def test_markdown_table_escapes_pipes():
    row = {"label": "a|b", "verdict": "PASS", "days_end": 365,
           "global_a2_rmse": 1.0, "na_raw_rmse": None,
           "near_wall_raw_rmse": None, "raw_bias": None,
           "heat_drift": None, "salt_drift": None}
    text = markdown_table([row])
    assert "baseline" not in text
    assert "\\|" in text

