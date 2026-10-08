"""Reproducible Stage-I closed-loop manifest tests."""
import json

import numpy as np
import pytest

from zhenmode.evaluation.ice import summarize_ice_closed_loop


def _write_run(tmp_path):
    npz_path = tmp_path / "run.npz"
    benchmark_path = tmp_path / "benchmark.json"
    days = np.array([0.0, 10.0])
    ice = np.zeros((2, 3, 3))
    ice[1, 0, 0] = 1.0
    ice[1, 1, 1] = 2.0
    wet = np.ones((3, 3), dtype=bool)
    np.savez(
        npz_path,
        days=days,
        ice_top=ice,
        T_init=np.full((3, 3, 1), 20.0),
        S_init=np.full((3, 3, 1), 35.0),
        wet_mask=wet,
        lat=np.array([30.0, 45.0, 55.0]),
        lon=np.array([300.0, 320.0, 0.0]),
        verdict=np.array("PASS"),
        max_u_peak=np.array(1.0),
        max_eta_last=np.array([0.1, 0.2]),
        heat_content_J=np.array([1e12, 1e12]),
        salt_content_kg=np.array([1e12, 1e12]),
        config="{}")
    benchmark = {
        "verdict": "PASS",
        "global_a2_rmse_c": 0.9,
        "north_atlantic_40_60": {"raw_rmse": 0.8, "raw_bias": -0.7},
        "near_wall_55_60": {"raw_bias": -0.6},
        "ice": {"n_cells": 2, "max_thickness_m": 2.0},
        "mld": {"mean_m": 20.0, "median_m": 20.0, "p90_m": 20.0},
        "heat_drift_percent": 0.0,
        "salt_drift_percent": 0.0,
        "max_u_peak": 1.0,
        "max_eta_last": 0.2,
    }
    benchmark_path.write_text(json.dumps(benchmark))
    return npz_path, benchmark_path


def test_stage_i_manifest_records_ice_growth_and_budget(tmp_path):
    npz_path, benchmark_path = _write_run(tmp_path)
    result = summarize_ice_closed_loop(npz_path, benchmark_path, "stage_i_test")
    assert result["climate_score"]["global_a2_rmse_c"] == 0.9
    assert result["ice"]["growth_volume_m3"] > 0.0
    assert result["ice"]["melt_volume_m3"] == 0.0
    assert result["surface_budget"]["latent_heat_growth_J"] > 0.0
    assert result["stability"]["verdict"] == "PASS"
    assert result["surface_budget"]["heat_budget_residual_J"] is None
    assert result["surface_budget"]["mean_residual_W_m2"] is None


@pytest.mark.parametrize("units,expected", [(None, 1e6), ("kg", 1e9)])
def test_salt_change_respects_diagnostics_units(tmp_path, units, expected):
    npz_path, benchmark_path = _write_run(tmp_path)
    with np.load(npz_path) as saved:
        fields = dict(saved)
    fields["salt_content_kg"] = np.array([1e12, 1e12 + 1e9])
    if units is not None:
        fields["salt_content_units"] = units
    np.savez(npz_path, **fields)
    result = summarize_ice_closed_loop(npz_path, benchmark_path, "units")
    assert result["brine_salt_flux"]["salt_content_change_kg"] == expected


def test_residual_requires_and_subtracts_recorded_surface_input(tmp_path):
    npz_path, benchmark_path = _write_run(tmp_path)
    before = summarize_ice_closed_loop(npz_path, benchmark_path, "before")
    with np.load(npz_path) as saved:
        fields = dict(saved)
    enthalpy_change = before["surface_budget"]["water_ice_enthalpy_change_J"]
    fields["surface_heat_input_J"] = np.array([0., enthalpy_change])
    np.savez(npz_path, **fields)
    after = summarize_ice_closed_loop(npz_path, benchmark_path, "after")
    assert after["surface_budget"]["heat_budget_residual_J"] == 0.
    assert after["surface_budget"]["mean_residual_W_m2"] == 0.
    assert after["surface_budget"]["budget_scope"] == "water_ice_enthalpy_minus_recorded_surface_heat_input"
