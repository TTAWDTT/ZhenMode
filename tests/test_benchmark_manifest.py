"""Manifest tests for the standardized benchmark protocol."""
import json

import numpy as np

from benchmark_manifest import make_manifest


def _write_run(path):
    np.savez(
        path,
        days=np.array([0.0, 365.0]),
        T_top=np.stack([np.full((2, 2), 20.0), np.full((2, 2), 21.0)]),
        T_init=np.full((2, 2, 1), 20.0),
        wet_mask=np.ones((2, 2), dtype=bool),
        lat=np.array([30.0, 50.0]),
        lon=np.array([300.0, 320.0]),
        max_u_peak=np.array(1.0),
        max_eta=np.array([0.1, 0.2]),
        heat_content_J=np.array([1e12, 1e12]),
        salt_content_kg=np.array([1e12, 1e12]),
        verdict=np.array("PASS"),
        config=np.array("{'resolution': 0.5, 'days': 365.0}"))


def test_make_manifest_records_config_and_metrics(tmp_path):
    npz_path = tmp_path / "run.npz"
    _write_run(npz_path)
    manifest = make_manifest(npz_path, commit="abc123")
    assert manifest["model"] == "ocean_solver"
    assert manifest["commit"] == "abc123"
    assert manifest["config"]["resolution"] == 0.5
    assert manifest["metrics"]["verdict"] == "PASS"


def test_make_manifest_accepts_precomputed_metrics(tmp_path):
    npz_path = tmp_path / "run.npz"
    metrics_path = tmp_path / "metrics.json"
    _write_run(npz_path)
    metrics_path.write_text(json.dumps({"verdict": "PASS", "days_end": 365.0}),
                            encoding="utf-8")
    manifest = make_manifest(npz_path, metrics_path=metrics_path)
    assert manifest["metrics"]["days_end"] == 365.0


def test_make_manifest_supports_external_model(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    config_path = tmp_path / "config.json"
    metrics_path.write_text(json.dumps({"verdict": "PASS", "days_end": 365.0}),
                            encoding="utf-8")
    config_path.write_text(json.dumps(
        {"model": "MOM6", "resolution_deg": 0.5}), encoding="utf-8")
    manifest = make_manifest("unused.npz", metrics_path=metrics_path,
                             model="MOM6", config_json=config_path)
    assert manifest["model"] == "MOM6"
    assert manifest["config"]["resolution_deg"] == 0.5

def test_make_manifest_adds_run_id_and_status(tmp_path):
    npz_path = tmp_path / "candidate_365d.npz"
    _write_run(npz_path)
    manifest = make_manifest(npz_path)
    assert manifest["run_id"] == "candidate_365d"
    assert manifest["status"] == "completed"


def test_make_manifest_supports_pre_registered_no_npz(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"days": 365.0}), encoding="utf-8")
    manifest = make_manifest(None, config_json=config_path,
                             run_id="candidate_365d", status="pre_registered")
    assert manifest["npz"] is None
    assert manifest["metrics"] is None
    assert manifest["config"]["days"] == 365.0
    assert manifest["status"] == "pre_registered"


def test_pre_registered_requires_config_and_run_id(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"days": 365.0}), encoding="utf-8")
    try:
        make_manifest(None, config_json=config_path,
                      status="pre_registered")
    except ValueError as exc:
        assert "--run-id" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing run_id")
