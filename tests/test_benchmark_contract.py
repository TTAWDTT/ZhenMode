"""Tests for the standardized benchmark contract validator."""
import json
from pathlib import Path

from benchmark_contract import validate_contract


def _base_manifest(**overrides):
    data = {
        "model": "MOM6",
        "run_id": "test_run",
        "status": "running",
        "grid": {"status": "comparable"},
        "bathymetry": {"status": "comparable"},
        "initial_state": {"status": "comparable"},
        "forcing": {"status": "comparable"},
        "sea_ice": {"status": "comparable"},
        "duration": {"status": "pre_registered"},
        "scoring": {"status": "comparable"},
        "provenance": {"status": "not_comparable"},
    }
    data.update(overrides)
    return data


def test_valid_contract_passes_with_explicit_not_comparable():
    report = validate_contract(_base_manifest())
    assert report["contract_pass"] is True
    assert report["missing_sections"] == []
    assert report["not_comparable_sections"] == ["provenance"]


def test_invalid_status_fails_closed():

    manifest = _base_manifest()
    manifest["grid"] = {"status": "unknown"}
    report = validate_contract(manifest)
    assert report["contract_pass"] is False
    assert "grid:invalid_status" in report["not_comparable_sections"]


def test_missing_section_and_status_fail_closed():
    manifest = _base_manifest()
    del manifest["forcing"]
    del manifest["sea_ice"]["status"]
    report = validate_contract(manifest)
    assert report["contract_pass"] is False
    assert "forcing" in report["missing_sections"]
    assert "sea_ice:missing_status" in report["not_comparable_sections"]


def test_unknown_manifest_fails_closed():
    report = validate_contract({})
    assert report["contract_pass"] is False
    assert report["missing_sections"] == [
        "model", "run_id", "status", "grid", "bathymetry", "initial_state",
        "forcing", "sea_ice", "duration", "scoring", "provenance"]
    assert report["not_comparable_sections"] == [
        "grid:missing_status", "bathymetry:missing_status", "initial_state:missing_status",
        "forcing:missing_status", "sea_ice:missing_status", "duration:missing_status",
        "scoring:missing_status", "provenance:missing_status"]


def test_cli_exits_nonzero_on_missing_section(tmp_path):
    import subprocess
    import sys
    manifest = {"model": "x"}
    path = Path("/tmp/test_missing_section.json")
    path.write_text(json.dumps(manifest))
    result = subprocess.run([sys.executable, "src/benchmark_contract.py", str(path)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert json.loads(result.stdout)["contract_pass"] is False
