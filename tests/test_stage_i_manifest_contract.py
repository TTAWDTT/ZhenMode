"""Contract checks for Stage-I minimal sea-ice/mixed-layer manifests."""
import json
from pathlib import Path

from benchmark_contract import validate_contract


def test_stage_i_30d_manifest_is_contract_ready():
    path = Path("research/experiments/mixed_layer_ice_050/stage_i_minimal_closed_loop_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = validate_contract(manifest)
    assert result["contract_pass"]

    assert manifest["model"] == "ocean_solver"
    assert manifest["run_id"] == "stage_i_minimal_closed_loop_050_30d_monthly"
    assert manifest["status"] == "completed_diagnostic"
    assert manifest["duration"]["days"] == 30.0
    assert manifest["duration"]["status"] == "comparable"


def test_stage_i_annual_manifest_is_contract_ready():
    path = Path("research/experiments/mixed_layer_ice_050/stage_i_annual_dynamic_ice_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result = validate_contract(manifest)
    assert result["contract_pass"]

    assert manifest["model"] == "ocean_solver"
    assert manifest["run_id"] == "stage_i_minimal_closed_loop_050_365d_monthly"
    assert manifest["status"] == "completed_diagnostic"
    assert manifest["duration"]["days"] == 365.0
    assert manifest["duration"]["status"] == "comparable"
