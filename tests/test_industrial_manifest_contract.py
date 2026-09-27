"""Regression test for the paired annual external-comparison manifests."""
import json
from pathlib import Path

from benchmark_contract import validate_contract

ROOT = Path(__file__).parents[1] / "research/experiments/industrial_comparison_045"


def test_paired_annual_manifests_are_contract_ready():
    reports = [validate_contract(json.loads(path.read_text(encoding="utf-8")))
               for path in [ROOT / "ocean_solver_stage_f_365d_3d_manifest.json",
                            ROOT / "mom6_stage_f_dynamic_365d_v12_manifest.json"]]
    assert all(report["contract_pass"] for report in reports)
    assert all(not report["missing_sections"] for report in reports)
    assert all("provenance" in report["not_comparable_sections"] for report in reports)


def test_stage_i_manifest_is_contract_ready():
    report = validate_contract(json.loads(
        (ROOT / "ocean_solver_stage_f_dynamic_ice_constant_mld_lat40_60_probe_30d_manifest.json")
        .read_text(encoding="utf-8")))
    assert report["contract_pass"] is True
    assert report["missing_sections"] == []
    assert "provenance" in report["not_comparable_sections"]


def test_direct_comparison_manifests_are_contract_ready():
    for name in [
        "wind_only_30d_manifest.json",
        "restore_30d_manifest.json",
        "stage_f_30d_manifest.json",
        "ocean_solver_stage_f_dyn_ice_band40_65_mld100_lat40_60_cooling_ice_gate_365d_manifest.json",
    ]:
        report = validate_contract(json.loads((ROOT / name).read_text(encoding="utf-8")))
        assert report["contract_pass"] is True, name
        assert report["missing_sections"] == []
