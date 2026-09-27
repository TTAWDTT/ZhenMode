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
