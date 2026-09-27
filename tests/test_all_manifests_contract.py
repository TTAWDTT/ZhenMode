"""Lock all standardized manifests to the benchmark contract."""
import json
from pathlib import Path

from benchmark_contract import validate_contract

ROOT = Path(__file__).parents[1] / "research/experiments"


def test_all_manifests_are_contract_ready():
    paths = sorted(ROOT.rglob("*manifest*.json"))
    assert paths
    failures = []
    for path in paths:
        report = validate_contract(json.loads(path.read_text(encoding="utf-8")))
        if not report["contract_pass"]:
            failures.append(str(path))
    assert failures == []
