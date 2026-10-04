"""Real expansion → manifest → scorer contract using declared array fixtures.

This is a protocol identity witness, not a claim that these arrays came from a
model integration. No evaluator or hash implementation is mocked.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from zhenmode.evaluation.pipeline import evaluate
from zhenmode.evaluation.protocols import digest
from zhenmode.execution.resolve import expand_experiment
from zhenmode.execution.runs import create_run, write_json

ROOT = Path(__file__).resolve().parents[2]


def test_protocol_byte_and_content_hashes_cross_real_pipeline(tmp_path):
    experiment = yaml.safe_load((ROOT / "experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml").read_text(encoding="utf-8"))
    for name, origin in (("case.yaml", "cases/synthetic-production-smoke-80s.yaml"),
                         ("preset.yaml", "configs/zhenmode/presets/synthetic-smoke.yaml"),
                         ("protocol.json", "protocols/production-smoke-v1.json")):
        (tmp_path / name).write_bytes((ROOT / origin).read_bytes())
    experiment.update(case="case.yaml", parent_preset="preset.yaml", protocol_path="protocol.json")
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment, allow_unicode=True), encoding="utf-8")
    expanded = expand_experiment("experiment.yaml", tmp_path)
    directory, manifest = create_run(expanded, tmp_path, tmp_path / "outputs")
    assert manifest["protocol_file_sha256"] == hashlib.sha256((tmp_path / "protocol.json").read_bytes()).hexdigest()
    assert manifest["protocol_content_sha256"] == digest(json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8")))
    assert manifest["protocol_content_sha256"] != manifest["protocol_file_sha256"]
    times = np.array(expanded["output_times_s"]) / 86400
    # Deliberately declared analytical array fixture, with no dynamic oracle.
    field = np.arange(64, dtype=float).reshape(8, 8) / 100 + 17
    path = directory / "fixture.npz"
    np.savez(path, days=times, T_top=np.stack([field] * len(times)), T_init=field[:, :, None],
             wet_mask=np.ones((8, 8), dtype=bool), lat=np.linspace(-30, 30, 8), lon=np.arange(8) * 45 + 22.5,
             verdict="PASS", max_u_peak=0., max_eta=np.zeros(len(times)), heat_content_J=np.ones(len(times)),
             salt_content_kg=np.ones(len(times)))
    manifest.update(execution_status="completed", result_path=str(path.resolve()),
                    result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_json(directory / "manifest.json", manifest)
    report = evaluate(path, tmp_path / "protocol.json", directory / "manifest.json", directory / "evaluation")
    assert report["effect"]["global"]["raw_rmse"] == 0
    assert report["acceptance"]["status"] == "not_declared"
    assert report["comparability"]["status"] == "limited"  # no invented execution receipt
    # Same semantic JSON with different bytes is still a different frozen file.
    (tmp_path / "protocol.json").write_text(json.dumps(expanded["protocol"]), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol .*changed"):
        evaluate(path, tmp_path / "protocol.json", directory / "manifest.json", directory / "rejected")
