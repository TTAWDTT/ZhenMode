"""Independent negative contracts for assembly and run records; no JAX runs."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from zhenmode.execution.options import decode_options
from zhenmode.execution.resolve import expand_experiment, expand_sweep, read_preset
from zhenmode.execution.runs import list_runs, run_experiment
from zhenmode.execution.schema import ConfigurationError, load_document
from zhenmode.provenance.sources import sha256_file as file_hash

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


@pytest.fixture
def catalog(tmp_path):
    case = yaml.safe_load((ROOT / "cases/synthetic-production-smoke-80s.yaml").read_text(encoding="utf-8"))
    preset = yaml.safe_load((ROOT / "configs/zhenmode/presets/synthetic-smoke.yaml").read_text(encoding="utf-8"))
    experiment = yaml.safe_load((ROOT / "experiments/zhenmode/synthetic-smoke/synthetic-smoke-baseline.yaml").read_text(encoding="utf-8"))
    experiment.update(case="case.yaml", parent_preset="preset.yaml", protocol_path="protocol.json")
    save(tmp_path / "case.yaml", case)
    save(tmp_path / "preset.yaml", preset)
    save(tmp_path / "experiment.yaml", experiment)
    (tmp_path / "protocol.json").write_bytes((ROOT / "protocols/production-smoke-v1.json").read_bytes())
    return tmp_path, case, preset, experiment


def test_full_defaults_and_real_method(catalog):
    root, _, _, _ = catalog
    result = expand_experiment("experiment.yaml", root)
    assert result["effective_grid"] == {"nx": 8, "ny": 8, "nz": 4, "kind": "synthetic"}
    assert result["output_times_s"] == [0, 20, 40, 60, 80]
    assert result["requested_steps"] == 8
    assert result["runtime_options"]["projection_niter"] == 150
    assert result["effective_timestepping"] == {"barotropic_substeps": 2, "barotropic_dt_s": 5}
    assert result["method"] == "zhenmode"
    assert result["changes"] == []
    assert expand_experiment("experiment.yaml", root)["config_hash"] == result["config_hash"]


def test_human_title_does_not_change_config_hash(catalog):
    root, _, _, experiment = catalog
    before = expand_experiment("experiment.yaml", root)["config_hash"]
    experiment.update(id="another-id", name_zh="标题修改", purpose="Purpose text changed")
    save(root / "experiment.yaml", experiment)
    assert expand_experiment("experiment.yaml", root)["config_hash"] == before


def test_unknown_and_duplicate_fields(catalog):
    root, _, _, experiment = catalog
    experiment["surprise"] = True
    save(root / "experiment.yaml", experiment)
    with pytest.raises(ConfigurationError, match="unknown fields"):
        expand_experiment("experiment.yaml", root)
    (root / "bad.yaml").write_text("schema_version: 1\nkind: case\nkind: preset\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="duplicate YAML key"):
        load_document(root / "bad.yaml")


@pytest.mark.parametrize("options", [{"fake_solver": True}, {"dt": {"value": 10, "unit": "day"}}, {"dt": 10}, {"use_scan": "true"}, {"dtype": "fp16"}, {"resolution": {"value": 1, "unit": "deg"}}, {"out_dir": "/tmp/out"}])
def test_invalid_options_rejected(options):
    with pytest.raises(ConfigurationError):
        decode_options(options)


def test_cycle_and_ambiguous_inheritance(catalog):
    root, _, preset, _ = catalog
    preset["parent"] = "preset.yaml"
    save(root / "preset.yaml", preset)
    with pytest.raises(ConfigurationError, match="cyclic"):
        read_preset(root / "preset.yaml", root)
    preset.pop("parent")
    save(root / "parent.yaml", preset)
    preset["parent"] = "parent.yaml"
    save(root / "preset.yaml", preset)
    with pytest.raises(ConfigurationError, match="explicit before/after"):
        read_preset(root / "preset.yaml", root)


def change(option="kappa_v", before=0, after=1e-6):
    return {"option": option, "before": {"value": before, "unit": "m2/s"}, "after": {"value": after, "unit": "m2/s"}, "factor": "vertical_diffusion", "reason": "explicit test change"}


def test_single_factor_diff_is_explicit(catalog):
    root, _, _, experiment = catalog
    experiment.update(variant="single_factor", changes=[change()])
    save(root / "experiment.yaml", experiment)
    result = expand_experiment("experiment.yaml", root)
    assert result["runtime_options"]["kappa_v"] == 1e-6
    assert [item["option"] for item in result["changes"]] == ["kappa_v"]
    experiment["changes"].append(change("kappa_conv"))
    save(root / "experiment.yaml", experiment)
    with pytest.raises(ConfigurationError, match="exactly one"):
        expand_experiment("experiment.yaml", root)
    experiment["variant"] = "combination"
    save(root / "experiment.yaml", experiment)
    assert len(expand_experiment("experiment.yaml", root)["changes"]) == 2


def test_wrong_before_and_duplicate_override(catalog):
    root, _, _, experiment = catalog
    experiment.update(variant="single_factor", changes=[change(before=99)])
    save(root / "experiment.yaml", experiment)
    with pytest.raises(ConfigurationError, match="does not match"):
        expand_experiment("experiment.yaml", root)
    experiment.update(variant="combination", changes=[change(), change()])
    save(root / "experiment.yaml", experiment)
    with pytest.raises(ConfigurationError, match="repeated override"):
        expand_experiment("experiment.yaml", root)


@pytest.mark.parametrize("alter", ["unit", "negative_duration", "output", "protocol", "unknown_case", "data_checksum"])
def test_case_and_protocol_negatives(catalog, alter):
    root, case, _, _ = catalog
    if alter == "unit":
        case["grid"]["depth"]["unit"] = "km"
    elif alter == "negative_duration":
        case["duration"]["value"] = -1
    elif alter == "output":
        case["output_interval"]["value"] = 81
    elif alter == "unknown_case":
        case["unregistered_field"] = "x"
    elif alter == "data_checksum":
        case["data"] = [{"role": "bad", "path": "missing", "version": "v1", "sha256": "bad", "status": "available", "source": "test"}]
    else:
        (root / "protocol.json").write_text(json.dumps({"id": "wrong", "case_id": "wrong"}), encoding="utf-8")
    save(root / "case.yaml", case)
    with pytest.raises(ConfigurationError):
        expand_experiment("experiment.yaml", root)


def test_sweep_only_expands_and_estimates(catalog, monkeypatch):
    root, _, _, _ = catalog
    sweep = {"schema_version": 1, "kind": "sweep", "id": "kv-three", "name_zh": "垂向扩散扫描", "purpose": "dry only", "experiment": "experiment.yaml", "axes": [{"option": "kappa_v", "values": [{"value": value, "unit": "m2/s"} for value in (0, 1e-6, 2e-6)], "factor": "kv"}], "max_runs": 3}
    save(root / "sweep.yaml", sweep)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("dry sweep must not execute anything"))
    result = expand_sweep("sweep.yaml", root)
    assert result["dry_run"] and result["run_count"] == 3
    assert [run["configuration"]["variant"] for run in result["runs"]] == ["baseline", "tuning", "tuning"]
    assert all(run["resource_estimate"]["wall_seconds_estimate"] is None for run in result["runs"])
    sweep["max_runs"] = 2
    save(root / "sweep.yaml", sweep)
    with pytest.raises(ConfigurationError, match="exceeding"):
        expand_sweep("sweep.yaml", root)


def test_runs_never_overwrite_and_source_is_not_facade(catalog):
    root, _, _, _ = catalog
    expanded = expand_experiment("experiment.yaml", root)
    first = run_experiment(expanded, root, root / "outputs", dry_run=True)
    second = run_experiment(expanded, root, root / "outputs", dry_run=True)
    assert first["run_id"] != second["run_id"]
    assert first["config_hash"] == second["config_hash"]
    assert first["execution_status"] == "proposed"
    assert first["acceptance"]["status"] == "not_assessed"
    assert "model/solver/factory.py" in first["source_identity"]
    assert "model/solver/timestepping/step.py" in first["source_identity"]
    assert len(list_runs(root / "outputs")) == 2


@pytest.mark.parametrize("failure", ["exception", "timeout", "malicious_report"])
def test_failures_stay_failed(catalog, monkeypatch, failure):
    root, _, _, _ = catalog
    expanded = expand_experiment("experiment.yaml", root)
    real_run = subprocess.run
    def fake_run(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        if failure == "exception":
            raise OSError("simulated failure")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 180)
        directory = Path(kwargs["cwd"])
        (directory / "worker-report.json").write_text(json.dumps({"execution_status": "completed"}), encoding="utf-8")
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_experiment(expanded, root, root / "outputs")
    assert result["execution_status"] == "failed"
    assert json.loads((Path(result["run_directory"]) / "manifest.json").read_text(encoding="utf-8"))["execution_status"] == "failed"


def test_completed_integration_not_automatic_acceptance(catalog, monkeypatch):
    root, _, _, _ = catalog
    real_run = subprocess.run
    def fake_run(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        directory = Path(kwargs["cwd"])
        artifact = directory / "result.npz"
        artifact.write_bytes(b"test fixture artifact; no numeric execution")
        report = {"result_path": str(artifact), "result_sha256": file_hash(artifact), "duration_complete": True, "verdict": "FAIL"}
        (directory / "worker-report.json").write_text(json.dumps(report), encoding="utf-8")
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_experiment(expand_experiment("experiment.yaml", root), root, root / "outputs")
    assert result["execution_status"] == "completed"
    assert result["production_gate"]["status"] == "failed"
    assert result["acceptance"]["status"] == "not_assessed"


def test_evaluation_refusal_preserves_execution_status(catalog, monkeypatch):
    from zhenmode.evaluation import pipeline

    root, _, _, _ = catalog
    real_run = subprocess.run
    def fake_run(command, **kwargs):
        if command[0] == "git":
            return real_run(command, **kwargs)
        directory = Path(kwargs["cwd"])
        artifact = directory / "result.npz"
        artifact.write_bytes(b"deliberately invalid metric fixture")
        (directory / "worker-report.json").write_text(json.dumps({"result_path": str(artifact), "result_sha256": file_hash(artifact), "duration_complete": True, "verdict": "PASS"}), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    def reject(*args, **kwargs):
        raise ValueError("independent scoring rejection")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline, "evaluate", reject)
    result = run_experiment(expand_experiment("experiment.yaml", root), root, root / "outputs", evaluate=True)
    assert result["execution_status"] == "completed"
    assert result["evaluation"]["status"] == "failed"


def test_loader_roles_and_freeze_are_checked(tmp_path):
    from zhenmode.execution.worker import verify_selected_inputs

    jan, feb = tmp_path / "monthly_mean_900.npz", tmp_path / "monthly_mean_901.npz"
    jan.write_bytes(b"January bytes")
    feb.write_bytes(b"February bytes")
    manifest = {"data": {"wind-01": {"resolved_path": str(jan.resolve()), "observed_sha256": file_hash(jan)},
                         "wind-02": {"resolved_path": str(feb.resolve()), "observed_sha256": file_hash(feb)}}}
    options = {"wind_year": 2023}
    verify_selected_inputs({"wind_900": jan, "wind_901": feb}, manifest, options)
    with pytest.raises(ValueError, match="path differs"):
        verify_selected_inputs({"wind_900": feb, "wind_901": jan}, manifest, options)
    jan.write_bytes(b"modified after freeze")
    with pytest.raises(ValueError, match="changed after"):
        verify_selected_inputs({"wind_900": jan, "wind_901": feb}, manifest, options)


@pytest.mark.parametrize("alter", ["initial", "forcing", "seasonal", "spacing", "window"])
def test_synthetic_actual_contract_and_window(catalog, alter):
    root, case, _, _ = catalog
    if alter == "initial":
        case["initial"] = {"kind": "woa2023"}
    elif alter == "forcing":
        case["forcing"]["kind"] = "wind_only"
    elif alter == "seasonal":
        case["forcing"]["options"]["seasonal_wind"] = False
    elif alter == "spacing":
        case["grid"]["resolution"]["value"] = 1
    else:
        protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        protocol["window"]["end_day"] = 1
        (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    save(root / "case.yaml", case)
    with pytest.raises(ConfigurationError):
        expand_experiment("experiment.yaml", root)
