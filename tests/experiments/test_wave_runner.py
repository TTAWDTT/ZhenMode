"""Independent source-revision and GPU-selection controls for the installed runner."""

from types import SimpleNamespace

import pytest

from zhenmode.execution import native_channel


def test_installed_package_does_not_take_unrelated_cwd_commit(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    installed = tmp_path / "installed"
    installed.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(native_channel, "source_root", lambda where: installed / "site-packages")
    with pytest.raises(ValueError, match="installed runs require"):
        native_channel.reported_revision()
    assert native_channel.reported_revision("a" * 40) == "a" * 40


def test_checkout_revision_query_is_scoped_to_actual_package(tmp_path, monkeypatch):
    checkout = tmp_path / "product"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    monkeypatch.setattr(native_channel, "source_root", lambda where: checkout / "src")
    observed = []

    def query(command, **kwargs):
        observed.append(command)
        return "a" * 40

    monkeypatch.setattr(native_channel.subprocess, "check_output", query)
    assert native_channel.reported_revision() == "a" * 40
    assert observed[0][:3] == ["git", "-C", str(checkout)]


def test_launcher_preserves_caller_gpu_visibility(tmp_path, monkeypatch):
    observed = []

    def supervise(command, **kwargs):
        observed.append(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
        return SimpleNamespace(returncode=0), {
            "elapsed_wall_seconds": 1,
            "peak_host_rss_bytes": 100,
        }

    monkeypatch.setattr(native_channel, "run_cuda_worker", supervise)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-task-device")
    native_channel.launch(tmp_path / "config.json", "zhenmode", tmp_path)
    assert observed == ["GPU-task-device"]


def test_runtime_receipt_identifies_real_interpreter_and_jax_binary_files():
    from zhenmode.execution.wave_zhenmode import runtime_identity
    from zhenmode.provenance.sources import sha256_file

    info = runtime_identity()
    assert info["actual_executable_sha256"] == sha256_file(info["actual_executable"])
    assert info["packages"]["jax"]
    assert info["packages"]["jaxlib"]
    assert info["native_jaxlib_sha256"]
    assert all(len(value) == 64 for value in info["native_jaxlib_sha256"].values())


def test_study_entry_routes_the_new_method_and_rejects_ignored_half_amplitude(tmp_path, monkeypatch):
    from zhenmode.benchmarks.standing_wave import METHOD_STUDY_SCHEMA
    from zhenmode.execution import standing_wave

    seen = []
    monkeypatch.setattr(standing_wave, "execute", lambda c, *a, **k: seen.append(c))
    standing_wave.run("coarse", tmp_path / "run", study=(128, 25), method="symmetric-external-mode")
    assert seen[0]["schema"] == METHOD_STUDY_SCHEMA
    assert seen[0]["nx"] == 128 and seen[0]["dt"] == 25
    assert seen[0]["ocean_options"]["external_mode_scheme"] == "symmetric"
    with pytest.raises(ValueError, match="fixed full amplitude"):
        standing_wave.run("coarse", tmp_path / "ignored-half", study=(64, 25), half=True)


def test_freeze_cli_routes_method_study_and_preserves_default(tmp_path):
    import json

    from zhenmode.benchmarks.standing_wave import study_contract
    from zhenmode.evaluation.standing_wave import main

    target = tmp_path / "candidate.json"
    main(["freeze", "--study", "64", "25", "--method", "symmetric-external-mode", "--out", str(target)])
    assert json.loads(target.read_text()) == study_contract(64, 25, "symmetric-external-mode")
    legacy = tmp_path / "baseline.json"
    main(["freeze", "--study", "64", "25", "--out", str(legacy)])
    assert json.loads(legacy.read_text()) == study_contract(64, 25)
