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
