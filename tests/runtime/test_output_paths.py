"""Existing run results survive both early conflicts and a late writer race."""

import os
import subprocess
import sys

import numpy as np
import pytest

import zhenmode.model.io.restart as restart_io
from tests.support.driver import run_controlled_driver
from tests.support.paths import REPOSITORY_ROOT
from zhenmode.model.io.output import prepare_output_paths
from zhenmode.model.runtime.cli import parse_run_configuration


def test_existing_result_rejected_before_log_or_stdout_changes(tmp_path):
    result = tmp_path / "global_keep.npz"
    result.write_bytes(b"existing run bytes")
    log_dir = tmp_path / "logs"
    args = parse_run_configuration(["--tag", "keep", "--out-dir", str(tmp_path),
                                    "--log-dir", str(log_dir)]).args
    stdout = sys.stdout
    with pytest.raises(FileExistsError, match="result already exists"):
        prepare_output_paths(args)
    assert result.read_bytes() == b"existing run bytes"
    assert sys.stdout is stdout
    assert not log_dir.exists()


def test_result_created_after_preflight_cannot_be_overwritten(tmp_path, monkeypatch):
    result = tmp_path / "global_controlled.npz"

    def competing_writer(state, count):
        if count == 1:
            result.write_bytes(b"other writer's result")
        return state

    with pytest.raises(FileExistsError):
        run_controlled_driver(monkeypatch, tmp_path, step_override=competing_writer)
    assert result.read_bytes() == b"other writer's result"
    assert not list(tmp_path.glob('.global_controlled.npz.*.tmp'))


@pytest.mark.parametrize('failure', ['write', 'fsync', 'publish'])
def test_failed_final_archive_can_be_retried_without_partial_result(tmp_path, monkeypatch, failure):
    result = tmp_path / 'global_retry.npz'

    def fail(*args, **kwargs):
        raise OSError('injected archive failure')

    with monkeypatch.context() as patch:
        if failure != 'write':
            patch.setattr(restart_io.os, 'fsync' if failure == 'fsync' else 'link', fail)
        with pytest.raises(OSError, match='injected archive failure'):
            with restart_io.atomic_archive(result) as stream:
                if failure == 'write':
                    stream.write(b'partial archive')
                    fail()
                np.savez_compressed(stream, values=np.arange(3, dtype=np.float32))
    assert not result.exists() and not list(tmp_path.glob('.*.tmp'))
    with restart_io.atomic_archive(result) as stream:
        np.savez_compressed(stream, values=np.arange(3, dtype=np.float32))
    with np.load(result) as saved:
        np.testing.assert_array_equal(saved['values'], np.array([0., 1., 2.], dtype=np.float32))


def test_final_archive_is_complete_and_synced_before_publication(tmp_path, monkeypatch):
    result = tmp_path / 'global_complete.npz'
    events = []
    sync, link = os.fsync, os.link

    def sync_file(fd):
        sync(fd)
        events.append('fsync')

    def publish(source, destination):
        assert events == ['fsync'] and not result.exists()
        with np.load(source) as saved:
            np.testing.assert_array_equal(saved['values'], [2., 3., 5.])
        link(source, destination)
        events.append('publish')

    monkeypatch.setattr(restart_io.os, 'fsync', sync_file)
    monkeypatch.setattr(restart_io.os, 'link', publish)
    with restart_io.atomic_archive(result) as stream:
        np.savez_compressed(stream, values=[2., 3., 5.])
        assert not result.exists()
    assert events == ['fsync', 'publish'] and list(tmp_path.iterdir()) == [result]


def test_abrupt_exit_leaves_only_temporary_archive(tmp_path):
    result = tmp_path / 'global_abrupt.npz'
    environment = {**os.environ, 'PYTHONPATH': str(REPOSITORY_ROOT / 'src')}
    probe = subprocess.run([sys.executable, '-c',
        "import os, sys; from zhenmode.model.io.restart import atomic_archive; "
        "context=atomic_archive(sys.argv[1]); stream=context.__enter__(); "
        "stream.write(b'partial archive'); stream.flush(); os._exit(23)", str(result)],
        cwd=tmp_path, env=environment, timeout=20)
    assert probe.returncode == 23 and not result.exists()
    orphan, = tmp_path.glob('.global_abrupt.npz.*.tmp')
    assert orphan.read_bytes() == b'partial archive'
    with restart_io.atomic_archive(result) as stream:
        np.savez_compressed(stream, values=[1., 2.])
    assert orphan.read_bytes() == b'partial archive'
    with np.load(result) as saved:
        np.testing.assert_array_equal(saved['values'], [1., 2.])


def test_failed_final_serialization_allows_strict_checkpoint_republish(tmp_path, monkeypatch):
    result = tmp_path / 'global_controlled.npz'
    original = np.savez_compressed

    def fail_final(stream, **values):
        if os.path.basename(stream.name).startswith('.global_controlled.npz.'):
            stream.write(b'partial archive')
            raise OSError('interrupted final serialization')
        return original(stream, **values)

    with monkeypatch.context() as patch:
        patch.setattr(np, 'savez_compressed', fail_final)
        with pytest.raises(OSError, match='interrupted final serialization'):
            run_controlled_driver(patch, tmp_path)
    checkpoint = tmp_path / 'ckpt_controlled.npz'
    assert checkpoint.is_file() and not result.exists()
    assert not list(tmp_path.glob('.global_controlled.npz.*.tmp'))
    run_controlled_driver(monkeypatch, tmp_path, restart=checkpoint)
    with np.load(result) as saved:
        assert bool(saved['duration_complete']) and str(saved['verdict']) == 'PASS'
