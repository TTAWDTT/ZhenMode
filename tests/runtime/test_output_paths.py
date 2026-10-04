"""Existing run results survive both early conflicts and a late writer race."""

import sys

import pytest

from tests.support.driver import run_controlled_driver
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
