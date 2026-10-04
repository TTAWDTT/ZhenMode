"""Production CLI histories and retained output files, using a controlled small FD grid."""
from dataclasses import replace

import numpy as np
import pytest

import zhenmode.model.io.records as owner_records
import zhenmode.model.io.recovery as owner_recovery
from tests.support.driver import run_controlled_driver as _run_driver
from zhenmode.model.io.restart import load_restart


@pytest.mark.parametrize("save_3d", [False, True])
def test_production_two_restarts_restore_complete_history_and_do_not_overwrite_boundary(tmp_path, monkeypatch, save_3d):
    continuous_dir = tmp_path / "continuous"
    resumed_dir = tmp_path / "resumed"
    with monkeypatch.context() as scoped:
        _run_driver(scoped, continuous_dir, save_3d=save_3d)
    with monkeypatch.context() as scoped:
        grid, contract = _run_driver(scoped, resumed_dir, crash_after=3, save_3d=save_3d)
    checkpoint = resumed_dir / "ckpt_controlled.npz"
    first = load_restart(checkpoint, contract)
    assert first.step == 2 and len(first.history["days"]) == 2
    with monkeypatch.context() as scoped:
        _run_driver(scoped, resumed_dir, restart=checkpoint, crash_after=3, save_3d=save_3d)
    second = load_restart(checkpoint, contract)
    assert second.step == 4 and len(second.history["days"]) == 3
    with monkeypatch.context() as scoped:
        _run_driver(scoped, resumed_dir, restart=checkpoint, save_3d=save_3d)
    with np.load(continuous_dir / "global_controlled.npz", allow_pickle=False) as continuous:
        with np.load(resumed_dir / "global_controlled.npz", allow_pickle=False) as resumed:
            for name in continuous.files:
                if name != "three_d_dir":
                    np.testing.assert_array_equal(continuous[name], resumed[name], err_msg=name)
            assert len(resumed["days"]) == 5
            assert int(resumed["n_3d_snaps"]) == (5 if save_3d else 0)
    if save_3d:
        for category in ("3d", "terms"):
            for reference in (continuous_dir / f"global_controlled_{category}").glob("*.npy"):
                np.testing.assert_array_equal(np.load(reference), np.load(
                    resumed_dir / f"global_controlled_{category}" / reference.name))
        assert len(list((resumed_dir / "global_controlled_3d").glob("*.npy"))) == 5
    altered = replace(second, history={**second.history, "days": second.history["days"] + 1.})
    with pytest.raises(ValueError, match="absolute snapshot timeline"):
        owner_recovery._validate_restart_history(altered, grid, 2, 10., save_3d, save_3d,
                                        {"3d": resumed_dir / "global_controlled_3d",
                                         "terms": resumed_dir / "global_controlled_terms"})


def test_production_missing_or_changed_retained_file_is_not_silently_ignored(tmp_path, monkeypatch):
    with monkeypatch.context() as scoped:
        grid, contract = _run_driver(scoped, tmp_path, crash_after=3, save_3d=True)
    saved = load_restart(tmp_path / "ckpt_controlled.npz", contract)
    directories = {"3d": tmp_path / "global_controlled_3d", "terms": tmp_path / "global_controlled_terms"}
    path = directories["3d"] / "snap_00000.npy"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        owner_recovery._validate_restart_history(saved, grid, 2, 10., True, True, directories)
    path.unlink()
    with pytest.raises(ValueError, match="missing or changed"):
        owner_recovery._validate_restart_history(saved, grid, 2, 10., True, True, directories)


def test_snapshot_collision_does_not_overwrite_existing_data(tmp_path):
    path = tmp_path / "snapshot.npy"
    path.write_bytes(b"valuable previous output")
    with pytest.raises(FileExistsError):
        owner_records._save_snapshot_file(path, np.zeros((2, 2)), {}, "3d/snapshot.npy")
    assert path.read_bytes() == b"valuable previous output"
