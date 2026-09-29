"""Production CLI histories and retained output files, using a controlled small FD grid."""
import sys
from dataclasses import replace

import numpy as np
import pytest
from _helpers import all_wet_grid

import run_long_integration_global as driver
from restart_contract import load_restart


def _run_driver(monkeypatch, directory, *, restart=None, crash_after=None, save_3d=False):
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    grid = replace(grid, f=np.zeros((8, 8)))
    monkeypatch.setattr(driver, "make_global_grid", lambda *args, **kwargs: grid)
    monkeypatch.setattr(driver, "get_initial_fields", lambda grid:
                        (np.full((8, 8, 4), 17.), np.full((8, 8, 4), 35.)))
    monkeypatch.setattr(driver, "build_seasonal_wind_global", lambda grid, year:
                        [(np.full((8, 8), (month + 1) * 0.001), np.zeros((8, 8)))
                         for month in range(12)])
    original_factory = driver.make_solver_global
    contract = []
    original_contract = driver.make_restart_contract

    def build_contract(*args, **kwargs):
        value = original_contract(*args, **kwargs)
        contract.append(value)
        return value

    monkeypatch.setattr(driver, "make_restart_contract", build_contract)

    def factory(*args, **kwargs):
        result = list(original_factory(*args, **kwargs))
        original_step = result[-1]
        calls = 0

        def step(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == crash_after:
                raise RuntimeError("controlled interruption")
            return original_step(*args, **kwargs)

        result[-1] = step
        return tuple(result)

    monkeypatch.setattr(driver, "make_solver_global", factory)
    arguments = ["ocean-solver", "--days", str(80. / 86400.), "--dt", "10", "--dt-bt", "5",
                 "--mode-split", "--seasonal-wind", "--snap-days", str(20. / 86400.),
                 "--checkpoint-days", str(20. / 86400.), "--tag", "controlled",
                 "--out-dir", str(directory), "--log-dir", str(directory), "--nu-h", "0",
                 "--nu-bi", "0", "--kappa-v", "0", "--kappa-conv", "0",
                 "--polar-cap-rows", "0", "--polar-cap-taper", "0",
                 "--no-meridional-heat-flux", "--no-bulk-flux"]
    if save_3d:
        arguments.extend(["--save-3d", "--save-3d-terms"])
    if restart:
        arguments.extend(["--restart-from", str(restart)])
    monkeypatch.setattr(sys, "argv", arguments)
    previous_stdout = sys.stdout
    try:
        if crash_after is None:
            driver.main()
        else:
            with pytest.raises(RuntimeError, match="controlled interruption"):
                driver.main()
    finally:
        if isinstance(sys.stdout, driver._Tee):
            sys.stdout.file.close()
        sys.stdout = previous_stdout
    return grid, contract[0]


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
        driver._validate_restart_history(altered, grid, 2, 10., save_3d, save_3d,
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
        driver._validate_restart_history(saved, grid, 2, 10., True, True, directories)
    path.unlink()
    with pytest.raises(ValueError, match="missing or changed"):
        driver._validate_restart_history(saved, grid, 2, 10., True, True, directories)


def test_snapshot_collision_does_not_overwrite_existing_data(tmp_path):
    path = tmp_path / "snapshot.npy"
    path.write_bytes(b"valuable previous output")
    with pytest.raises(FileExistsError):
        driver._save_snapshot_file(path, np.zeros((2, 2)), {}, "3d/snapshot.npy")
    assert path.read_bytes() == b"valuable previous output"
