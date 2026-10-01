"""Versioned full-state checkpoints, corruption and incompatible restart controls."""
import copy
import hashlib
import json
from collections import namedtuple
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from _helpers import all_wet_grid

from jax_solver_global import JaxStateG
from restart_contract import (
    file_sha256,
    load_restart,
    make_restart_contract,
    save_restart,
)


def _fixture(dtype="float64"):
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    shape = (8, 8, 4)
    zero = np.zeros(shape, dtype=dtype)
    surface = np.zeros((8, 8), dtype=dtype)
    state = JaxStateG(zero, zero, zero - 1.8, zero + 35., surface, surface + 0.5)
    contract = make_restart_contract(
        grid, namedtuple("Params", ["dt"])(60.), dtype=dtype,
        forcing={"heat": np.zeros((8, 8))}, controls={"calendar": "360_day"},
        code_paths={"solver": Path(__file__).resolve().parents[1] / "src/jax_solver_global.py"},
        execution={"backend": "cpu"})
    return grid, state, contract


def _save(path, state, contract, step=12, **options):
    settings = dict(counters={"n_3d_snaps": 2}, cumulative={"salt": np.array([1., 2.])},
                    history={"days": np.array([0., 1.]), "heat": np.array([3., 4.])})
    settings.update(options)
    save_restart(path, state, contract, step=step, **settings)


def _rewrite(path, edit):
    with np.load(path, allow_pickle=False) as saved:
        payload = {name: np.array(saved[name]) for name in saved.files}
    edit(payload)
    np.savez_compressed(path, **payload)


def _edit_metadata(payload, edit):
    metadata = json.loads(str(payload["metadata_json"]))
    edit(metadata)
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["metadata_json"] = np.asarray(encoded)
    payload["metadata_sha256"] = np.asarray(hashlib.sha256(encoded.encode()).hexdigest())


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_checkpoint_preserves_all_fields_dtype_time_counters_history_and_cumulative(tmp_path, dtype):
    _, state, contract = _fixture(dtype)
    path = tmp_path / "checkpoint.npz"
    _save(path, state, contract)
    saved = load_restart(path, contract)
    for field in state._fields:
        np.testing.assert_array_equal(saved.state[field], getattr(state, field))
        assert saved.state[field].dtype == np.dtype(dtype)
    assert saved.step == 12 and saved.elapsed_seconds == 720.
    assert saved.counters == {"n_3d_snaps": 2}
    np.testing.assert_array_equal(saved.cumulative["salt"], [1., 2.])
    np.testing.assert_array_equal(saved.history["days"], [0., 1.])
    with np.load(path, allow_pickle=False) as raw:
        assert all(raw[name].dtype.kind != "O" for name in raw.files)


@pytest.mark.parametrize("field", ["grid", "effective_params", "forcing", "sources",
                                   "state_dtype", "execution", "controls", "dt_s"])
def test_incompatible_contract_is_not_a_resume(tmp_path, field):
    _, state, contract = _fixture()
    path = tmp_path / "checkpoint.npz"
    _save(path, state, contract)
    altered = copy.deepcopy(contract)
    altered[field] = "different"
    with pytest.raises(ValueError, match=f"contract mismatch: {field}"):
        load_restart(path, altered)


@pytest.mark.parametrize("field", ["lon", "z", "wet_mask_3d", "dx_2d"])
def test_same_dimensions_but_changed_geometry_fails(tmp_path, field):
    grid, state, contract = _fixture()
    changed = np.array(getattr(grid, field), copy=True)
    changed.flat[-1] += 0.125
    new_grid = replace(grid, **{field: changed})
    new_contract = make_restart_contract(
        new_grid, namedtuple("Params", ["dt"])(60.), dtype="float64", forcing={"heat": np.zeros((8, 8))},
        controls={"calendar": "360_day"},
        code_paths={"solver": Path(__file__).resolve().parents[1] / "src/jax_solver_global.py"},
        execution={"backend": "cpu"})
    path = tmp_path / "checkpoint.npz"
    _save(path, state, contract)
    with pytest.raises(ValueError, match="contract mismatch: grid"):
        load_restart(path, new_contract)


@pytest.mark.parametrize("step", [-1, 1.5, True, np.array([2])])
def test_noninteger_or_negative_steps_are_not_truncated(tmp_path, step):
    _, state, contract = _fixture()
    with pytest.raises(ValueError, match="nonnegative integer"):
        _save(tmp_path / "checkpoint.npz", state, contract, step=step)


def test_old_file_cannot_reset_ice_or_claim_verified_continuation(tmp_path):
    grid, state, contract = _fixture()
    path = tmp_path / "legacy.npz"
    np.savez(path, **{name: getattr(state, name) for name in state._fields if name != "ice"},
             grid_nx=grid.nx, grid_ny=grid.ny, grid_nz=grid.nz, cur_step=12)
    with pytest.raises(ValueError, match="unversioned.*migration required"):
        load_restart(path, contract)


@pytest.mark.parametrize("corruption", ["missing_ice", "nonfinite", "shape", "dtype", "content",
                                       "history", "cumulative", "checksum", "extra"])
def test_corrupted_payload_is_rejected(tmp_path, corruption):
    _, state, contract = _fixture()
    path = tmp_path / "checkpoint.npz"
    _save(path, state, contract)

    def corrupt(payload):
        if corruption == "missing_ice":
            del payload["state__ice"]
        elif corruption == "nonfinite":
            payload["state__T"][0, 0, 0] = np.nan
        elif corruption == "shape":
            payload["state__eta"] = payload["state__eta"][:, :-1]
        elif corruption == "dtype":
            payload["state__T"] = payload["state__T"].astype("float32")
        elif corruption == "content":
            payload["state__T"][0, 0, 0] += 1.
        elif corruption == "history":
            payload["history__days"][1] += 1.
        elif corruption == "cumulative":
            payload["cumulative__salt"] *= 2.
        elif corruption == "checksum":
            payload["metadata_sha256"] = np.asarray("0" * 64)
        else:
            payload["unregistered"] = np.asarray(1.)

    _rewrite(path, corrupt)
    with pytest.raises(ValueError):
        load_restart(path, contract)


@pytest.mark.parametrize("corruption", ["step", "time", "counter", "schema"])
def test_valid_json_checksum_does_not_hide_invalid_progress(tmp_path, corruption):
    _, state, contract = _fixture()
    path = tmp_path / "checkpoint.npz"
    _save(path, state, contract)

    def edit(metadata):
        if corruption == "step":
            metadata["step"] = 12.5
        elif corruption == "time":
            metadata["elapsed_seconds"] += 0.1
        elif corruption == "counter":
            metadata["counters"]["n_3d_snaps"] = -1
        else:
            metadata["schema_version"] += 1

    _rewrite(path, lambda payload: _edit_metadata(payload, edit))
    with pytest.raises(ValueError):
        load_restart(path, contract)


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_atomic_save_keeps_previous_checkpoint_and_removes_temporary(tmp_path, monkeypatch, failure):
    import restart_contract

    _, state, contract = _fixture()
    path = tmp_path / "checkpoint.npz"
    _save(path, state, contract)
    digest = file_sha256(path)

    def fail(*args, **kwargs):
        raise OSError("injected storage failure")

    if failure == "write":
        monkeypatch.setattr(restart_contract.np, "savez_compressed", fail)
    else:
        monkeypatch.setattr(restart_contract.os, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        _save(path, state, contract, step=13)
    assert file_sha256(path) == digest
    assert load_restart(path, contract).step == 12
    assert list(tmp_path.iterdir()) == [path]


def test_truncated_file_is_not_loaded(tmp_path):
    _, state, contract = _fixture()
    path = tmp_path / "checkpoint.npz"
    _save(path, state, contract)
    original = path.read_bytes()
    path.write_bytes(original[:len(original) // 2])
    with pytest.raises(Exception):
        load_restart(path, contract)


@pytest.mark.parametrize("field", ["T", "ice"])
def test_invalid_state_cannot_overwrite_good_checkpoint(tmp_path, field):
    _, state, contract = _fixture()
    path = tmp_path / "checkpoint.npz"
    _save(path, state, contract)
    digest = file_sha256(path)
    bad = np.array(getattr(state, field), copy=True)
    bad.flat[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        _save(path, state._replace(**{field: bad}), contract)
    assert file_sha256(path) == digest
