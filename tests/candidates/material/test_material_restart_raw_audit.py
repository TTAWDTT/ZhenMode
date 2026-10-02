"""Independent raw checkpoint audit rejects corrupted or incomplete payloads."""
import hashlib
import json
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np
import pytest

from research.experiments.material_restart_replay.audit import _checkpoint, _workspace_input_path


def _fixture():
    shape = (2, 3, 2)
    arrays = {"state": {name: np.ones(shape[:2] if name in {"eta", "ice"} else shape)
                        for name in ("u", "v", "T", "S", "eta", "ice")},
              "cumulative": {"source_inputs": np.arange(12., dtype=float).reshape(6, 2)},
              "history": {"minimum_thickness_m": np.array([1., .8, .6, .5])}}
    contract = {"dt_s": 600., "state_shapes": {name: list(values.shape) for name, values in arrays["state"].items()},
                "state_dtype": np.dtype(float).str}
    metadata = {"schema_version": 1, "step": 4, "elapsed_seconds": 2400., "counters": {"accepted_steps": 4},
                "contract": contract, "outputs": {}, "arrays": {}}
    payload = {}
    for group, values in arrays.items():
        metadata["arrays"][group] = {}
        for name, field in values.items():
            payload[f"{group}__{name}"] = field.copy()
            metadata["arrays"][group][name] = {"shape": list(field.shape), "dtype": field.dtype.str,
                                               "sha256": hashlib.sha256(field.tobytes()).hexdigest()}
    return metadata, payload


def _save(path, metadata, payload):
    encoded = json.dumps(metadata, sort_keys=True, allow_nan=False)
    np.savez(path, **payload, metadata_json=np.asarray(encoded),
             metadata_sha256=np.asarray(hashlib.sha256(encoded.encode()).hexdigest()))


def test_raw_checkpoint_preserves_all_fields_budgets_and_history(tmp_path):
    metadata, payload = _fixture()
    path = tmp_path / "checkpoint.npz"
    _save(path, metadata, payload)
    restored, arrays = _checkpoint(path)
    assert restored == metadata
    for group, values in arrays.items():
        for name, field in values.items():
            np.testing.assert_array_equal(field, payload[f"{group}__{name}"])


@pytest.mark.parametrize("corruption", ["state_value", "ledger_value", "extra_payload", "clock", "thickness"])
def test_raw_checkpoint_refuses_corruption(tmp_path, corruption):
    metadata, payload = _fixture()
    if corruption == "state_value":
        payload["state__T"][0, 0, 0] += .1
    elif corruption == "ledger_value":
        payload["cumulative__source_inputs"][0, 0] += 1.
    elif corruption == "extra_payload":
        payload["not_declared"] = np.zeros(2)
    elif corruption == "clock":
        metadata["elapsed_seconds"] += 600.
    else:
        payload["history__minimum_thickness_m"][2] = -.001
        metadata["arrays"]["history"]["minimum_thickness_m"]["sha256"] = hashlib.sha256(payload["history__minimum_thickness_m"].tobytes()).hexdigest()
    path = tmp_path / "bad.npz"
    _save(path, metadata, payload)
    with pytest.raises(ValueError):
        _checkpoint(path)


@pytest.mark.parametrize("root,name,expected", [
    (PureWindowsPath("D:/Github/ocean-solver"), "/mnt/d/Github/ocean-solver/src/config.py", PureWindowsPath("D:/Github/ocean-solver/src/config.py")),
    (PurePosixPath("/mnt/d/Github/ocean-solver"), "D:\\Github\\ocean-solver\\src\\config.py", PurePosixPath("/mnt/d/Github/ocean-solver/src/config.py")),
])
def test_raw_audit_maps_only_the_same_workspace_between_windows_and_wsl(root, name, expected):
    assert _workspace_input_path(name, root) == expected


@pytest.mark.parametrize("name", ["/mnt/d/other/src/config.py", "/mnt/d/Github/ocean-solver/../outside.py"])
def test_raw_audit_refuses_outside_workspace_alias(name):
    with pytest.raises(ValueError):
        _workspace_input_path(name, PureWindowsPath("D:/Github/ocean-solver"))
