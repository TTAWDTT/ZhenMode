"""Strict, atomic FD restarts; old state-only files are not verified continuations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from zhenmode.provenance.sources import sha256_file as file_sha256

SCHEMA_VERSION = 1
STATE_FIELDS = ("u", "v", "T", "S", "eta", "ice")




def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _array_descriptor(value):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("restart arrays must be finite real numeric arrays without pickle")
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def fingerprint(value):
    """Fingerprint actual effective values, not file names or requested switches."""
    if is_dataclass(value):
        return {field.name: fingerprint(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "_asdict"):
        return fingerprint(value._asdict())
    if isinstance(value, dict):
        if any(not isinstance(name, str) for name in value):
            raise ValueError("restart mappings require string keys")
        return {name: fingerprint(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("restart contract contains a nonfinite scalar")
        return value
    return _array_descriptor(value)


def make_restart_contract(grid, params, *, dtype, forcing, controls, code_paths, execution):
    """Freeze point-sample semantics, all geometry/parameters, forcing and runtime."""
    dimensions = [int(grid.nx), int(grid.ny), int(grid.nz)]
    compute_dtype = np.dtype(dtype)
    if compute_dtype.name not in {"float32", "float64"}:
        raise ValueError("restart compute dtype must be float32 or float64")
    dt = float(params.dt)
    if not np.isfinite(dt) or dt <= 0.:
        raise ValueError("restart dt must be finite and positive")
    return {"schema_version": SCHEMA_VERSION,
            "state_family": "FD_point_samples_linear_free_surface",
            "state_shapes": {name: dimensions[:2] if name in {"eta", "ice"} else dimensions
                             for name in STATE_FIELDS},
            "state_dtype": compute_dtype.str, "dt_s": dt,
            "grid": fingerprint(grid), "effective_params": fingerprint(params),
            "forcing": fingerprint(forcing), "controls": fingerprint(controls),
            "sources": {name: file_sha256(path) for name, path in code_paths.items()},
            "execution": fingerprint(execution)}


def _integer(value, name):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu" or int(array) < 0:
        raise ValueError(f"restart {name} must be a nonnegative integer scalar")
    return int(array)


def _numeric_mapping(values, name):
    if not isinstance(values, dict) or any(
            not isinstance(key, str) or not key or "__" in key for key in values):
        raise ValueError(f"restart {name} requires nonempty string array names without '__'")
    arrays = {key: np.array(value, copy=True) for key, value in values.items()}
    for array in arrays.values():
        _array_descriptor(array)
    return arrays


def _validate_state(state, contract):
    if set(state) != set(STATE_FIELDS):
        raise ValueError("restart must contain all six state fields including ice")
    arrays = _numeric_mapping(state, "state")
    for name, array in arrays.items():
        if list(array.shape) != contract["state_shapes"][name]:
            raise ValueError(f"restart {name} shape differs from the frozen grid")
        if array.dtype.str != contract["state_dtype"]:
            raise ValueError(f"restart {name} dtype differs from the frozen compute dtype")
    return arrays


def _validate_outputs(outputs):
    if not isinstance(outputs, dict):
        raise ValueError("restart outputs must be a mapping")
    for name, digest in outputs.items():
        if (not isinstance(name, str) or "\\" in name or ":" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or not name or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("restart output paths/hashes must be safe relative paths and SHA256")


@dataclass(frozen=True)
class RestartRecord:
    state: dict
    step: int
    elapsed_seconds: float
    counters: dict
    cumulative: dict
    history: dict
    outputs: dict


@contextmanager
def atomic_archive(path, *, replace=False):
    """Serialize and fsync before publishing; links atomically refuse an existing target.

    Checkpoints explicitly permit replacement. Final results use no-clobber
    publication on the same filesystem. An abrupt stop before publication can
    leave only a uniquely named temporary file, never a partial final archive.
    """
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_restart(path, state, contract, *, step, counters, cumulative, history, outputs=None):
    """Write to a same-directory temporary file, fsync, then replace atomically."""
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported restart contract version")
    step = _integer(step, "step")
    counters = {name: _integer(value, name) for name, value in counters.items()}
    state = state._asdict() if hasattr(state, "_asdict") else state
    arrays = {"state": _validate_state(state, contract),
              "cumulative": _numeric_mapping(cumulative, "cumulative"),
              "history": _numeric_mapping(history, "history")}
    outputs = dict(outputs or {})
    _validate_outputs(outputs)
    elapsed = step * contract["dt_s"]
    if not np.isfinite(elapsed):
        raise ValueError("restart elapsed time is nonfinite")
    metadata = {"schema_version": SCHEMA_VERSION, "contract": contract,
                "step": step, "elapsed_seconds": elapsed, "counters": counters,
                "outputs": outputs, "arrays": {group: {name: _array_descriptor(value)
                                                        for name, value in values.items()}
                                              for group, values in arrays.items()}}
    metadata_json = _json(metadata)
    payload = {f"{group}__{name}": value for group, values in arrays.items()
               for name, value in values.items()}
    payload.update(metadata_json=np.asarray(metadata_json),
                   metadata_sha256=np.asarray(hashlib.sha256(metadata_json.encode()).hexdigest()))
    with atomic_archive(path, replace=True) as stream:
        np.savez_compressed(stream, **payload)


def load_restart(path, expected_contract):
    """Reject incompatible/corrupt files before constructing any device state."""
    with np.load(path, allow_pickle=False) as saved:
        if "metadata_json" not in saved or "metadata_sha256" not in saved:
            raise ValueError("unversioned checkpoint is not a verified restart; migration required")
        encoded = saved["metadata_json"]
        checksum = saved["metadata_sha256"]
        if encoded.shape != () or encoded.dtype.kind != "U" or checksum.shape != ():
            raise ValueError("restart metadata must be scalar JSON and checksum")
        metadata_json = str(encoded)
        if hashlib.sha256(metadata_json.encode()).hexdigest() != str(checksum):
            raise ValueError("restart metadata checksum mismatch")
        metadata = json.loads(metadata_json)
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported restart schema version")
        if _json(metadata["contract"]) != _json(expected_contract):
            mismatches = [key for key in expected_contract
                          if metadata["contract"].get(key) != expected_contract[key]]
            raise ValueError(f"restart contract mismatch: {', '.join(mismatches)}")
        step = _integer(metadata["step"], "step")
        elapsed = step * expected_contract["dt_s"]
        if not np.isfinite(elapsed) or metadata["elapsed_seconds"] != elapsed:
            raise ValueError("restart absolute time differs from step * frozen dt")
        counters = {name: _integer(value, name) for name, value in metadata["counters"].items()}
        _validate_outputs(metadata["outputs"])
        arrays = {}
        required_names = {"metadata_json", "metadata_sha256"}
        if set(metadata["arrays"]) != {"state", "cumulative", "history"}:
            raise ValueError("restart array groups mismatch")
        for group, descriptors in metadata["arrays"].items():
            arrays[group] = {}
            for name, descriptor in descriptors.items():
                key = f"{group}__{name}"
                required_names.add(key)
                if key not in saved:
                    raise ValueError(f"restart missing array {key}")
                value = np.array(saved[key], copy=True)
                if _array_descriptor(value) != descriptor:
                    raise ValueError(f"restart array content/shape/dtype mismatch: {key}")
                arrays[group][name] = value
            _numeric_mapping(arrays[group], group)
        if set(saved.files) != required_names:
            raise ValueError("restart contains unexpected arrays")
    state = _validate_state(arrays["state"], expected_contract)
    return RestartRecord(state, step, elapsed, counters, arrays["cumulative"],
                         arrays["history"], metadata["outputs"])
