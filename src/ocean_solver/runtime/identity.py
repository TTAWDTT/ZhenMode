"""Hash current execution sources and compare complete state bytes."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from ocean_solver.io.restart import file_sha256
from ocean_solver.numerics.backend import np
from ocean_solver.provenance.sources import production_source_modules, source_paths, source_root

SOURCE_MODULES = production_source_modules()


def _source_identity(source_directory=None):
    source_dir = (
        Path(source_directory)
        if source_directory is not None
        else Path(__file__).resolve().parents[2]
    )
    source_dir = source_root(source_dir)
    root = source_dir.parent
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unavailable_installed_distribution"
    return {
        "git_head": head,
        "source_sha256": {
            name + ".py": file_sha256(path) for name, path in source_paths(source_dir, SOURCE_MODULES).items()
        },
    }


def _state_identity(state):
    # Incoming invalid states still need a failure report; do not pass them
    # through restart's finite-only contract encoder.
    return {
        name: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "finite": bool(np.isfinite(value).all()),
        }
        for name, field in zip(state._fields, state, strict=True)
        for value in [np.asarray(field)]
    }


def _same_state_bytes(left, right):
    return all(
        np.asarray(a).dtype == np.asarray(b).dtype
        and np.asarray(a).shape == np.asarray(b).shape
        and np.asarray(a).tobytes() == np.asarray(b).tobytes()
        for a, b in zip(left, right, strict=True)
    )
