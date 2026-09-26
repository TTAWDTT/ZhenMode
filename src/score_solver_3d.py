"""Score a solver 3D snapshot against the same reference T/S."""
from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from benchmark_metrics import regional_error_metrics, regional_masks


def score_solver_3d(npz_path, snap_path, *, steady_days: float = 10.0) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    snap = np.load(snap_path, allow_pickle=True)
    if snap.ndim != 4 or snap.shape[0] != 4:
        raise RuntimeError("expected snapshot [T,u,v,S]")
    T = np.asarray(snap[0], dtype=float)
    reference = np.asarray(z["T_init"], dtype=float)
    ocean2d = np.asarray(z["wet_mask"], dtype=bool)
    lat = np.asarray(z["lat"], dtype=float)
    lon = np.asarray(z["lon"], dtype=float)
    zlevels = np.asarray(z["z"], dtype=float)
    if T.shape != reference.shape:
        raise RuntimeError(f"snapshot shape mismatch: {T.shape} vs {reference.shape}")
    ocean3d = np.broadcast_to(ocean2d[:, :, None], T.shape)
    error = T - reference
    finite = np.isfinite(error[ocean3d])
    if not finite.any():
        raise RuntimeError("no finite ocean cells")
    result = {
        "global_3d": regional_error_metrics(error, ocean3d),
        "layers": {},
        "n_scored_cells": int(ocean3d.sum()),
        "verdict": "PASS" if finite.all() else "FAIL",
        "source_snapshot": str(Path(snap_path)),
        "source_npz": str(Path(npz_path)),
    }
    for k, depth in enumerate(zlevels):
        out = regional_error_metrics(error[:, :, k], ocean2d)
        if out["n"]:
            result["layers"][f"z{int(depth):04d}m"] = out
    for name, mask2d in regional_masks(lat, lon, ocean2d).items():
        mask3d = np.broadcast_to(mask2d[:, :, None], T.shape)
        result[name + "_3d"] = regional_error_metrics(error, mask3d)
    return result


def main() -> None:
    p = ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--snap", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    result = score_solver_3d(args.npz, args.snap)
    out = Path(args.out or Path(args.snap).with_name("solver_3d_benchmark.json"))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
