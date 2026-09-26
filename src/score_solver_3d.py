"""Score a solver 3D snapshot against the same reference T/S."""
from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from benchmark_metrics import regional_error_metrics, regional_masks


def score_solver_3d(npz_path, snap_path, *, steady_days: float = 10.0,
                    depth_file: str | None = None,
                    depth_var: str | None = None) -> dict:
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

    if "wet_mask_z" in z.files:
        ocean3d = np.asarray(z["wet_mask_z"], dtype=bool)
    elif depth_file is not None:
        import netCDF4
        with netCDF4.Dataset(depth_file) as ds:
            depth = np.asarray(ds[depth_var or "D"][:], dtype=float).T
        if depth.shape != ocean2d.shape:
            raise RuntimeError(f"depth shape mismatch: {depth.shape} vs {ocean2d.shape}")
        ocean3d = ocean2d[:, :, None] & (
            np.abs(zlevels)[None, None, :] <= depth[:, :, None])
    else:
        ocean3d = np.broadcast_to(ocean2d[:, :, None], T.shape)

    error = T - reference
    finite = np.isfinite(error[ocean3d])
    if not finite.any():
        raise RuntimeError("no finite ocean cells")
    result = {
        "global_3d": regional_error_metrics(error, ocean3d),
        "layers": {},
        "n_scored_cells": int(ocean3d.sum()),
        "depth_mask_applied": bool(depth_file is not None or "wet_mask_z" in z.files),
        "verdict": "PASS" if finite.all() else "FAIL",
        "source_snapshot": str(Path(snap_path)),
        "source_npz": str(Path(npz_path)),
        "source_depth": str(Path(depth_file)) if depth_file else None,
    }
    for k, level in enumerate(zlevels):
        out = regional_error_metrics(error[:, :, k], ocean3d[:, :, k])
        if out["n"]:
            result["layers"][f"z{int(level):04d}m"] = out
    for name, mask2d in regional_masks(lat, lon, ocean2d).items():
        mask3d = mask2d[:, :, None] & ocean3d
        result[name + "_3d"] = regional_error_metrics(error, mask3d)
    return result


def main() -> None:
    p = ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--snap", required=True)
    p.add_argument("--depth-file", default=None,
                   help="NetCDF with column depth D on (lat,lon)")
    p.add_argument("--depth-var", default="D")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    result = score_solver_3d(args.npz, args.snap, depth_file=args.depth_file,
                             depth_var=args.depth_var)
    out = Path(args.out or Path(args.snap).with_name("solver_3d_benchmark.json"))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
