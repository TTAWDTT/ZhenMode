"""Score a solver 3D snapshot or snapshot window against the same reference T/S."""
from __future__ import annotations

import json
import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from benchmark_metrics import regional_error_metrics, regional_masks


def _select_snapshot_paths(snap_path, snap_dir=None, steady_days=0.0,
                           snap_days=10.0):
    if snap_dir is not None:
        paths = [path for path in sorted(Path(snap_dir).glob("snap_*.npy")) if path.stat().st_size > 0]
        if not paths:
            raise RuntimeError(f"no snap_*.npy files in {snap_dir}")
        if steady_days > 0:
            n = max(1, int(math.ceil(float(steady_days) / float(snap_days))))
        else:
            n = len(paths)
        return paths[-n:]
    if snap_path is None:
        raise RuntimeError("either snap_path or snap_dir is required")
    return [Path(snap_path)]


def score_solver_3d(npz_path, snap_path=None, *, snap_dir=None,
                    snap_days: float = 10.0, steady_days: float = 10.0,
                    depth_file: str | None = None,
                    depth_var: str | None = None) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    paths = _select_snapshot_paths(
        snap_path, snap_dir=snap_dir, steady_days=steady_days,
        snap_days=snap_days)
    temps = []
    for path in paths:
        snap = np.load(path, allow_pickle=True)
        if snap.ndim != 4 or snap.shape[0] != 4:
            raise RuntimeError(f"expected snapshot [T,u,v,S] in {path}")
        temps.append(np.asarray(snap[0], dtype=float))
    T = np.mean(np.stack(temps), axis=0)
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

    depth_ok = bool(depth_file is not None or "wet_mask_z" in z.files)
    error = T - reference
    finite = np.isfinite(error[ocean3d])
    if not finite.any():
        raise RuntimeError("no finite ocean cells")
    result = {
        "global_3d": regional_error_metrics(error, ocean3d),
        "layers": {},
        "n_scored_cells": int(ocean3d.sum()),
        "n_snapshots": len(paths),
        "steady_days": float(steady_days),
        "snap_days": float(snap_days),
        "depth_mask_applied": depth_ok,
        "verdict": "PASS" if finite.all() and depth_ok else "FAIL",
        "source_snapshots": [str(path) for path in paths],
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
    p.add_argument("--snap", default=None,
                   help="single [T,u,v,S] snapshot .npy")
    p.add_argument("--snap-dir", default=None,
                   help="directory of snap_*.npy; uses the final steady window")
    p.add_argument("--snap-days", type=float, default=10.0)
    p.add_argument("--steady-days", type=float, default=10.0)
    p.add_argument("--depth-file", default=None,
                   help="NetCDF with column depth D on (lat,lon)")
    p.add_argument("--depth-var", default="D")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not args.snap and not args.snap_dir:
        p.error("one of --snap or --snap-dir is required")
    result = score_solver_3d(
        args.npz, args.snap, snap_dir=args.snap_dir, snap_days=args.snap_days,
        steady_days=args.steady_days, depth_file=args.depth_file,
        depth_var=args.depth_var)
    default_out = (Path(args.snap_dir) / "solver_3d_benchmark.json"
                   if args.snap_dir else Path(args.snap).with_name("solver_3d_benchmark.json"))
    out = Path(args.out or default_out)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

