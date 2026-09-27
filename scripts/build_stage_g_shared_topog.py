#!/usr/bin/env python3
"""Build a 0.1-degree ETOPO twin from the shared 0.5-degree MOM6 topography.

The Stage-G artifact and the MOM6 slice share a 720x260 0.5-degree grid.  The
solver grid reader expects the original 0.1-degree ETOPO relief file.  This
helper expands the already-matched 0.5-degree depth field into that expected
0.1-degree twin so the two models can be launched on the same wet mask.

The output is intended for the Stage-G launcher, which then uses
SMOOTH_PASSES=0 because the 0.5-degree topography is already processed.
"""
from __future__ import annotations

import argparse

import netCDF4
import numpy as np

HALF_DEG = 0.5
BLOCK = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topog", required=True, help="NetCDF with a depth(lat,lon) variable")
    ap.add_argument("--depth-variable", default="depth")
    ap.add_argument("--out", required=True, help="output ETOPO-like .npz twin path")
    ap.add_argument("--min-depth", type=float, default=500.0,
                    help="wet mask floor in metres; used only for diagnostics")
    args = ap.parse_args()

    with netCDF4.Dataset(args.topog) as ds:
        if args.depth_variable not in ds.variables:
            raise SystemExit(f"missing depth variable {args.depth_variable!r}")
        depth = np.asarray(ds.variables[args.depth_variable][:], dtype=np.float64)

    lat_half = np.arange(-64.75, 64.75 + 1e-9, HALF_DEG)
    lon_half = np.arange(0.25, 359.75 + 1e-9, HALF_DEG)
    if depth.shape != (lat_half.size, lon_half.size):
        raise SystemExit(
            f"topog shape {depth.shape} does not match shared grid "
            f"{(lat_half.size, lon_half.size)}")
    if np.nanmin(depth) < 0.0:
        raise SystemExit("depth must be positive downward")

    lat010 = np.arange(-89.95, 89.95 + 1e-9, 0.1)
    lon010 = np.arange(0.05, 359.95 + 1e-9, 0.1)
    z010 = np.zeros((lat010.size, lon010.size), dtype=np.float64)
    block = np.repeat(np.repeat(-depth, BLOCK, axis=0), BLOCK, axis=1)
    if block.shape != (1300, 3600):
        raise SystemExit(f"block shape {block.shape} is not (1300, 3600)")
    z010[250:250 + block.shape[0], :] = block
    np.savez_compressed(
        args.out, z=z010.astype(np.float32), lon=lon010, lat=lat010)
    ocean_fraction = float(np.mean(depth > args.min_depth))
    print(f"saved {args.out}: ocean={ocean_fraction:.1%}")


if __name__ == "__main__":
    main()
