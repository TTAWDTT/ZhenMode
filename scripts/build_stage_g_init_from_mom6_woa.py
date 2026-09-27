#!/usr/bin/env python3
"""Create a Stage-G initial-state NPZ from the shared MOM6 WOA T/S file.

The shared MOM6 slice stores WOA temperature/salinity on the exact 720x260
0.5-degree grid.  The solver launcher expects a precomputed NPZ with T_init
and S_init in (nx, ny, nz) order.  This helper performs that conversion so
Stage-G does not depend on the prior Stage-F output artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--woa", required=True, help="shared MOM6 WOA T/S NetCDF")
    ap.add_argument("--temperature-variable", default="PTEMP")
    ap.add_argument("--salinity-variable", default="SALT")
    ap.add_argument("--out", required=True, help="output init NPZ path")
    args = ap.parse_args()

    with netCDF4.Dataset(args.woa) as ds:
        for name in (args.temperature_variable, args.salinity_variable):
            if name not in ds.variables:
                raise SystemExit(f"missing variable {name!r}")
        temp = np.asarray(ds.variables[args.temperature_variable][0], dtype=np.float64)
        salt = np.asarray(ds.variables[args.salinity_variable][0], dtype=np.float64)
        depth = np.asarray(ds.variables["depth"][:], dtype=np.float64)

    if temp.shape != salt.shape or temp.ndim != 3:
        raise SystemExit(f"WOA T/S shape mismatch: {temp.shape} vs {salt.shape}")
    if depth.shape != (temp.shape[0],):
        raise SystemExit(f"depth shape {depth.shape} does not match vertical levels {temp.shape[0]}")
    if np.any(depth < 0.0):
        raise SystemExit("depth must be positive downward")

    z = -depth

    T_init = np.transpose(temp, (2, 1, 0)).copy()
    S_init = np.transpose(salt, (2, 1, 0)).copy()
    if not np.all(np.isfinite(T_init)) or not np.all(np.isfinite(S_init)):
        raise SystemExit("WOA T/S contain non-finite values")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        T_init=T_init,
        S_init=S_init,
        z=z,
    )
    print(f"saved {out}: shape={T_init.shape}, T={T_init.min():.2f}..{T_init.max():.2f} C, S={S_init.min():.2f}..{S_init.max():.2f}")


if __name__ == "__main__":
    main()
