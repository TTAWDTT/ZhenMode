"""Write a SYNTHETIC 0.1 deg bathymetry so the data-dependent tests can run.

This is NOT ETOPO and must never be used for a science run. It exists so the
grid / bathymetry / remap / smoothing code paths are actually exercised on a
machine that has no ETOPO2022 relief file -- without it those tests skip
themselves (tests/test_resolution.py::requires_bathy).

The output is the ".npz twin" that grid._read_etopo_global already accepts:
z as int16 on the 1800x3600 0.1 deg grid, plus lon/lat. It is written to
<repo>/data/ETOPO_2022_v1_r3600x1800_surface.nc.npz by default, which is
exactly where config._default_bathymetry_file looks, so nothing else needs
to be configured afterwards.

Usage:
    python scripts/make_synthetic_bathymetry.py [--out-dir data]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from config import ETOPO_FILENAME  # noqa: E402

N_LON, N_LAT = 3600, 1800
RES = 0.1


def synthetic_relief():
    """A smooth, plausible-looking global relief field -- not real data.

    Two zonal harmonics for the depth, plus sinusoidal "continents" and land
    caps at the poles, so the coastline is irregular enough that the masking,
    smoothing and conservative-remap paths all do real work.
    """
    lon = np.arange(N_LON, dtype=np.float64) * RES
    lat = -90.0 + (np.arange(N_LAT, dtype=np.float64) + 0.5) * RES
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    depth = (4200.0 + 1800.0 * np.sin(np.radians(lat_2d) * 3.0)
             * np.cos(np.radians(lon_2d) * 2.0))
    continents = (np.cos(np.radians(lon_2d) * 3.0)
                  * np.cos(np.radians(lat_2d) * 2.0)) > 0.62
    land = continents | (np.abs(lat_2d) > 78.0)
    # ETOPO sign convention: z is ELEVATION (positive up), so ocean is NEGATIVE
    # and land sits at or above 0. Writing positive depths here makes
    # _read_etopo_global see an all-land world -- the grid still builds and
    # every dimension assertion still passes, so only an ocean-fraction check
    # catches it (tests/test_resolution.py::test_bathymetry_reads_as_ocean).
    relief = np.where(land, 0.0, -np.clip(depth, 120.0, 6000.0))
    return lon, lat, relief.astype(np.int16)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=os.path.join(repo_root, "data"),
                    help="where to write the .npz twin (default: <repo>/data)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lon, lat, z = synthetic_relief()
    path = os.path.join(args.out_dir, ETOPO_FILENAME + ".npz")
    np.savez(path, z=z, lon=lon, lat=lat)
    print(f"wrote SYNTHETIC bathymetry -> {path}")
    print(f"  {z.shape[1]} x {z.shape[0]} cells at {RES} deg, "
          f"ocean fraction {float((z < 0).mean()):.1%}")
    print("  NOT ETOPO: use it only to un-skip the grid tests, never for a "
          "science run.")


if __name__ == "__main__":
    main()
