"""Read ETOPO bathymetry and assemble the model grid."""

from __future__ import annotations

import os

import numpy as np

from zhenmode.model.inputs.sources import BATHYMETRY_ENV_VAR, ETOPO_FILENAME
from zhenmode.model.solver.geometry.grid import _remap_etopo_area, build_global_grid


def _read_etopo_global(filepath, resolution=1.0, lat_max=85.0, remap="legacy"):
    """Read global ETOPO2022 bathymetry, downsampled to target resolution.

    ETOPO is 0.1° (3600×1800).  ``legacy`` averages integer 0.1° blocks and
    is retained for bit-for-bit compatibility with all historical runs.
    ``area`` performs a conservative spherical-area remap and therefore also
    supports arbitrary target spacings such as 0.37°.
    Returns global depth on (nx, ny) with lon centers, periodic, and lat
    covering ±lat_max. Land = 0 depth.
    """
    # npz twin support: offline nodes (no netCDF4/HDF) can read a pre-extracted
    # "<file>.npz" (z int16, lon, lat). Identical values to the netCDF path.
    #
    # The REAL file wins when both are present. The twin is a fallback for
    # nodes without netCDF4, and letting a leftover twin shadow the real relief
    # is silent wrong-data: the two paths return the same shape, so nothing
    # downstream (including every dimension test) can tell which one ran.
    dataset_factory = Dataset
    npz_path = filepath + ".npz"
    have_nc = os.path.exists(filepath)
    have_npz = os.path.exists(npz_path)
    if not have_nc and not have_npz:
        raise FileNotFoundError(
            f"ETOPO bathymetry not found: neither {filepath!r} nor "
            f"{npz_path!r} exists. Set ${BATHYMETRY_ENV_VAR} to the "
            f"ETOPO2022 0.1 deg relief file, or place {ETOPO_FILENAME} "
            f"(or its .npz twin) in <repo>/data/."
        )
    if have_npz and (not have_nc or dataset_factory is None):
        d = np.load(npz_path)
        etopo_lon = np.asarray(d['lon'], dtype=np.float64)
        etopo_lat = np.asarray(d['lat'], dtype=np.float64)
        z_full = np.asarray(d['z'], dtype=np.float64)
    else:
        if dataset_factory is None:
            raise ImportError(
                f"netCDF4 is required to read {filepath!r}; on nodes without "
                f"it, provide a pre-extracted {npz_path!r} twin instead."
            )
        ds = dataset_factory(filepath)
        etopo_lon = np.array(ds.variables['lon'][:])   # 0.0..359.9 (3600,)
        etopo_lat = np.array(ds.variables['lat'][:])   # -89.95..89.95 (1800,)
        # Full field: z[lat, lon] = (1800, 3600)
        z_full = ds.variables['z'][:, :]
        ds.close()

        if hasattr(z_full, 'filled'):
            z_full = z_full.filled(-99999.0)
        z_full = np.asarray(z_full, dtype=np.float64)
    z_full[z_full <= -9999.0] = 0.0

    if remap == "area":
        return _remap_etopo_area(z_full, etopo_lon, etopo_lat,
                                 resolution, lat_max)
    if remap != "legacy":
        raise ValueError(f"unknown remap mode {remap!r}; use 'legacy' or 'area'")

    # Block-average to target resolution
    step = int(round(resolution / 0.1))   # 10 for 1°
    nlon = len(etopo_lon) // step         # 360
    nlat_full = len(etopo_lat) // step    # 180

    # Trim to full blocks
    z_trim = z_full[:nlat_full * step, :nlon * step]   # (1800, 3600)
    # Reshape and mean over blocks: (nlat_full, step, nlon, step) -> (nlat_full, nlon)
    z_coarse = z_trim.reshape(nlat_full, step, nlon, step).mean(axis=(1, 3))

    # Coarse lat/lon centers
    lon_c = step * 0.1 * (0.5 + np.arange(nlon))    # 0.05, 1.05, ... 359.05
    lat_c = -90.0 + step * 0.1 * (0.5 + np.arange(nlat_full))  # -89.95, ...

    # Clip to ±lat_max
    lat_keep = np.abs(lat_c) <= lat_max
    lat_c = lat_c[lat_keep]
    z_coarse = z_coarse[lat_keep, :]    # (ny, nlon)

    # ETOPO z positive up, ocean negative -> depth positive
    depth = np.where(z_coarse < 0, -z_coarse, 0.0)   # (ny, nlon)

    # Transpose to (nx, ny) = (lon, lat) convention
    depth = depth.T    # (nlon, ny)
    return depth, lon_c, lat_c

try:
    from netCDF4 import Dataset
except ImportError:
    Dataset = None


def make_global_grid(grid_config, bathymetry_file, smooth_passes=0,
                     min_depth=None, remap="legacy"):
    """Prepare ETOPO input, then construct the unchanged FD metric and masks."""
    depth, lon, lat = _read_etopo_global(
        bathymetry_file, resolution=grid_config.resolution, lat_max=grid_config.lat_max,
        remap=remap,
    )
    return build_global_grid(
        grid_config, depth, lon, lat, smooth_passes=smooth_passes,
        min_depth=min_depth, remap=remap,
    )
