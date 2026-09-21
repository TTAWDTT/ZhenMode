"""NCEP/NCAR R1 2-m air temperature loader for surface bulk heat flux.

The default bulk-flux target is the ocean-only zonal mean of WOA SST.  That
keeps the A2 climate test non-circular, but it is much smoother than a real
atmospheric state.  Mature OGCMs such as MOM6, NEMO, ROMS, HYCOM and MPAS use
spatially varying atmospheric forcing.  This loader adds an optional real
NCEP 2-m air temperature field while leaving the original target as default.

NCEP R1 provides monthly 2-m air temperature in kelvin.  We average twelve
monthly means for a requested calendar year and bilinearly interpolate them
onto the solver grid.
"""
from __future__ import annotations

import os

import netCDF4
import numpy as np

from wind_reanalysis import _bilinear

# NCEP R1 monthly mean 2-m air temperature on the T62 Gaussian surface grid.
PSL_AIR_BASE = ("https://psl.noaa.gov/thredds/dodsC/Datasets/"
                "ncep.reanalysis.derived/surface_gauss/")
AIR_FILENAME = "air.2m.mon.mean.nc"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "air")
KELVIN_TO_CELSIUS = 273.15


def _annual_mean_from_dataset(ds, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one calendar year from an open NCEP air-temperature dataset."""
    start = (int(year) - 1948) * 12
    ntime = ds.variables["air"].shape[0]
    if start < 0 or start + 12 > ntime:
        raise ValueError(f"year {year} is outside NCEP air-temperature coverage")
    # Select a contiguous 12-month slab before converting to host memory.
    monthly = ds.variables["air"][start:start + 12, :, :]
    annual = np.asarray(monthly, dtype=np.float64).mean(axis=0)
    if hasattr(annual, "filled"):
        annual = annual.filled(np.nan)
    lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
    return annual, lon, lat


def load_annual_mean_air_temp(grid, year: int = 2023,
                              url_prefix: str = PSL_AIR_BASE,
                              cache_dir: str = CACHE_DIR) -> np.ndarray:
    """Return annual-mean NCEP 2-m air temperature on the solver grid, in C.

    Returns an array shaped ``(nx, ny)`` matching ``GlobalOceanGrid``.  The
    local cache stores the annual mean in kelvin on the native NCEP grid; the
    interpolation and unit conversion are deterministic.
    """
    year = int(year)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"air_2m_annual_{year:04d}.npz")
    if os.path.exists(cache_file):
        z = np.load(cache_file)
        air_native, lon, lat = z["air_2m"], z["lon"], z["lat"]
    else:
        ds = netCDF4.Dataset(url_prefix + AIR_FILENAME)
        try:
            air_native, lon, lat = _annual_mean_from_dataset(ds, year)
        finally:
            ds.close()
        np.savez(cache_file, air_2m=air_native, lon=lon, lat=lat)

    if air_native.shape != (len(lat), len(lon)):
        raise ValueError(f"unexpected NCEP air shape {air_native.shape}")
    if not np.all(np.isfinite(air_native)):
        raise ValueError("NCEP annual-mean air temperature contains non-finite values")

    # _bilinear returns (ny, nx); transpose to the solver's (nx, ny) convention.
    air_fine = _bilinear(air_native, grid.lat, grid.lon, lat, lon)
    air_c = np.asarray(air_fine.T, dtype=np.float64) - KELVIN_TO_CELSIUS
    if not np.all(np.isfinite(air_c)):
        raise ValueError("interpolated NCEP air temperature contains non-finite values")
    return air_c
