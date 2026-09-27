"""Generic monthly NCEP surface forcing loader for Stage-G runs."""
from __future__ import annotations

import os
from pathlib import Path

import netCDF4
import numpy as np

from wind_reanalysis import _bilinear

PSL_SURFACE_BASE = ("https://psl.noaa.gov/thredds/dodsC/Datasets/"
                    "ncep.reanalysis.derived/surface_gauss/")
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "surface")


def _slab_from_dataset(ds, variable, year):
    start = (int(year) - 1948) * 12
    ntime = ds.variables[variable].shape[0]
    if start < 0 or start + 12 > ntime:
        raise ValueError(f"year {year} is outside NCEP {variable} coverage")
    slab = ds.variables[variable][start:start + 12, :, :]
    slab = np.asarray(slab, dtype=np.float64)
    if hasattr(slab, "filled"):
        slab = slab.filled(np.nan)
    return slab, np.asarray(ds.variables["lon"][:]), np.asarray(ds.variables["lat"][:])


def _interp_months(monthly, grid, lat, lon, scale):
    if monthly.shape != (12, len(lat), len(lon)):
        raise ValueError(f"unexpected NCEP monthly field shape {monthly.shape}")
    if not np.all(np.isfinite(monthly)):
        raise ValueError("NCEP monthly surface forcing contains non-finite values")
    return np.stack([
        np.asarray(_bilinear(monthly[m], grid.lat, grid.lon, lat, lon).T,
                   dtype=np.float64) * scale
        for m in range(12)
    ])


def load_monthly_surface_field(grid, filename, variable, year=2023, scale=1.0,
                               url_prefix=PSL_SURFACE_BASE,
                               cache_dir=None):
    """Return 12 monthly surface fields interpolated to the solver grid."""
    year = int(year)
    cache = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem.replace(".", "_")
    cache_file = cache / f"{variable}_{year:04d}.npz"
    if cache_file.exists():
        z = np.load(cache_file)
        native, lon, lat = z["months"], z["lon"], z["lat"]
    else:
        ds = netCDF4.Dataset(url_prefix + filename)
        try:
            native, lon, lat = _slab_from_dataset(ds, variable, year)
        finally:
            ds.close()
        np.savez(cache_file, months=native, lon=lon, lat=lat)
    out = _interp_months(native, grid, lat, lon, scale)
    if not np.all(np.isfinite(out)):
        raise ValueError("interpolated NCEP surface forcing contains non-finite values")
    return out


def load_monthly_downward_longwave(grid, year=2023, **kwargs):
    """Return monthly downward longwave radiation on the solver grid, W/m2."""
    return load_monthly_surface_field(grid, "dlwrf.sfc.mon.mean.nc", "dlwrf",
                                      year=year, **kwargs)


def load_monthly_downward_shortwave(grid, year=2023, **kwargs):
    """Return monthly downward shortwave radiation on the solver grid, W/m2."""
    return load_monthly_surface_field(grid, "dswrf.sfc.mon.mean.nc", "dswrf",
                                      year=year, **kwargs)


def load_monthly_precipitation(grid, year=2023, **kwargs):
    """Return monthly precipitation rate on the solver grid, kg/m2/s."""
    return load_monthly_surface_field(grid, "prate.sfc.mon.mean.nc", "prate",
                                      year=year, **kwargs)
