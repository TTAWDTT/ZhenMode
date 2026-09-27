"""NCEP R1 2-m specific-humidity forcing loader."""
from __future__ import annotations

import netCDF4
import os
import numpy as np


from wind_reanalysis import _bilinear

PSL_HUMIDITY_BASE = ("https://psl.noaa.gov/thredds/dodsC/Datasets/"
                     "ncep.reanalysis.derived/surface_gauss/")
HUMIDITY_FILENAME = "shum.2m.mon.mean.nc"
SHUM_FILENAME = HUMIDITY_FILENAME
CACHE_DIR = ("data", "humidity")


def _cache_dir(cache_dir):
    import os
    return cache_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "humidity")


def _slab_from_dataset(ds, year):
    start = (int(year) - 1948) * 12
    ntime = ds.variables["shum"].shape[0]
    if start < 0 or start + 12 > ntime:
        raise ValueError(f"year {year} is outside NCEP humidity coverage")
    slab = ds.variables["shum"][start:start + 12, :, :]
    slab = np.asarray(slab, dtype=np.float64)
    if hasattr(slab, "filled"):
        slab = slab.filled(np.nan)
    return slab, np.asarray(ds.variables["lon"][:]), np.asarray(ds.variables["lat"][:])


def _interp_months(monthly, grid, lat, lon):
    if monthly.shape != (12, len(lat), len(lon)):
        raise ValueError(f"unexpected NCEP monthly humidity shape {monthly.shape}")
    if not np.all(np.isfinite(monthly)):
        raise ValueError("NCEP monthly specific humidity contains non-finite values")
    # NCEP R1 surface specific humidity is g/kg.
    return np.stack([
        np.asarray(_bilinear(monthly[m], grid.lat, grid.lon, lat, lon).T,
                   dtype=np.float64) / 1000.0
        for m in range(12)
    ])


def load_monthly_mean_specific_humidity(grid, year=2023,
                                        url_prefix=PSL_HUMIDITY_BASE,
                                        cache_dir=None):
    """Return 12 monthly 2-m specific-humidity fields on the solver grid, kg/kg."""
    year = int(year)
    cache = _cache_dir(cache_dir)
    os.makedirs(cache, exist_ok=True)
    cache_file = os.path.join(cache, f"shum_2m_monthly_{year:04d}.npz")
    if os.path.exists(cache_file):
        z = np.load(cache_file)
        monthly_native, lon, lat = z["shum_2m_months"], z["lon"], z["lat"]
    else:
        ds = netCDF4.Dataset(url_prefix + SHUM_FILENAME)
        try:
            monthly_native, lon, lat = _slab_from_dataset(ds, year)
        finally:
            ds.close()
        np.savez(cache_file, shum_2m_months=monthly_native, lon=lon, lat=lat)
    humidity = _interp_months(monthly_native, grid, lat, lon)
    if not np.all(np.isfinite(humidity)):
        raise ValueError("interpolated NCEP monthly specific humidity contains non-finite values")
    return humidity


def load_annual_mean_specific_humidity(grid, year=2023,
                                       url_prefix=PSL_HUMIDITY_BASE,
                                       cache_dir=None):
    """Return annual-mean 2-m specific humidity on the solver grid, kg/kg."""
    year = int(year)
    cache = _cache_dir(cache_dir)
    os.makedirs(cache, exist_ok=True)
    cache_file = os.path.join(cache, f"shum_2m_annual_{year:04d}.npz")
    if os.path.exists(cache_file):
        z = np.load(cache_file)
        native, lon, lat = z["shum_2m"], z["lon"], z["lat"]
    else:
        ds = netCDF4.Dataset(url_prefix + SHUM_FILENAME)
        try:
            monthly_native, lon, lat = _slab_from_dataset(ds, year)
        finally:
            ds.close()
        np.savez(cache_file, shum_2m=native.mean(axis=0), lon=lon, lat=lat)
    native = np.asarray(z["shum_2m"], dtype=np.float64)
    humidity = _bilinear(native, grid.lat, grid.lon, lat, lon).T / 1000.0
    if not np.all(np.isfinite(humidity)):
        raise ValueError("interpolated NCEP specific humidity contains non-finite values")
    return humidity
