"""NCEP reanalysis wind and air inputs, caching and grid interpolation."""

from __future__ import annotations

import os

import netCDF4
import numpy as np

from zhenmode.model.inputs.forcing.idealized import FORCING_TAPER_CELLS, taper_2d_y
from zhenmode.provenance.sources import source_root

RHO_AIR = 1.225      # kg/m^3

CD = 1.3e-3          # drag coefficient (Large & Pond order)

PSL_BASE = ("https://psl.noaa.gov/thredds/dodsC/Datasets/"
            "ncep.reanalysis.derived/surface_gauss/")

WIND_CACHE_DIR = os.path.join(str(source_root(__file__).parent),
                         "data", "wind")

def _bilinear(field, fine_lat, fine_lon, coarse_lat, coarse_lon):
    """Simpler readable bilinear interpolator (vectorized).

    field: (nlat_c, nlon_c). Returns (len(fine_lat), len(fine_lon)).
    """
    if coarse_lat[1] < coarse_lat[0]:
        field = field[::-1, :]
        coarse_lat = coarse_lat[::-1]
    clat = np.asarray(coarse_lat, float)
    clon = np.asarray(coarse_lon, float)
    flat = np.clip(np.asarray(fine_lat, float), clat[0], clat[-1])
    flon = np.clip(np.asarray(fine_lon, float), clon[0], clon[-1])

    jl = np.clip(np.searchsorted(clon, flon, "right") - 1, 0, len(clon) - 2)
    il = np.clip(np.searchsorted(clat, flat, "right") - 1, 0, len(clat) - 2)
    fx = (flon - clon[jl]) / (clon[jl + 1] - clon[jl])
    fy = (flat - clat[il]) / (clat[il + 1] - clat[il])

    # np.ix_ for open mesh indexing: il is (nlat_f,), jl is (nlon_f,). field is
    # (nlat_c, nlon_c). field[np.ix_(il, jl)] -> (nlat_f, nlon_f) via outer
    # indexing, NOT broadcasting -- broadcasting silently produces the wrong
    # shape whenever nlat_f != nlon_f, which is the case on every global grid.
    f00 = field[np.ix_(il, jl)]
    f01 = field[np.ix_(il, jl + 1)]
    f10 = field[np.ix_(il + 1, jl)]
    f11 = field[np.ix_(il + 1, jl + 1)]
    return ((1 - fy)[:, None] * ((1 - fx)[None, :] * f00
                                 + fx[None, :] * f01)
            + fy[:, None] * ((1 - fx)[None, :] * f10
                             + fx[None, :] * f11))

def load_monthly_wind(grid, month_idx=-1, url_prefix=PSL_BASE,
                      cache_dir=None):
    """Load monthly-mean 10m u/v over the solver domain.

    Args:
        month_idx: month index into the time axis (default -1 = latest).
        url_prefix: NOAA PSL OPeNDAP base for the *.*.10m.mon.mean.nc files.
        cache_dir: local cache for the raw monthly fields. Cached as .npz so
            repeat runs (and the solver) need no network.
        grid: GlobalOceanGrid defining lon/lat (needed for the interpolation).

    Returns:
        (u10, v10): (nx, ny) arrays of 10m wind components [m/s] on the
        solver grid; orientation S->N, lon ascending.
    """
    month_idx = int(month_idx)   # netCDF time-index must be int (float -> IndexError)
    cache_dir = WIND_CACHE_DIR if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"monthly_mean_{month_idx}.npz")

    # ── Fetch (or load cache) ──
    if os.path.exists(cache_file):
        z = np.load(cache_file)
        u10, v10, lon, lat = (z["u10"], z["v10"], z["lon"], z["lat"])
    else:
        u_ds = netCDF4.Dataset(url_prefix + "uwnd.10m.mon.mean.nc")
        v_ds = netCDF4.Dataset(url_prefix + "vwnd.10m.mon.mean.nc")
        lon = u_ds.variables["lon"][:]
        lat = u_ds.variables["lat"][:]
        u10 = u_ds.variables["uwnd"][month_idx, :, :]   # (nlat, nlon)
        v10 = v_ds.variables["vwnd"][month_idx, :, :]
        u_ds.close()
        v_ds.close()
        u10 = np.asarray(u10, dtype=np.float64)
        v10 = np.asarray(v10, dtype=np.float64)
        np.savez(cache_file, u10=u10, v10=v10, lon=lon, lat=lat)

    # ── Bilinear interpolate coarse -> solver grid ──
    u_fine = _bilinear(u10, grid.lat, grid.lon, lat, lon)  # (ny, nx)
    v_fine = _bilinear(v10, grid.lat, grid.lon, lat, lon)
    # Transpose (ny, nx) -> (nx, ny) to match solver axis convention
    return u_fine.T, v_fine.T

def wind_stress_from_wind(u10, v10):
    """Bulk wind stress from 10m wind: tau = rho_a * Cd * |U| * U.

    Args:
        u10, v10: (nx, ny) 10m wind components [m/s].
    Returns:
        tau_x, tau_y: (nx, ny) wind stress [N/m^2].
    """
    spd = np.sqrt(u10 ** 2 + v10 ** 2)
    tau_x = RHO_AIR * CD * spd * u10
    tau_y = RHO_AIR * CD * spd * v10
    return tau_x, tau_y

def real_wind_forcing(grid, month_idx=-1, taper_cells=None):
    """End-to-end: load real 10m wind and return (tau_x, tau_y) on solver grid.

    Convenience wrapper for make_solver_global(grid, physics, dt,
    forcing=(tau_x, tau_y, Q)); this is the wind path the global runner uses.

    ``taper_cells`` defaults to ``forcing.FORCING_TAPER_CELLS`` (8), which
    ramps the stress to zero over the outermost rows at each lat edge (see
    forcing._taper_y).
    """
    u10, v10 = load_monthly_wind(grid, month_idx=month_idx)
    tau_x, tau_y = wind_stress_from_wind(u10, v10)
    if taper_cells is None:
        taper_cells = FORCING_TAPER_CELLS
    ny = grid.ny
    tau_x = taper_2d_y(tau_x, ny, taper_cells)
    tau_y = taper_2d_y(tau_y, ny, taper_cells)
    return tau_x, tau_y

PSL_AIR_BASE = ("https://psl.noaa.gov/thredds/dodsC/Datasets/"
                "ncep.reanalysis.derived/surface_gauss/")

AIR_FILENAME = "air.2m.mon.mean.nc"

AIR_CACHE_DIR = os.path.join(str(source_root(__file__).parent),
                         "data", "air")

KELVIN_TO_CELSIUS = 273.15

def _monthly_slab_from_dataset(ds, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read twelve calendar months from an open NCEP air-temperature dataset."""
    start = (int(year) - 1948) * 12
    ntime = ds.variables["air"].shape[0]
    if start < 0 or start + 12 > ntime:
        raise ValueError(f"year {year} is outside NCEP air-temperature coverage")
    # Select a contiguous 12-month slab before converting to host memory.
    monthly = ds.variables["air"][start:start + 12, :, :]
    monthly = np.asarray(monthly, dtype=np.float64)
    if hasattr(monthly, "filled"):
        monthly = monthly.filled(np.nan)
    lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
    return monthly, lon, lat

def _annual_mean_from_dataset(ds, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one calendar year from an open NCEP air-temperature dataset."""
    monthly, lon, lat = _monthly_slab_from_dataset(ds, year)
    return monthly.mean(axis=0), lon, lat

def _air_native(year, cache_dir, url_prefix, period):
    """Read one explicitly named native NCEP air cache, or populate it once."""
    cache_dir = AIR_CACHE_DIR if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"air_2m_{period}_{year:04d}.npz")
    key = "air_2m_months" if period == "monthly" else "air_2m"
    if os.path.exists(cache_file):
        with np.load(cache_file) as z:
            return z[key], z["lon"], z["lat"]
    ds = netCDF4.Dataset(url_prefix + AIR_FILENAME)
    try:
        loader = _monthly_slab_from_dataset if period == "monthly" else _annual_mean_from_dataset
        field, lon, lat = loader(ds, year)
    finally:
        ds.close()
    np.savez(cache_file, **{key: field, "lon": lon, "lat": lat})
    return field, lon, lat


def load_monthly_mean_air_temp(grid, year: int = 2023,
                               url_prefix: str = PSL_AIR_BASE,
                               cache_dir: str | None = None) -> np.ndarray:
    """Return twelve monthly NCEP 2m air fields on the solver grid, in C.

    Returns shape ``(12, nx, ny)``.  Month 0 is January.  A single cached
    ``(12, nlat, nlon)`` slab avoids one remote request per month.  The caller
    can use the same 30-day/blend schedule as seasonal wind.
    """
    year = int(year)
    monthly_native, lon, lat = _air_native(year, cache_dir, url_prefix, "monthly")

    if monthly_native.shape != (12, len(lat), len(lon)):
        raise ValueError(f"unexpected NCEP monthly air shape {monthly_native.shape}")
    if not np.all(np.isfinite(monthly_native)):
        raise ValueError("NCEP monthly air temperature contains non-finite values")

    # Transpose each field from _bilinear's (ny, nx) to solver (nx, ny).
    air = np.stack([
        np.asarray(
            _bilinear(monthly_native[m], grid.lat, grid.lon, lat, lon).T,
            dtype=np.float64,
        )
        for m in range(12)
    ]) - KELVIN_TO_CELSIUS
    if not np.all(np.isfinite(air)):
        raise ValueError("interpolated NCEP monthly air contains non-finite values")
    return air

def load_annual_mean_air_temp(grid, year: int = 2023,
                              url_prefix: str = PSL_AIR_BASE,
                              cache_dir: str | None = None) -> np.ndarray:
    """Return annual-mean NCEP 2-m air temperature on the solver grid, in C.

    Returns an array shaped ``(nx, ny)`` matching ``GlobalOceanGrid``.  The
    local cache stores the annual mean in kelvin on the native NCEP grid; the
    interpolation and unit conversion are deterministic.
    """
    year = int(year)
    air_native, lon, lat = _air_native(year, cache_dir, url_prefix, "annual")

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
