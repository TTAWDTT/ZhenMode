"""
Reanalysis wind stress loader -- NOAA PSL NCEP/NCAR R1 10m wind (no-auth).

Loads monthly-mean 10m wind from the NOAA Physical Sciences Laboratory
THREDDS OPeNDAP server (auth-free, no .cdsapirc / Earthdata login),
bilinear-interpolates the coarse ~1.9 deg (T62, 192x94 Gaussian) grid onto
the solver's lat-lon grid, and converts to surface wind stress via the bulk
formula:

    tau = rho_air * Cd * |U_10| * (u_10, v_10)

with rho_air = 1.225 kg/m^3 and Cd = 1.3e-3 (Large & Pond order).

Why this source:
  - ERA5 / CDS requires a .cdsapirc account (not available here).
  - MERRA-2 / JRA-55 are Earthdata / JRA auth-gated.
  - NOAA PSL NCEP/NCAR R1 10m wind is public and served as OPeNDAP, so it
    can be read directly with netCDF4 (no xarray / pooch needed).

Source URLs (monthly mean + daily climatology + 6-hourly per-year):
  https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/
      uwnd.10m.mon.mean.nc      (938 months)
      vwnd.10m.mon.mean.nc
      uwnd.10m.day.ltm.nc       (365-day climatology)
      uwnd.10m.gauss.YYYY.nc    (6-hourly for year YYYY)
  Note: the *surface/* (sigma 0.995) fields are NOT 10m; must use
  surface_gauss/*.10m.*.

NetCDF layout (verified 2026-08-22):
  dims: [time, lat, lon], time in hours since 1800-01-01
  lat: N->S (88.542 .. -88.542)  -- FLIPPED on read
  lon: 0 .. 358.125 (1.875 deg)
"""

import os

import netCDF4
import numpy as np

from ocean_solver.forcing.fields import FORCING_TAPER_CELLS, taper_2d_y
from ocean_solver.provenance.locations import source_root

# ── Bulk-formula constants ──────────────────────────────────────────
RHO_AIR = 1.225      # kg/m^3
CD = 1.3e-3          # drag coefficient (Large & Pond order)

# ── Source ──────────────────────────────────────────────────────────
PSL_BASE = ("https://psl.noaa.gov/thredds/dodsC/Datasets/"
            "ncep.reanalysis.derived/surface_gauss/")

# Local cache dir (gitignored via data/). If a month's field is already
# cached, we read it locally — the solver then needs no network.
CACHE_DIR = os.path.join(str(source_root(__file__).parent),
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
                      cache_dir=CACHE_DIR):
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




if __name__ == "__main__":
    from ocean_solver.config.definitions import DEFAULT_CONFIG, GlobalGridConfig
    from ocean_solver.io.grid import make_global_grid

    grid = make_global_grid(GlobalGridConfig(), DEFAULT_CONFIG.bathymetry_file)
    tau_x, tau_y = real_wind_forcing(grid, month_idx=-1)
    spd = np.sqrt(tau_x ** 2 + tau_y ** 2)
    print("=== Real wind stress (NCEP/NCAR R1 10m, latest month) ===")
    print(f"grid: {grid.nx} x {grid.ny}")
    print("tau_x range: %.4f .. %.4f N/m^2" % (tau_x.min(), tau_x.max()))
    print("tau_y range: %.4f .. %.4f N/m^2" % (tau_y.min(), tau_y.max()))
    print("|tau| max:   %.4f N/m^2" % spd.max())
    print("|tau| mean:  %.4f N/m^2" % spd.mean())
