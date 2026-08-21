"""
WOA2023 Climatology Reader and Interpolator

Reads World Ocean Atlas 2023 annual mean temperature/salinity NetCDF
files, interpolates them onto the ocean solver grid (horizontal lon/lat
and vertical z-levels), and returns initial T/S fields for the solver.

WOA grid conventions:
  lon: 360 points, [-179.5, 179.5], 1.00 degree
  lat: 180 points, [-89.5, 89.5], 1.00 degree
  depth: 102 levels, [0, 5500] m, positive downward
  t_an/s_an: (time, depth, lat, lon) = (1, 102, 180, 360)

Solver grid conventions:
  lon/lat: (nx,) / (ny,) from OceanGrid
  z: (nz,) negative downward (z=0 at surface)
  T/S output: (nx, ny, nz) with axis 0=lon, axis 1=lat, axis 2=depth
"""
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

from config import DEFAULT_CONFIG
from grid import OceanGrid, make_grid


# ── WOA file paths ───────────────────────────────────────────────────
WOA_DIR = r"C:\Users\zhen.luo\ocean_solver\data\woa"

WOA_FILES = {
    'temperature': WOA_DIR + r"\woa23_decav_t00_01.nc",
    'salinity':    WOA_DIR + r"\woa23_decav_s00_01.nc",
}


def load_woa_climatology(var_name, filepath=None):
    """Load a WOA2023 annual mean field.

    Args:
        var_name: 'temperature' or 'salinity'.
        filepath: Override path. If None, uses WOA_FILES[var_name].

    Returns:
        dict with keys:
            'lon':   (nlon,) array, degrees E
            'lat':   (nlat,) array, degrees N
            'depth': (ndepth,) array, meters positive downward
            'data':  (ndepth, nlat, nlon) array, T in degC or S in PSU
                    Masked/missing values replaced with NaN.
    """
    if filepath is None:
        filepath = WOA_FILES[var_name]

    ds = Dataset(filepath)

    lon = np.array(ds.variables['lon'][:])        # (360,)
    lat = np.array(ds.variables['lat'][:])        # (180,)
    depth = np.array(ds.variables['depth'][:])    # (102,)

    if var_name == 'temperature':
        var = ds.variables['t_an']
    elif var_name == 'salinity':
        var = ds.variables['s_an']
    else:
        raise ValueError(f"Unknown variable: {var_name}")

    # (1, 102, 180, 360) -> squeeze time -> (102, 180, 360) = (depth, lat, lon)
    # Preserve mask: fill masked values with NaN so they don't corrupt
    # the interpolation (WOA _FillValue = 9.96921e+36).
    raw = var[0][:]
    if hasattr(raw, 'filled'):
        data = raw.filled(np.nan).astype(np.float64)
    else:
        data = np.array(raw, dtype=np.float64)
        data[data > 1e10] = np.nan
    ds.close()

    return {
        'lon': lon,
        'lat': lat,
        'depth': depth,
        'data': data,
    }


def _fill_nan_vertical(data):
    """Replace NaN values by propagating nearest non-NaN vertically.

    WOA data has NaN where ocean is shallower than the depth level
    (e.g., near coastlines or at deep levels over continental shelves).
    This fills NaNs by carrying the last valid value downward (or the
    first valid value upward for NaNs above the first valid level).

    Args:
        data: (ndepth, nlat, nlon) array with NaN for missing values.

    Returns:
        (ndepth, nlat, nlon) array with NaNs filled.
    """
    data = data.copy()
    ndepth, nlat, nlon = data.shape

    # Flatten spatial dims for vectorized processing
    data_flat = data.reshape(ndepth, -1)  # (ndepth, nlat*nlon)

    # Forward fill (top -> bottom): carry last valid value downward
    for k in range(1, ndepth):
        nan_mask = np.isnan(data_flat[k])
        if nan_mask.any():
            data_flat[k, nan_mask] = data_flat[k - 1, nan_mask]

    # Backward fill (bottom -> top): carry first valid value upward
    # (for levels above the shallowest valid data, rare but possible)
    for k in range(ndepth - 2, -1, -1):
        nan_mask = np.isnan(data_flat[k])
        if nan_mask.any():
            data_flat[k, nan_mask] = data_flat[k + 1, nan_mask]

    # Any remaining NaNs (entire column is NaN) -> fill with column mean
    col_nan = np.isnan(data_flat).all(axis=0)
    if col_nan.any():
        global_mean = np.nanmean(data_flat)
        data_flat[:, col_nan] = global_mean

    return data_flat.reshape(ndepth, nlat, nlon)


def interpolate_to_grid(woa, grid_lon, grid_lat, grid_z):
    """Interpolate WOA field onto the solver grid.

    Horizontal: linear interpolation in lon/lat (periodic in lon).
    Vertical: linear interpolation in depth.

    NaN values in the WOA data (missing ocean points) are filled
    vertically before interpolation to prevent NaN propagation.

    Args:
        woa: dict from load_woa_climatology.
        grid_lon: (nx,) target longitudes [degrees E].
        grid_lat: (ny,) target latitudes [degrees N].
        grid_z: (nz,) target depths [m, negative downward].

    Returns:
        field: (nx, ny, nz) interpolated field.
    """
    woa_lon = woa['lon']       # (nlon,)
    woa_lat = woa['lat']       # (nlat,)
    woa_depth = woa['depth']   # (ndepth,)
    woa_data = woa['data']     # (ndepth, nlat, nlon)

    # Fill NaN values vertically before interpolation
    woa_data = _fill_nan_vertical(woa_data)

    # Extend longitude for periodic interpolation
    # WOA lon: [-179.5, ..., 179.5]. Add wrap-around point at 180.5.
    woa_lon_ext = np.concatenate([woa_lon, [woa_lon[-1] + 1.0]])
    woa_data_ext = np.concatenate([woa_data, woa_data[:, :, :1]], axis=2)

    # Convert solver z (negative downward) to depth (positive)
    grid_depth = -grid_z  # e.g., z=-100 -> depth=100

    # Build interpolator: (depth, lat, lon) -> value
    rgi = RegularGridInterpolator(
        (woa_depth, woa_lat, woa_lon_ext),
        woa_data_ext,
        method='linear',
        bounds_error=False,
        fill_value=None,  # extrapolate edges
    )

    # Build target meshgrid: (nx, ny, nz)
    # Output axis order: lon (axis 0), lat (axis 1), depth (axis 2)
    lon_3d, lat_3d, depth_3d = np.meshgrid(
        grid_lon, grid_lat, grid_depth, indexing='ij'
    )  # each (nx, ny, nz)

    points = np.stack([depth_3d.ravel(), lat_3d.ravel(), lon_3d.ravel()], axis=-1)
    field = rgi(points).reshape(grid_lon.shape[0], grid_lat.shape[0], grid_z.shape[0])

    return field


def get_initial_fields(grid):
    """Get initial T and S fields from WOA climatology for the solver grid.

    Args:
        grid: OceanGrid with lon, lat, z attributes.

    Returns:
        T_init: (nx, ny, nz) temperature [degC]
        S_init: (nx, ny, nz) salinity [PSU]
    """
    woa_temp = load_woa_climatology('temperature')
    woa_salt = load_woa_climatology('salinity')

    T_init = interpolate_to_grid(woa_temp, grid.lon, grid.lat, grid.z)
    S_init = interpolate_to_grid(woa_salt, grid.lon, grid.lat, grid.z)

    return T_init, S_init


# ── Smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)

    print("=== WOA2023 Initial Fields ===")
    print(f"Solver grid: {grid.nx} x {grid.ny} x {grid.nz}")
    print(f"  lon: [{grid.lon[0]:.1f}, {grid.lon[-1]:.1f}]E")
    print(f"  lat: [{grid.lat[0]:.1f}, {grid.lat[-1]:.1f}]N")
    print(f"  z:   {grid.z}")
    print()

    T_init, S_init = get_initial_fields(grid)

    print(f"T_init shape: {T_init.shape}")
    print(f"T_init range: [{T_init.min():.2f}, {T_init.max():.2f}] degC")
    print(f"S_init shape: {S_init.shape}")
    print(f"S_init range: [{S_init.min():.2f}, {S_init.max():.2f}] PSU")
    print(f"T_init NaN count: {np.isnan(T_init).sum()}")
    print(f"S_init NaN count: {np.isnan(S_init).sum()}")
    print()

    # Vertical profile at domain center
    ic = grid.nx // 2
    jc = grid.ny // 2
    print(f"Profile at center ({grid.lon[ic]:.1f}E, {grid.lat[jc]:.1f}N):")
    print(f"  {'z(m)':>8s}  {'T(C)':>8s}  {'S(PSU)':>8s}")
    for k in range(grid.nz):
        print(f"  {grid.z[k]:8.0f}  {T_init[ic, jc, k]:8.2f}  {S_init[ic, jc, k]:8.2f}")
