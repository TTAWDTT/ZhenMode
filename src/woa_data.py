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
import os
import numpy as np
try:
    from netCDF4 import Dataset
except ImportError:   # offline nodes: npz twins only (see load_woa_climatology)
    Dataset = None
from scipy.interpolate import RegularGridInterpolator

from config import DEFAULT_CONFIG
from grid import OceanGrid, make_grid


# ── WOA file paths ───────────────────────────────────────────────────
# WSL (/mnt/c) > offline node (/data/tmp/ocean) > Windows
WOA_DIR = (
    "/mnt/c/Users/zhen.luo/ocean_solver/data/woa"
    if os.path.exists("/mnt/c") else
    "/data/tmp/ocean/data/woa"
    if os.path.exists("/data/tmp/ocean") else
    r"C:\Users\zhen.luo\ocean_solver\data\woa"
)

WOA_FILES = {
    'temperature': os.path.join(WOA_DIR, "woa23_decav_t00_01.nc"),
    'salinity':    os.path.join(WOA_DIR, "woa23_decav_s00_01.nc"),
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

    # npz twin support: offline nodes (no netCDF4/HDF) can read a pre-extracted
    # "<file>.npz" (lon, lat, depth, data float32 with NaN). Identical to netCDF.
    if os.path.exists(filepath + ".npz"):
        d = np.load(filepath + ".npz")
        return {
            'lon':   np.asarray(d['lon'], dtype=np.float64),
            'lat':   np.asarray(d['lat'], dtype=np.float64),
            'depth': np.asarray(d['depth'], dtype=np.float64),
            'data':  np.asarray(d['data'], dtype=np.float64),
        }

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

    # Any remaining NaNs (entire column is NaN) -> leave as NaN; these are
    # land columns that will be filled AFTER interpolation by horizontal
    # nearest-neighbour fill against the solver wet mask (see
    # _fill_ocean_horizontal in get_initial_fields). Filling here with the
    # global mean (T~5.5 C) lets land values bleed into adjacent ocean points
    # during bilinear interpolation, producing spurious cold spikes.
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

    # Normalize target longitudes to the WOA convention [-180, 180).
    # The global grid uses 0..360 lon centers; the regional grid used
    # ~150E (already in range). This makes both work without extrapolation.
    grid_lon = np.mod(np.asarray(grid_lon, dtype=np.float64) + 180.0, 360.0) - 180.0

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

    # Coastal artifact cleanup. WOA has NaN over land; bilinear interpolation
    # propagates NaN into any ocean point touching a land WOA cell. We fill
    # those ocean NaNs from valid horizontal neighbours (iterative diffusion
    # fill, ocean-only). This replaces the old approach where land columns
    # were filled with the GLOBAL MEAN (T~5.5 C) before interpolation, which
    # bled cold spikes into the warm pool (T=5.5 next to T=29, a 24-C jump
    # over one cell) and drove the GM closure to spurious warming.
    wm = np.asarray(grid.wet_mask, dtype=bool)
    if wm.any():
        T_init = _fill_ocean_horizontal(T_init, wm)
        S_init = _fill_ocean_horizontal(S_init, wm)
        # Land points: fill NaN with a neutral value so no NaN leaks into the
        # solver (NaN*0 = NaN, not 0). The wet_mask zeros them at use sites,
        # so the exact value is immaterial; use the ocean mean for cleanliness.
        ocean_full = np.broadcast_to(wm[:, :, None], T_init.shape)
        land_nan = ~ocean_full & np.isnan(T_init)
        if land_nan.any():
            gm = np.nanmean(T_init[ocean_full])
            T_init = np.where(land_nan, gm if np.isfinite(gm) else 0.0, T_init)
        land_nan = ~ocean_full & np.isnan(S_init)
        if land_nan.any():
            gm = np.nanmean(S_init[ocean_full])
            S_init = np.where(land_nan, gm if np.isfinite(gm) else 0.0, S_init)

    return T_init, S_init


def _fill_ocean_horizontal(field, wet_mask, max_pass=50):
    """Fill NaN values at ocean points from valid horizontal neighbours.

    Iterative nearest-neighbour diffusion: each pass replaces ocean NaNs that
    have at least one valid wet neighbour with the mean of those neighbours.
    Repeated until no ocean NaN remains or max_pass reached. Land points are
    never read or written (they stay whatever they are, masked later).

    Args:
      field: (nx, ny, nz) array, NaN at ocean points needing fill.
      wet_mask: (nx, ny) bool, True = ocean.
      max_pass: safety cap on iterations.
    Returns:
      (nx, ny, nz) with ocean NaNs filled.
    """
    f = np.array(field, dtype=np.float64)
    nx, ny, nz = f.shape
    ocean = wet_mask[:, :, None]
    nan_oc = ocean & np.isnan(f)
    if not nan_oc.any():
        return f
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for _ in range(max_pass):
        nan_oc = ocean & np.isnan(f)
        if not nan_oc.any():
            break
        nbr_sum = np.zeros_like(f)
        cnt = np.zeros((nx, ny, nz), dtype=np.int16)
        for di, dj in offsets:
            ii = (np.arange(nx) + di) % nx          # lon periodic
            jj = np.clip(np.arange(ny) + dj, 0, ny - 1)
            nbr = f[ii][:, jj, :]
            valid = ocean[ii][:, jj, :] & np.isfinite(nbr)
            nbr_sum += np.where(valid, nbr, 0.0)
            cnt += valid.astype(np.int16)
        fill = nan_oc & (cnt > 0)
        f = np.where(fill, nbr_sum / np.maximum(cnt, 1), f)
    # Any ocean point still NaN after max_pass (isolated) -> field mean
    still_nan = ocean & np.isnan(f)
    if still_nan.any():
        # ocean is (nx,ny,1); broadcast to full nz for indexing
        ocean_full = np.broadcast_to(ocean, f.shape)
        gm = np.nanmean(f[ocean_full])
        if np.isfinite(gm):
            f = np.where(still_nan, gm, f)
    return f


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
