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
  lon/lat: (nx,) / (ny,) from the solver grid (GlobalOceanGrid)
  z: (nz,) negative downward (z=0 at surface)
  T/S output: (nx, ny, nz) with axis 0=lon, axis 1=lat, axis 2=depth
"""

import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from zhenmode.provenance.sources import source_root


def convert_teos_reference_fields(in_situ_temperature, practical_salinity, sea_pressure_dbar):
    """Explicit native-field conversion: in-situ T/SP -> CT/SR≈SA.

    Pressure is supplied in dbar on the same grid, not guessed from layer
    indices. Call after the declared native mapping. No file reads, clipping,
    missing-value repair or geographical Absolute Salinity claim occurs here.
    The ordinary WOA initialization path does not call this automatically.
    """
    from zhenmode.model.solver.physics.teos10 import (
        conservative_from_in_situ,
        reference_salinity,
        validate_state,
    )
    values = (in_situ_temperature,practical_salinity,sea_pressure_dbar)
    if any(np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any() for value in values):
        raise ValueError('native thermodynamic fields contain masked values')
    if any(np.asarray(value).dtype.kind not in 'fiu' for value in values):
        raise ValueError('native thermodynamic fields must be real numeric arrays')
    temperature,salinity,pressure = np.broadcast_arrays(*(np.asarray(value) for value in values))
    sr = reference_salinity(salinity)
    validate_state(sr,temperature,pressure)
    ct = conservative_from_in_situ(sr,temperature,pressure)
    validate_state(sr,ct,pressure)
    return np.asarray(ct),np.asarray(sr)

try:
    from netCDF4 import Dataset
except ImportError:   # offline nodes: npz twins only (see load_woa_climatology)
    Dataset = None

# ── WOA file paths ───────────────────────────────────────────────────
# Resolution order: $OCEAN_SOLVER_WOA_DIR, then <repo>/data/woa. Never
# hard-code a machine-specific absolute path here.
_REPO_ROOT = str(source_root(__file__).parent)
WOA_DIR_ENV_VAR = "OCEAN_SOLVER_WOA_DIR"
WOA_DIR = (os.environ.get(WOA_DIR_ENV_VAR)
           or os.path.join(_REPO_ROOT, "data", "woa"))

WOA_FILES = {
    'temperature': os.path.join(WOA_DIR, "woa23_decav_t00_01.nc"),
    'salinity':    os.path.join(WOA_DIR, "woa23_decav_s00_01.nc"),
}


def selected_woa_path(filepath, *, exact=False):
    """Default readers retain twin precedence; frozen runs request exact files."""
    filepath = os.fspath(filepath)
    if exact or filepath.endswith(".npz"):
        return filepath
    return filepath + ".npz" if os.path.isfile(filepath + ".npz") else filepath


def load_woa_climatology(var_name, filepath=None, *, exact=False):
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
    filepath = selected_woa_path(filepath, exact=exact)

    # npz twin support: offline nodes (no netCDF4/HDF) can read a pre-extracted
    # "<file>.npz" (lon, lat, depth, data with NaN). A twin needs its own identity.
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"WOA2023 file not found: neither {filepath!r} nor its .npz twin "
            f"exists. Set ${WOA_DIR_ENV_VAR} to the directory holding "
            f"woa23_decav_t00_01.nc / woa23_decav_s00_01.nc, or place them in "
            f"<repo>/data/woa/, or pass --init-from with a precomputed npz."
        )
    if filepath.endswith(".npz"):
        with np.load(filepath, allow_pickle=False) as d:
            return {
                'lon':   np.asarray(d['lon'], dtype=np.float64),
                'lat':   np.asarray(d['lat'], dtype=np.float64),
                'depth': np.asarray(d['depth'], dtype=np.float64),
                'data':  np.asarray(d['data'], dtype=np.float64),
            }

    if Dataset is None:
        raise ImportError(f"netCDF4 is required to read the selected WOA file: {filepath}")

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


def _fill_nan_horizontal_per_level(data, max_pass=200):
    """Fill NaNs at each depth level horizontally from same-level neighbours.

    WOA marks a cell NaN where the seafloor is shallower than the level
    (below-terrain) or data is missing. The old `_fill_nan_vertical`
    forward-filled such NaNs downward, so a column whose WOA seafloor sits
    at ~300 m carried its ~28 C surface water to all 4000-m levels; after
    interpolation this gave constant-T columns and ~4 kg/m3 horizontal
    density jumps at depth (12,965 columns in the production init npz),
    which drive 0.9 m/s bottom-layer shear and tracer blow-up within a few
    steps. Standard practice (MOM, NEMO) is to never carry surface water
    below the seafloor: fill each level only from SAME-LEVEL valid ocean
    neighbours, so missing deep cells receive neighbouring deep water.

    Longitudes are periodic; latitude edges clamp. Remaining NaNs after
    max_pass (isolated cells with no same-level ocean data nearby at all)
    fall back to vertical carry as a last resort.

    Args:
        data: (ndepth, nlat, nlon) array with NaN for missing values.

    Returns:
        (ndepth, nlat, nlon) array with NaNs filled.
    """
    data = data.copy()
    ndepth, nlat, nlon = data.shape

    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for _ in range(max_pass):
        nan_mask = np.isnan(data)
        if not nan_mask.any():
            return data
        nbr_sum = np.zeros_like(data)
        cnt = np.zeros(data.shape, dtype=np.int16)
        for di, dj in offsets:
            ii = (np.arange(nlon) + di) % nlon          # lon periodic
            jj = np.clip(np.arange(nlat) + dj, 0, nlat - 1)
            nbr = data[:, jj, :][:, :, ii]
            valid = np.isfinite(nbr)
            nbr_sum += np.where(valid, nbr, 0.0)
            cnt += valid.astype(np.int16)
        fill = nan_mask & (cnt > 0)
        data = np.where(fill, nbr_sum / np.maximum(cnt, 1), data)

    # Isolated NaNs left after max_pass (no same-level ocean data anywhere
    # near, e.g. a tiny isolated sea in an otherwise-NaN level): vertical
    # carry as a last resort.
    data_flat = data.reshape(ndepth, -1)
    for k in range(1, ndepth):
        m = np.isnan(data_flat[k])
        if m.any():
            data_flat[k, m] = data_flat[k - 1, m]
    for k in range(ndepth - 2, -1, -1):
        m = np.isnan(data_flat[k])
        if m.any():
            data_flat[k, m] = data_flat[k + 1, m]
    return data_flat.reshape(ndepth, nlat, nlon)


def interpolate_to_grid(woa, grid_lon, grid_lat, grid_z):
    """Interpolate WOA field onto the solver grid.

    Horizontal: linear interpolation in lon/lat (periodic in lon).
    Vertical: linear interpolation in depth.

    NaN values in the WOA data (missing ocean points, i.e. below-terrain
    levels) are filled horizontally per level before interpolation to
    prevent NaN propagation (see _fill_nan_horizontal_per_level).

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
    # The grid uses 0..360 lon centers, so a searchsorted against the WOA
    # axis without this shift lands at the far end and extrapolates.
    grid_lon = np.mod(np.asarray(grid_lon, dtype=np.float64) + 180.0, 360.0) - 180.0

    # Fill NaN values horizontally per level before interpolation (never
    # carry surface water below the seafloor — see the function docstring
    # for the constant-T-column blow-up this previously caused).
    woa_data = _fill_nan_horizontal_per_level(woa_data)

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


def get_initial_fields(grid, *, files=None):
    """Get initial T and S fields from WOA climatology for the solver grid.

    Args:
        grid: GlobalOceanGrid; needs lon, lat, z and wet_mask.

    Returns:
        T_init: (nx, ny, nz) temperature [degC]
        S_init: (nx, ny, nz) salinity [PSU]
    """
    woa_temp = load_woa_climatology('temperature', None if files is None else files['temperature'], exact=files is not None)
    woa_salt = load_woa_climatology('salinity', None if files is None else files['salinity'], exact=files is not None)

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


def smooth_native_paired_holes(fields, wet_mask, lon, lat):
    """Dirichlet infill of paired holes on one declared native wet level.

    Fields have shape (nfield, nx, ny); original finite paired values remain
    exact. A spherical four-neighbour graph closes latitude and wraps longitude.
    Positive face conductances are length/distance on a uniform angular grid.
    Unanchored hole components remain NaN and are listed, never given a global
    mean. Nearest original anchors/distances describe support proximity, not
    the sole donor of the harmonic value (all boundary anchors influence it).
    This is the explicit new preparation policy, separate from the old reader.
    """
    import heapq
    from collections import deque

    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    raw = np.ma.asarray(fields)
    if raw.dtype.kind not in 'fiu':
        raise ValueError('native infill fields must be real numeric arrays')
    for coordinate in (wet_mask,lon,lat):
        if np.ma.isMaskedArray(coordinate) and np.ma.getmaskarray(coordinate).any():
            raise ValueError('native infill coordinates or wet mask are masked')
    if any(np.asarray(axis).ndim != 1 or np.asarray(axis).dtype.kind not in 'fiu' for axis in (lon,lat)):
        raise ValueError('native infill axes must be real one-dimensional coordinates')
    values = np.asarray(np.ma.asarray(raw, dtype=float).filled(np.nan))
    wet = np.asarray(wet_mask)
    x, y = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    if (values.ndim != 3 or not values.shape[0] or values.shape[1:] != (len(x), len(y))
            or wet.shape != values.shape[1:] or not np.isin(wet, [0, 1]).all()
            or len(x) < 2 or len(y) < 2 or not np.isfinite(x).all()
            or not np.isfinite(y).all() or np.any(np.abs(y) >= 90)
            or not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0)
            or not np.allclose(np.diff(x), 360./len(x), rtol=0, atol=1e-10)
            or not np.allclose(np.diff(y), np.diff(y)[0], rtol=0, atol=1e-10)
            or np.isinf(values).any()):
        raise ValueError('native infill requires paired numeric fields and a uniform periodic angular grid')
    wet = wet.astype(bool)
    paired = np.isfinite(values).all(axis=0) & wet
    holes = wet & ~paired
    result = np.where(paired[None], values, np.nan)
    supported = np.zeros(wet.shape, dtype=bool)
    seen = np.zeros(wet.shape, dtype=bool)
    unsupported = []
    nx, ny = wet.shape
    dlon, dlat = np.deg2rad(360./nx), np.deg2rad(y[1]-y[0])
    cos_lat = np.cos(np.deg2rad(y))

    def neighbours(i, j):
        for ii,jj in (((i-1)%nx,j), ((i+1)%nx,j), (i,j-1), (i,j+1)):
            if 0 <= jj < ny and wet[ii,jj]:
                yield ii,jj

    for i,j in zip(*np.where(holes), strict=True):
        if seen[i,j]:
            continue
        component, queue, anchored = [], deque([(i,j)]), False
        seen[i,j] = True
        while queue:
            ii,jj = queue.popleft()
            component.append((ii,jj))
            for ni,nj in neighbours(ii,jj):
                anchored |= bool(paired[ni,nj])
                if holes[ni,nj] and not seen[ni,nj]:
                    seen[ni,nj] = True
                    queue.append((ni,nj))
        if anchored:
            for ii,jj in component:
                supported[ii,jj] = True
        else:
            unsupported.append({'native_flat_indices':[int(ii*ny+jj) for ii,jj in component]})

    positions = np.argwhere(supported)
    index = np.full(wet.shape, -1, dtype=np.int64)
    index[supported] = np.arange(len(positions))
    rows, cols, coefficients = [], [], []
    rhs = np.zeros((len(positions), values.shape[0]))
    nearest = np.full(wet.shape, -1, dtype=np.int64)
    distance = np.full(wet.shape, -1., dtype=float)
    nearest[paired] = np.flatnonzero(paired)
    distance[paired] = 0.
    heap = []

    def edge(i,j,ni,nj):
        if j == nj:
            conductance = dlat/(dlon*cos_lat[j])
            length = 2.*6371000.*np.arcsin(cos_lat[j]*np.sin(dlon/2.))
        else:
            conductance = dlon*np.cos(np.deg2rad(.5*(y[j]+y[nj])))/dlat
            length = 6371000.*dlat
        return conductance, length

    for row,(i,j) in enumerate(positions):
        diagonal = 0.
        for ni,nj in neighbours(i,j):
            weight,length = edge(i,j,ni,nj)
            diagonal += weight
            if paired[ni,nj]:
                rhs[row] += weight*values[:,ni,nj]
                heapq.heappush(heap,(length,int(ni*ny+nj),int(i),int(j)))
            else:
                if index[ni,nj] < 0:
                    raise ValueError('anchored native component has an unresolved neighbour')
                rows.append(row)
                cols.append(int(index[ni,nj]))
                coefficients.append(-weight)
        rows.append(row)
        cols.append(row)
        coefficients.append(diagonal)
    matrix = coo_matrix((coefficients,(rows,cols)),shape=(len(positions),len(positions))).tocsr()
    residual = 0.
    if len(positions):
        solved = np.asarray(spsolve(matrix,rhs)).reshape(rhs.shape)
        residual = float(np.max(np.abs(matrix@solved-rhs))/max(1.,float(np.max(np.abs(rhs)))))
        if not np.isfinite(solved).all() or residual > 1e-10:
            raise ValueError('native harmonic infill failed its linear-system residual')
        result[:,supported] = solved.T
    while heap:
        length,donor,i,j = heapq.heappop(heap)
        if distance[i,j] >= 0.:
            continue
        distance[i,j],nearest[i,j] = length,donor
        for ni,nj in neighbours(i,j):
            if supported[ni,nj] and distance[ni,nj] < 0.:
                heapq.heappush(heap,(length+edge(i,j,ni,nj)[1],donor,ni,nj))
    if np.any(supported & (nearest < 0)):
        raise ValueError('native infill proximity trace is incomplete')
    return {'fields':result, 'original_paired_mask':paired, 'infill_mask':supported,
            'unresolved_mask':holes & ~supported, 'unsupported_components':unsupported,
            'nearest_original_anchor_flat_index':nearest, 'nearest_anchor_path_distance_m':distance,
            'linear_system_relative_residual':residual,
            'infill_matrix':matrix, 'infill_unknown_flat_indices':np.flatnonzero(supported)}
