"""
Ocean Grid — horizontal and vertical grid generation from ETOPO bathymetry.

Reads ETOPO2022 bathymetry, extracts regional subset for the NW Pacific
domain, generates land mask, Coriolis parameter, and vertical z-level grid.

Grid convention (matches spectral_ops.py):
  - 2D arrays: shape (nx, ny), axis 0 = zonal (lon), axis 1 = meridional (lat)
  - d_dx operates on axis 0, d_dy on axis 1
  - Vertical: z negative downward, z=0 at surface
"""
import os

import numpy as np
try:
    from netCDF4 import Dataset
except ImportError:   # offline nodes: npz twin only (see _read_etopo_global)
    Dataset = None
from dataclasses import dataclass

from config import GridConfig, GlobalGridConfig, R_EARTH, OMEGA


# ── Data structures ─────────────────────────────────────────────────

@dataclass
class OceanGrid:
    """Complete ocean grid — horizontal + vertical + bathymetry."""
    # Horizontal coordinates (1D)
    lon: np.ndarray       # (nx,) longitude [degrees E]
    lat: np.ndarray       # (ny,) latitude [degrees N]
    x: np.ndarray         # (nx,) zonal distance from center [m]
    y: np.ndarray         # (ny,) meridional distance from center [m]

    # Grid spacing
    dx: float             # [m]
    dy: float             # [m]

    # Coriolis
    f: np.ndarray         # (nx, ny) Coriolis parameter [1/s]
    f0: float             # Coriolis at domain center [1/s]
    beta: float           # beta = df/dy at center [1/(m*s)]

    # Vertical grid
    z: np.ndarray         # (nz,) level depths [m, negative downward]
    dz: np.ndarray        # (nz-1,) layer thicknesses [m, positive]
    nz: int

    # Bathymetry
    depth: np.ndarray     # (nx, ny) ocean depth [m, positive; 0 on land]
    ocean_mask: np.ndarray  # (nx, ny) bool, True = ocean
    land_mask: np.ndarray   # (nx, ny) bool, True = land

    # Dimensions
    nx: int
    ny: int


# ── ETOPO reading ───────────────────────────────────────────────────

def _read_etopo_subset(filepath, lon_bounds, lat_bounds, nx=128, ny=128,
                       resolution=0.1):
    """
    Read ETOPO2022 bathymetry subset for given lon/lat bounds.

    ETOPO grid:
      lon = 0.0, 0.1, ..., 359.9   (3600 pts, exact integer-tenths)
      lat = -89.95, -89.85, ..., 89.95  (1800 pts, offset by 0.05 deg)

    Our target grid:
      lon[j] = lon0 + j * resolution   (exact ETOPO lon points)
      lat[j] = lat0 + j * resolution   (halfway between ETOPO lat points)

    So lon matches exactly; lat requires averaging adjacent ETOPO rows.

    Args:
        filepath: Path to ETOPO NetCDF file.
        lon_bounds: (lon_min, lon_max) in degrees E.
        lat_bounds: (lat_min, lat_max) in degrees N.
        nx, ny: Target grid dimensions.
        resolution: Grid resolution in degrees (default 0.1).

    Returns:
        depth: (nx, ny) array, positive = ocean depth [m], 0 = land
        lon: (nx,) array [degrees E]
        lat: (ny,) array [degrees N]
    """
    ds = Dataset(filepath)

    # Read coordinate variables
    etopo_lon = ds.variables['lon'][:]
    etopo_lat = ds.variables['lat'][:]

    # ── Longitude indices (exact match) ──
    lon0 = lon_bounds[0]
    i_lon0 = int(round(lon0 / resolution))
    lon_indices = i_lon0 + np.arange(nx)

    # Verify alignment
    lon_err = abs(etopo_lon[lon_indices[0]] - lon0)
    assert lon_err < 1e-6, \
        f"Lon misalignment: ETOPO={etopo_lon[lon_indices[0]]:.4f} vs target={lon0:.4f}"

    # ── Latitude indices (need nlat+1 rows for interpolation) ──
    # ETOPO lat[i] = -89.95 + i * 0.1
    # Target lat[j] = lat0 + j * 0.1, which is at ETOPO index (lat0 + 89.95)/0.1 = x.5
    lat0 = lat_bounds[0]
    i_lat0 = int(np.floor((lat0 - etopo_lat[0]) / resolution))
    lat_indices = i_lat0 + np.arange(ny + 1)

    # Verify we're within ETOPO bounds
    assert lat_indices[-1] < len(etopo_lat), \
        f"Lat index {lat_indices[-1]} exceeds ETOPO lat array ({len(etopo_lat)})"

    # ── Read bathymetry: z[lat, lon] ──
    z_raw = ds.variables['z'][lat_indices, lon_indices]  # (ny+1, nx)
    ds.close()

    # Handle masked arrays and fill values
    if hasattr(z_raw, 'filled'):
        z_raw = z_raw.filled(-99999.0)
    z_raw = np.asarray(z_raw, dtype=np.float64)
    z_raw[z_raw <= -9999.0] = 0.0  # treat fill/missing as land (z=0)

    # Average adjacent lat rows: shifts from xx.x5 to xx.x0
    z_interp = 0.5 * (z_raw[:-1, :] + z_raw[1:, :])  # (ny, nx)

    # Convert ETOPO convention (z positive up, ocean=negative) to
    # depth convention (positive = ocean depth, 0 = land)
    depth = np.where(z_interp < 0, -z_interp, 0.0)  # (ny, nx)

    # Transpose to (nx, ny) to match spectral_ops axis convention
    depth = depth.T  # (nx, ny)

    # Coordinate arrays
    lon = lon0 + np.arange(nx) * resolution     # (nx,)
    lat = lat0 + np.arange(ny) * resolution     # (ny,)

    return depth, lon, lat


# ── Grid generation ─────────────────────────────────────────────────

def make_grid(grid_config: GridConfig, bathymetry_file: str) -> OceanGrid:
    """
    Generate complete ocean grid from ETOPO bathymetry.

    Args:
        grid_config: GridConfig with nx, ny, lon_center, lat_center, etc.
        bathymetry_file: Path to ETOPO NetCDF file.

    Returns:
        OceanGrid with all coordinates, bathymetry, Coriolis, and masks.
    """
    gc = grid_config

    # Read bathymetry
    depth, lon, lat = _read_etopo_subset(
        bathymetry_file, gc.lon_bounds, gc.lat_bounds,
        nx=gc.nx, ny=gc.ny, resolution=gc.etopo_resolution,
    )

    assert depth.shape == (gc.nx, gc.ny), \
        f"Depth shape {depth.shape} != ({gc.nx}, {gc.ny})"

    # ── Grid spacing ──
    dx = gc.dx
    dy = gc.dy

    # ── Distance from domain center ──
    x = (np.arange(gc.nx) - (gc.nx - 1) / 2.0) * dx    # (nx,) symmetric about 0
    y = (np.arange(gc.ny) - (gc.ny - 1) / 2.0) * dy    # (ny,) symmetric about 0

    # ── Coriolis parameter: f = 2*Omega*sin(lat) ──
    # Exact formula on the 2D grid (includes beta-plane implicitly)
    lon_2d, lat_2d = np.meshgrid(lon, lat, indexing='ij')  # (nx, ny)
    f = 2.0 * OMEGA * np.sin(np.radians(lat_2d))

    # ── Vertical grid ──
    z = np.array(gc.z_levels, dtype=np.float64)   # (nz,) negative downward
    dz = np.abs(np.diff(z))                        # (nz-1,) positive thicknesses

    # ── Masks ──
    ocean_mask = depth > 0.0      # (nx, ny) True = ocean
    land_mask = ~ocean_mask       # (nx, ny) True = land

    return OceanGrid(
        lon=lon, lat=lat, x=x, y=y,
        dx=dx, dy=dy,
        f=f, f0=gc.f0, beta=gc.beta,
        z=z, dz=dz, nz=gc.nz,
        depth=depth, ocean_mask=ocean_mask, land_mask=land_mask,
        nx=gc.nx, ny=gc.ny,
    )


# ── Global grid (finite-difference solver) ───────────────────────────

@dataclass
class GlobalOceanGrid:
    """Global lat-lon grid for the FD solver.

    Convention (same as regional OceanGrid):
      - 2D arrays: shape (nx, ny), axis 0 = zonal (lon), axis 1 = meridional (lat)
      - lon is periodic (axis 0 wraps); lat is bounded (±lat_max, no wrap)
      - Vertical: z negative downward, z=0 at surface
    """
    # Horizontal coordinates (1D)
    lon: np.ndarray       # (nx,) longitude [degrees E], periodic
    lat: np.ndarray       # (ny,) latitude [degrees N], ±lat_max

    # Grid spacing — dx varies with latitude (spherical metric)
    dx_2d: np.ndarray     # (nx, ny) zonal spacing [m] = R*cos(lat)*dlon
    dy: float             # (ny,) meridional spacing [m] (constant)
    cos_lat: np.ndarray   # (ny,) cos(lat) metric factor

    # Coriolis
    f: np.ndarray         # (nx, ny) Coriolis parameter [1/s] = 2*Omega*sin(lat)

    # Vertical grid
    z: np.ndarray         # (nz,) level depths [m, negative downward]
    dz: np.ndarray        # (nz-1,) layer thicknesses [m, positive]
    nz: int

    # Bathymetry
    depth: np.ndarray     # (nx, ny) ocean depth [m, positive; 0 on land]
    wet_mask: np.ndarray  # (nx, ny) float, 1.0 = ocean (wet), 0.0 = land (dry)
    ocean_mask: np.ndarray  # (nx, ny) bool, True = ocean (alias of wet_mask>0)
    land_mask: np.ndarray   # (nx, ny) bool, True = land
    # Vertical wet mask: True where layer k is above the seafloor AND the
    # column is ocean. This is the FIX for the ghost-water-column bug —
    # without it, _compute_hydrostatic_pressure integrates density over
    # layers below the seafloor (WOA-interpolated T that has no physical
    # water), producing huge spurious PGF at steep topography.
    # Convention: layer k (at z[k]) is wet iff |z[k]| <= depth AND ocean.
    wet_mask_3d: np.ndarray  # (nx, ny, nz) float, 1.0 = wet (water present)

    # Dimensions
    nx: int
    ny: int

    # True for make_global_grid products: y is bounded (no periodic seam),
    # land cells are REAL land (wet_mask=0) rather than sponge/fringe nodes,
    # so forcing profiles must be built from ocean-only statistics and must
    # NOT be y-tapered to the domain mean (that is a regional-periodic-seam
    # artifact — tapering T_atm to ~14 C at 59.5 N/S injects +0.3..0.9 K/d
    # of spurious polar warming and flattened the model meridional SST
    # gradient by ~30% vs WOA).
    is_global: bool = True


def _smooth_depth_once(depth):
    """One Laplacian smoothing pass on the ocean depth field.

    Each ocean cell's depth becomes the mean of itself and its 4 neighbours
    (lon-periodic, lat-bounded; land neighbours contribute depth 0). Land
    cells stay 0. This damps steep topographic gradients without moving the
    coastline (a cell that was ocean stays >= 0; a near-zero cell may later
    be re-masked as land by make_global_grid's depth>0 test). Operates on
    (nx, ny) = (lon, lat).
    """
    d = np.array(depth, dtype=np.float64)
    # 4-neighbour mean with lon wrap (axis 0), lat clamp (axis 1).
    left = np.roll(d, 1, axis=0)
    right = np.roll(d, -1, axis=0)
    up = np.empty_like(d); up[:, 1:] = d[:, :-1]; up[:, 0] = d[:, 0]
    down = np.empty_like(d); down[:, :-1] = d[:, 1:]; down[:, -1] = d[:, -1]
    nb_mean = 0.25 * (left + right + up + down)
    # Blend toward neighbour mean only at ocean cells; land stays 0.
    ocean = d > 0.0
    out = np.where(ocean, 0.5 * d + 0.5 * nb_mean, 0.0)
    return np.maximum(out, 0.0)


def global_grid_dims(resolution, lat_max, etopo_nlon=3600, etopo_nlat=1800,
                     etopo_res=0.1, remap="legacy"):
    """Horizontal dimensions the ETOPO reader will produce at this resolution.

    ``legacy`` mirrors the historical INTEGER-floor block division exactly:
    ``n = len(source) // step``, NOT ``round(len(source)/step)``. The two differ
    whenever step does not divide the source length (e.g. 1.3° -> step=13 ->
    3600//13 = 276, but 360/1.3 = 277), and since make_global_grid asserts the
    read matches gc.nx/gc.ny, using round() there would reject valid grids.

    ``area`` is the continuous-resolution mode.  It closes the longitude band
    exactly by using ``nx = round(360/res)`` and ``dlon_eff = 360/nx``; latitude
    likewise uses ``dlat_eff = 180/nlat_full``.  This lets --resolution accept
    0.37°, 0.85°, etc., while preserving a physically consistent periodic
    domain.  The returned nx/ny are therefore exactly the reader's output.
    """
    if remap == "legacy":
        step = int(round(resolution / etopo_res))
        if step < 1:
            raise ValueError(f"resolution {resolution} below the {etopo_res}° "
                             f"source grid")
        nx = etopo_nlon // step
        nlat_full = etopo_nlat // step
        # Lat centers sit at -90 + step*etopo_res*(k + 0.5); keep |lat| <= lat_max.
        lat_c = -90.0 + step * etopo_res * (0.5 + np.arange(nlat_full))
        ny = int(np.sum(np.abs(lat_c) <= lat_max))
        return int(nx), int(ny)
    if remap == "area":
        if resolution <= 0.0:
            raise ValueError(f"resolution must be positive (got {resolution})")
        nx = int(round(360.0 / resolution))
        nlat_full = int(round(180.0 / resolution))
        if nx < 1 or nlat_full < 1:
            raise ValueError(f"resolution {resolution} too coarse")
        dlat_eff = 180.0 / nlat_full
        lat_c = -90.0 + dlat_eff * (0.5 + np.arange(nlat_full))
        ny = int(np.sum(np.abs(lat_c) <= lat_max))
        return int(nx), int(ny)
    raise ValueError(f"unknown remap mode {remap!r}; use 'legacy' or 'area'")


def _overlap_matrix(src_edges, dst_edges, sine_weight=False):
    """Sparse-ish dense overlap matrix mapping source cells to target cells.

    For longitude the cells have equal angular width, so the overlap fraction
    is sufficient.  For latitude on a sphere, area ∝ sin(lat); the optional
    sine weighting uses that analytic primitive to build an exact conservative
    spherical-area remap.
    """
    n_src = len(src_edges) - 1
    n_dst = len(dst_edges) - 1
    W = np.zeros((n_dst, n_src), dtype=np.float64)
    for m in range(n_dst):
        a, b = dst_edges[m], dst_edges[m + 1]
        k0 = max(0, int(np.searchsorted(src_edges, a, side='right') - 1))
        k1 = min(n_src, int(np.searchsorted(src_edges, b, side='left')))
        for k in range(k0, k1):
            lo = max(a, src_edges[k])
            hi = min(b, src_edges[k + 1])
            if hi <= lo:
                continue
            if sine_weight:
                ov = np.sin(np.radians(hi)) - np.sin(np.radians(lo))
                norm = np.sin(np.radians(b)) - np.sin(np.radians(a))
            else:
                ov = hi - lo
                norm = b - a
            if norm > 0.0:
                W[m, k] = ov / norm
    return W


def _remap_etopo_area(z_full_ll, src_lon, src_lat, resolution, lat_max):
    """Conservative spherical-area remap to an arbitrary uniform target grid.

    Longitude uses plain overlap (all longitude cells at the same latitude have
    the same physical area).  Latitude uses sin-weighted overlap, matching the
    area element cos(lat)*dlat*dlon.  This replaces integer block extraction
    without changing the physical meaning of target resolution.
    """
    nlon_src = len(src_lon)
    nlat_src = len(src_lat)
    # The ETOPO reader/npz twin stores lon as the left edge of each 0.1° cell.
    src_lon_edges = np.arange(nlon_src + 1, dtype=np.float64) * 0.1
    src_lat_edges = -90.0 + np.arange(nlat_src + 1, dtype=np.float64) * 0.1

    nx = int(round(360.0 / resolution))
    nlat_full = int(round(180.0 / resolution))
    if nx < 1 or nlat_full < 1:
        raise ValueError(f"resolution {resolution} too coarse")
    dlon_eff = 360.0 / nx
    dlat_eff = 180.0 / nlat_full

    dst_lon_edges = np.arange(nx + 1, dtype=np.float64) * dlon_eff
    nlat_all = int(round(180.0 / resolution))
    dst_lat_edges_full = -90.0 + np.arange(nlat_full + 1, dtype=np.float64) * dlat_eff
    lat_c_all = -90.0 + dlat_eff * (0.5 + np.arange(nlat_full))
    keep = np.abs(lat_c_all) <= lat_max
    j0 = int(np.argmax(keep))
    j1 = len(keep) - int(np.argmax(keep[::-1]))
    dst_lat_edges = dst_lat_edges_full[j0:j1 + 1]
    lat_c = lat_c_all[j0:j1]

    W_lon = _overlap_matrix(src_lon_edges, dst_lon_edges)
    W_lat = _overlap_matrix(src_lat_edges, dst_lat_edges, sine_weight=True)
    # z_full_ll @ W_lon.T gives (nlat_src, nx); W_lat @ that gives (ny, nx).
    mapped = W_lat @ (z_full_ll @ W_lon.T)
    depth = np.where(mapped < 0.0, -mapped, 0.0)
    lon_c = dlon_eff * (0.5 + np.arange(nx))
    return depth.T, lon_c, lat_c


def _read_etopo_global(filepath, resolution=1.0, lat_max=85.0,
                       remap="legacy"):
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
    npz_path = filepath + ".npz"
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        etopo_lon = np.asarray(d['lon'], dtype=np.float64)
        etopo_lat = np.asarray(d['lat'], dtype=np.float64)
        z_full = np.asarray(d['z'], dtype=np.float64)
    else:
        ds = Dataset(filepath)
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


def make_global_grid(grid_config, bathymetry_file, smooth_passes=0,
                     min_depth=None, remap="legacy"):
    """Generate a global lat-lon OceanGrid from ETOPO bathymetry.

    Unlike make_grid (regional plane), this builds a true global grid with:
      - periodic longitude (axis 0 wraps),
      - spherical metric (dx varies with latitude),
      - a real wet_mask (1=ocean, 0=land) for the FD solver's no-flux land BC,
      - full 2D Coriolis f = 2*Omega*sin(lat).
    Polar regions above lat_max are excluded (polar cap); the FD solver
    applies a polar-cap filter on the poleward-most row to handle the
    cos(lat)->0 metric singularity.

    smooth_passes: number of Laplacian smoothing passes applied to the ocean
      depth field (land stays 0). Smooths steep topographic gradients
      (continental slopes, trenches) that at 1° resolution drive an
      under-resolved topographic PGF which destabilizes the FD solver.
      Standard OGCM practice (MOM6 applies bathymetry filtering by default).
      0 = raw ETOPO (no smoothing). Each pass replaces an ocean cell's depth
      with the mean of itself + its valid (ocean or land=0) 4-neighbours,
      lon-periodic, lat-bounded. Land/sea mask is re-derived AFTER smoothing
      so a cell that smooths to ~0 becomes land (prevents flooded coast).
    min_depth: ocean points shallower than this become land (depth=0).
      Default = |z[1]| (top interior level thickness), so every wet column
      has at least 2 wet layers (surface + one interior). Ultra-shallow
      coastal points (<5m at 1°) carry unreliable WOA T and produce extreme
      horizontal gradients that destabilize the FD solver. Standard OGCM
      practice (a "minimum depth" / partial-cell floor). 0 = no floor.
    """
    gc = grid_config
    depth, lon, lat = _read_etopo_global(
        bathymetry_file, resolution=gc.resolution, lat_max=gc.lat_max,
        remap=remap,
    )
    nx, ny = depth.shape
    assert nx == gc.nx, f"lon dim {nx} != config nx {gc.nx}"
    assert ny == gc.ny, f"lat dim {ny} != config ny {gc.ny}"

    # ── Bathymetry smoothing (option A) ──
    for _ in range(int(smooth_passes)):
        depth = _smooth_depth_once(depth)

    # ── Minimum-depth floor (drop ultra-shallow points) ──
    if min_depth is None:
        z = np.array(gc.z_levels, dtype=np.float64)
        min_depth = float(abs(z[1])) if len(z) > 1 else 5.0   # ~5m
    depth = np.where(depth > 0.0, np.where(depth < min_depth, 0.0, depth), 0.0)

    # ── Spherical metric ──
    # Continuous area remap closes 360° with nx cells, so dlon_eff may differ
    # from the requested value by the usual rounding residual. Legacy keeps the
    # requested spacing exactly for bit-for-bit backwards compatibility.
    dlon = 360.0 / nx if remap == "area" else gc.dlon
    dlat = (float(np.median(np.diff(lat))) if ny > 1 and remap == "area"
            else gc.dlat)
    cos_lat = np.cos(np.radians(lat))            # (ny,)
    dx_2d = np.broadcast_to(
        R_EARTH * np.radians(dlon) * cos_lat, (nx, ny)
    ).copy()                                      # (nx, ny) varies with lat
    dy = R_EARTH * np.radians(dlat)

    # ── Coriolis: f = 2*Omega*sin(lat), full 2D field ──
    lon_2d, lat_2d = np.meshgrid(lon, lat, indexing='ij')   # (nx, ny)
    f = 2.0 * OMEGA * np.sin(np.radians(lat_2d))

    # ── Vertical grid ──
    z = np.array(gc.z_levels, dtype=np.float64)
    dz = np.abs(np.diff(z))

    # ── Masks ──
    ocean_mask = depth > 0.0
    land_mask = ~ocean_mask
    wet_mask = ocean_mask.astype(np.float64)     # 1.0 ocean, 0.0 land

    # Vertical wet mask: layer k is wet iff |z[k]| <= depth (above seafloor)
    # and the column is ocean. Layers below the seafloor are dry ("ghost
    # water" excluded from pressure integration). z is negative downward
    # so |z[k]| is the depth of level k.
    abs_z = np.abs(z)                                   # (nz,) depth of each level
    wet_mask_3d = (
        (abs_z[None, None, :] <= depth[:, :, None])    # level above seafloor
        & ocean_mask[:, :, None]                        # column is ocean
    ).astype(np.float64)                                # (nx, ny, nz)

    return GlobalOceanGrid(
        lon=lon, lat=lat,
        dx_2d=dx_2d, dy=float(dy), cos_lat=cos_lat,
        f=f,
        z=z, dz=dz, nz=gc.nz,
        depth=depth, wet_mask=wet_mask,
        ocean_mask=ocean_mask, land_mask=land_mask,
        wet_mask_3d=wet_mask_3d,
        nx=nx, ny=ny,
    )


# ── CLI / smoke test ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from config import DEFAULT_CONFIG, GlobalGridConfig

    if "--global" in sys.argv:
        gc = GlobalGridConfig()
        grid = make_global_grid(gc, DEFAULT_CONFIG.bathymetry_file)
        print("=== Global Ocean Grid (FD solver) ===")
        print(f"Horizontal: {grid.nx} x {grid.ny} (lon x lat)")
        print(f"Vertical:   {grid.nz} levels")
        print(f"lon: [{grid.lon[0]:.2f}, {grid.lon[-1]:.2f}E] (periodic)")
        print(f"lat: [{grid.lat[0]:.2f}, {grid.lat[-1]:.2f}N] (±{gc.lat_max}°)")
        print(f"dy = {grid.dy:.1f} m (constant)")
        print(f"dx_2d: equator = {grid.dx_2d[0, grid.ny//2]:.1f} m, "
              f"edge = {grid.dx_2d[0, 0]:.1f} m")
        print(f"cos_lat: equator = {grid.cos_lat[grid.ny//2]:.4f}, "
              f"edge = {grid.cos_lat[0]:.4f}")
        print(f"f: equator = {grid.f[0, grid.ny//2]:.5e} /s, "
              f"edge = {grid.f[0, 0]:.5e} /s")
        n_total = grid.nx * grid.ny
        n_ocean = grid.ocean_mask.sum()
        print(f"Ocean points: {n_ocean} / {n_total}  ({100*n_ocean/n_total:.1f}%)")
        print(f"Land points:  {n_total - n_ocean} / {n_total}  "
              f"({100*(n_total-n_ocean)/n_total:.1f}%)")
        d = grid.depth[grid.ocean_mask]
        print(f"Depth range:  {d.min():.0f} - {d.max():.0f} m, mean {d.mean():.0f} m")
        print()
        print(f"z levels: {grid.z}")
        print(f"dz:       {grid.dz}")
    else:
        grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)

        print("=== Ocean Grid ===")
        print(f"Horizontal: {grid.nx} x {grid.ny} (lon x lat)")
        print(f"Vertical:   {grid.nz} levels")
        print(f"lon: [{grid.lon[0]:.1f}, {grid.lon[-1]:.1f}E]")
        print(f"lat: [{grid.lat[0]:.1f}, {grid.lat[-1]:.1f}N]")
        print(f"dx = {grid.dx:.1f} m,  dy = {grid.dy:.1f} m")
        print(f"f0 = {grid.f0:.5e} /s,  beta = {grid.beta:.5e} /(m*s)")
        print()
        n_total = grid.nx * grid.ny
        n_ocean = grid.ocean_mask.sum()
        n_land = grid.land_mask.sum()
        print(f"Ocean points: {n_ocean} / {n_total}  ({100*n_ocean/n_total:.1f}%)")
        print(f"Land points:  {n_land} / {n_total}  ({100*n_land/n_total:.1f}%)")
        if n_ocean > 0:
            d = grid.depth[grid.ocean_mask]
            print(f"Depth range:  {d.min():.0f} - {d.max():.0f} m")
            print(f"Mean depth:   {d.mean():.0f} m")
        print()
        print(f"z levels: {grid.z}")
        print(f"dz:       {grid.dz}")
