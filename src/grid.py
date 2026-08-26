"""
Ocean Grid — horizontal and vertical grid generation from ETOPO bathymetry.

Reads ETOPO2022 bathymetry, extracts regional subset for the NW Pacific
domain, generates land mask, Coriolis parameter, and vertical z-level grid.

Grid convention (matches spectral_ops.py):
  - 2D arrays: shape (nx, ny), axis 0 = zonal (lon), axis 1 = meridional (lat)
  - d_dx operates on axis 0, d_dy on axis 1
  - Vertical: z negative downward, z=0 at surface
"""
import numpy as np
from netCDF4 import Dataset
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


def _read_etopo_global(filepath, resolution=1.0, lat_max=85.0):
    """Read global ETOPO2022 bathymetry, downsampled to target resolution.

    ETOPO is 0.1° (3600×1800); for a 1° grid we average 10×10 blocks.
    Returns global depth on (nx, ny) with lon=0.05..359.95 (periodic) and
    lat covering ±lat_max. Land = 0 depth.
    """
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
                     min_depth=None):
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
    cos_lat = np.cos(np.radians(lat))            # (ny,)
    dx_2d = np.broadcast_to(
        R_EARTH * np.radians(gc.dlon) * cos_lat, (nx, ny)
    ).copy()                                      # (nx, ny) varies with lat
    dy = R_EARTH * np.radians(gc.dlat)

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
