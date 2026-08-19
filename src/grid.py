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

from config import GridConfig, R_EARTH, OMEGA


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


# ── CLI / smoke test ────────────────────────────────────────────────

if __name__ == "__main__":
    from config import DEFAULT_CONFIG

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
