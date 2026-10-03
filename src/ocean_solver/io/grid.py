"""Read bathymetry and pass arrays to the pure geometry builder."""


from ocean_solver.config.definitions import GlobalGridConfig
from ocean_solver.geometry.columns import nodal_control_thickness as nodal_control_thickness
from ocean_solver.geometry.mesh import (
    _overlap_matrix as _overlap_matrix,
)
from ocean_solver.geometry.mesh import (
    _remap_etopo_area as _remap_etopo_area,
)
from ocean_solver.geometry.mesh import (
    _smooth_depth_once as _smooth_depth_once,
)
from ocean_solver.geometry.mesh import (
    build_global_grid,
)
from ocean_solver.geometry.mesh import (
    global_grid_dims as global_grid_dims,
)
from ocean_solver.geometry.mesh import (
    land_distance_from_land_mask as land_distance_from_land_mask,
)
from ocean_solver.geometry.types import GlobalOceanGrid as GlobalOceanGrid
from ocean_solver.io.bathymetry import read_etopo_global

try:
    from netCDF4 import Dataset
except ImportError:
    Dataset = None


def _read_etopo_global(filepath, resolution=1.0, lat_max=85.0, remap="legacy"):
    return read_etopo_global(
        filepath, resolution=resolution, lat_max=lat_max,
        remap=remap, dataset_factory=Dataset,
    )


def make_global_grid(grid_config, bathymetry_file, smooth_passes=0,
                     min_depth=None, remap="legacy"):
    """Prepare ETOPO input, then construct the unchanged FD metric and masks."""
    depth, lon, lat = _read_etopo_global(
        bathymetry_file, resolution=grid_config.resolution, lat_max=grid_config.lat_max,
        remap=remap,
    )
    return build_global_grid(
        grid_config, depth, lon, lat, smooth_passes=smooth_passes,
        min_depth=min_depth, remap=remap,
    )




if __name__ == "__main__":
    from ocean_solver.config.definitions import DEFAULT_CONFIG

    gc = GlobalGridConfig()
    grid = make_global_grid(gc, DEFAULT_CONFIG.bathymetry_file)
    print("=== Global Ocean Grid (FD solver) ===")
    print(f"Horizontal: {grid.nx} x {grid.ny} (lon x lat)")
    print(f"Vertical:   {grid.nz} levels")
    print(f"lon: [{grid.lon[0]:.2f}, {grid.lon[-1]:.2f}E] (periodic)")
    print(f"lat: [{grid.lat[0]:.2f}, {grid.lat[-1]:.2f}N] (\u00b1{gc.lat_max}\u00b0)")
    print(f"dy = {grid.dy:.1f} m (constant)")
    print(f"dx_2d: equator = {grid.dx_2d[0, grid.ny//2]:.1f} m, "
          f"edge = {grid.dx_2d[0, 0]:.1f} m")
    print(f"f: equator = {grid.f[0, grid.ny//2]:.5e} /s, "
          f"edge = {grid.f[0, 0]:.5e} /s")
    n_total = grid.nx * grid.ny
    n_ocean = int(grid.ocean_mask.sum())
    print(f"Ocean points: {n_ocean} / {n_total}  ({100*n_ocean/n_total:.1f}%)")
    d = grid.depth[grid.ocean_mask]
    print(f"Depth range:  {d.min():.0f} - {d.max():.0f} m, mean {d.mean():.0f} m")
    print()
    print(f"z levels: {grid.z}")
    print(f"dz:       {grid.dz}")
