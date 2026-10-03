"""Read bathymetry and pass arrays to the pure geometry builder."""


from ocean_solver.geometry.mesh import build_global_grid
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
