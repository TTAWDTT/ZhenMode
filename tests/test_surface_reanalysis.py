"""Generic NCEP surface-forcing loader tests."""
import sys
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from surface_reanalysis import load_monthly_surface_field


class Grid:
    lat = np.array([10.0, 20.0])
    lon = np.array([100.0, 110.0])


def test_monthly_surface_field_shape_and_scale(tmp_path):
    path = tmp_path / "test.sfc.mon.mean.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("lat", 3)
        ds.createDimension("lon", 4)
        ds.createVariable("time", "f8", ("time",))[:] = np.arange(12)
        ds.createVariable("lat", "f8", ("lat",))[:] = np.array([5.0, 15.0, 25.0])
        ds.createVariable("lon", "f8", ("lon",))[:] = np.array([95.0, 105.0, 115.0, 125.0])
        var = ds.createVariable("test", "f4", ("time", "lat", "lon"))
        var[:] = np.full((12, 3, 4), 3.0)
    out = load_monthly_surface_field(
        Grid(), filename="test.sfc.mon.mean.nc", variable="test", year=1948,
        scale=2.0, url_prefix=str(tmp_path) + "/", cache_dir=tmp_path / "cache")
    assert out.shape == (12, 2, 2)
    assert np.allclose(out, 6.0)
