"""NCEP specific-humidity forcing loader tests."""
import sys
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from humidity_reanalysis import load_monthly_mean_specific_humidity


class Grid:
    lat = np.array([10.0, 20.0])
    lon = np.array([100.0, 110.0])


def test_monthly_humidity_units_and_shape(tmp_path):
    path = tmp_path / "shum.2m.mon.mean.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("lat", 3)
        ds.createDimension("lon", 4)
        ds.createVariable("time", "f8", ("time",))[:] = np.arange(12)
        lat = ds.createVariable("lat", "f8", ("lat",))[:] = np.array([5.0, 15.0, 25.0])
        lon = ds.createVariable("lon", "f8", ("lon",))[:] = np.array([95.0, 105.0, 115.0, 125.0])
        var = ds.createVariable("shum", "f4", ("time", "lat", "lon"))
        var.units = "grams/kg"
        var[:] = np.full((12, 3, 4), 10.0)
    out = load_monthly_mean_specific_humidity(
        Grid(), year=1948, url_prefix=str(tmp_path) + "/", cache_dir=tmp_path / "cache")
    assert out.shape == (12, 2, 2)
    assert np.allclose(out, 0.01)
