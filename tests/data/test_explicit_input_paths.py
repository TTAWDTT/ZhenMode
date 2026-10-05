"""Conflicting twins witness path selection independently of numerical scoring."""

from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset

from zhenmode.model.inputs import bathymetry, initial_conditions
from zhenmode.model.inputs.prepare import _input_files


def test_explicit_bathymetry_npz_is_read_even_with_netcdf_installed(tmp_path):
    path = tmp_path / "relief.nc.npz"
    np.savez(path, lon=np.arange(40) * .1, lat=-90 + (np.arange(20) + .5) * .1,
             z=np.full((20, 40), -1234, dtype=np.int16))
    # An invalid sibling must never be opened when the NPZ was selected.
    path.with_suffix("").write_bytes(b"unreadable sibling")
    depth, _, _ = bathymetry._read_etopo_global(path, resolution=1, lat_max=90)
    np.testing.assert_array_equal(depth, np.full((4, 2), 1234))
    with pytest.raises(FileNotFoundError):
        bathymetry._read_etopo_global(tmp_path / "missing.npz")


def test_exact_woa_and_inventory_ignore_conflicting_twin(tmp_path, monkeypatch):
    path = tmp_path / "temperature.nc"
    with Dataset(path, "w") as ds:
        for name, size in (("time", 1), ("depth", 2), ("lat", 3), ("lon", 4)):
            ds.createDimension(name, size)
        for name, values in (("lon", np.arange(4)), ("lat", np.arange(3)), ("depth", [0, 100])):
            ds.createVariable(name, "f8", (name,))[:] = values
        ds.createVariable("t_an", "f8", ("time", "depth", "lat", "lon"))[:] = 7
    twin = str(path) + ".npz"
    np.savez(twin, lon=np.arange(4), lat=np.arange(3), depth=[0, 100], data=np.full((2, 3, 4), 99))
    np.testing.assert_array_equal(initial_conditions.load_woa_climatology("temperature", path)["data"], 99)
    np.testing.assert_array_equal(initial_conditions.load_woa_climatology("temperature", path, exact=True)["data"], 7)
    np.testing.assert_array_equal(initial_conditions.load_woa_climatology("temperature", twin)["data"], 99)
    monkeypatch.setattr(initial_conditions, "WOA_FILES", {"temperature": str(path)})
    args = SimpleNamespace(init_from=None, seasonal_wind=True, wind_year=2023, no_bulk_flux=True)
    assert _input_files(args)["temperature"].name.endswith(".npz")
    assert _input_files(args, initial_files={"temperature": str(path)})["temperature"] == path


def test_frozen_woa_file_cannot_fall_back_to_a_twin(tmp_path):
    path = tmp_path / "missing.nc"
    np.savez(str(path) + ".npz", lon=[0], lat=[0], depth=[0], data=[[[99]]])
    with pytest.raises(FileNotFoundError):
        initial_conditions.load_woa_climatology("temperature", path, exact=True)
