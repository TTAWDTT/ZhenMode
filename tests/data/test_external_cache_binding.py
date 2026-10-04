"""Exercise the real worker-to-loader cache contract without remote access."""

from types import SimpleNamespace

import numpy as np
import pytest

from zhenmode.execution.worker import bind_external_inputs
from zhenmode.model.config.definitions import DEFAULT_CONFIG
from zhenmode.model.forcing import air, wind
from zhenmode.model.io import climatology


@pytest.mark.parametrize("kind", ["wind", "air_annual", "air_monthly"])
def test_worker_cache_binding_reaches_actual_reader(tmp_path, monkeypatch, kind):
    selected = tmp_path / "selected"
    selected.mkdir()
    lon = np.array([0.0, 10.0, 20.0])
    lat = np.array([-10.0, 0.0, 10.0])
    field = np.full((3, 3), 280.0)
    np.savez(selected / "monthly_mean_900.npz", u10=field * 0 + 3, v10=field * 0 + 4, lon=lon, lat=lat)
    np.savez(selected / "air_2m_annual_2023.npz", air_2m=field, lon=lon, lat=lat)
    np.savez(selected / "air_2m_monthly_2023.npz", air_2m_months=np.tile(field, (12, 1, 1)), lon=lon, lat=lat)
    # Register restoration before the real worker mutates these module values.
    monkeypatch.setattr(wind, "CACHE_DIR", str(tmp_path / "wrong-wind"))
    monkeypatch.setattr(air, "CACHE_DIR", str(tmp_path / "wrong-air"))
    monkeypatch.setattr(climatology, "WOA_FILES", dict(climatology.WOA_FILES))

    def refuse_remote(*args, **kwargs):
        raise AssertionError("a selected local cache must not open a remote dataset")

    monkeypatch.setattr(wind.netCDF4, "Dataset", refuse_remote)
    data = {role: {"resolved_path": str(selected / filename)} for role, filename in {
        "bathymetry": "bathy.nc", "temperature": "temperature.nc", "salinity": "salinity.nc",
        "wind-01": "monthly_mean_900.npz", "air": "air_2m_annual_2023.npz",
    }.items()}
    configured = bind_external_inputs({"data": data})
    assert configured.bathymetry_file == str(selected / "bathy.nc")
    assert DEFAULT_CONFIG.bathymetry_file != configured.bathymetry_file
    grid = SimpleNamespace(lon=np.array([1.0, 19.0]), lat=np.array([-8.0, 8.0]))
    if kind == "wind":
        u, v = wind.load_monthly_wind(grid, month_idx=900)
        np.testing.assert_allclose(u, 3.0, rtol=0, atol=1e-14)
        np.testing.assert_allclose(v, 4.0, rtol=0, atol=1e-14)
    else:
        loader = air.load_annual_mean_air_temp if kind == "air_annual" else air.load_monthly_mean_air_temp
        actual = loader(grid, year=2023)
        np.testing.assert_allclose(actual, 280.0 - 273.15, rtol=0, atol=1e-13)
    assert not (tmp_path / "wrong-wind").exists()
    assert not (tmp_path / "wrong-air").exists()
