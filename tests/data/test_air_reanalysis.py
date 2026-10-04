import numpy as np
import pytest

from zhenmode.model.forcing.air import load_annual_mean_air_temp, load_monthly_mean_air_temp


class Grid:
    """Minimal grid interface expected by the bilinear interpolator."""
    nx = 5
    ny = 4
    lon = np.linspace(1.0, 19.0, nx)
    lat = np.linspace(-8.0, 8.0, ny)


def test_cached_annual_mean_air_temp_interpolates(tmp_path):
    lon = np.array([0.0, 10.0, 20.0])
    lat = np.array([-10.0, 0.0, 10.0])
    air_native = 280.0 + 0.1 * lat[:, None] + 0.2 * lon[None, :]
    np.savez(tmp_path / "air_2m_annual_2023.npz",
             air_2m=air_native, lon=lon, lat=lat)

    air = load_annual_mean_air_temp(Grid(), year=2023, cache_dir=str(tmp_path))
    assert air.shape == (Grid.nx, Grid.ny)
    assert np.all(np.isfinite(air))
    # Input is already in kelvin; loader converts to Celsius.
    assert air.min() > -100.0
    assert air.max() < 100.0


def test_cached_monthly_mean_air_temp_interpolates(tmp_path):
    lon = np.array([0.0, 10.0, 20.0])
    lat = np.array([-10.0, 0.0, 10.0])
    lon2, lat2 = np.meshgrid(lon, lat)
    monthly = (280.0 + 0.1 * lat2[None, :, :] + 0.2 * lon2[None, :, :]
               + 5.0 * np.cos(2.0 * np.pi * np.arange(12)[:, None, None] / 12.0))
    np.savez(tmp_path / "air_2m_monthly_2023.npz",
             air_2m_months=monthly, lon=lon, lat=lat)

    air = load_monthly_mean_air_temp(Grid(), year=2023, cache_dir=str(tmp_path))
    assert air.shape == (12, Grid.nx, Grid.ny)
    assert np.all(np.isfinite(air))
    # January and July differ in the synthetic seasonal cache.
    assert not np.allclose(air[0], air[6])


def test_monthly_mean_air_temp_rejects_bad_shape(tmp_path):
    lon = np.array([0.0, 10.0])
    lat = np.array([0.0, 10.0])
    np.savez(tmp_path / "air_2m_monthly_2023.npz",
             air_2m_months=np.full((11, 2, 2), 280.0), lon=lon, lat=lat)
    with pytest.raises(ValueError, match="unexpected NCEP monthly air shape"):
        load_monthly_mean_air_temp(Grid(), year=2023, cache_dir=str(tmp_path))


def test_annual_mean_air_temp_rejects_bad_cache(tmp_path):
    lon = np.array([0.0, 10.0])
    lat = np.array([0.0, 10.0])
    np.savez(tmp_path / "air_2m_annual_2023.npz",
             air_2m=np.array([[np.nan, 280.0], [281.0, 282.0]]),
             lon=lon, lat=lat)
    with pytest.raises(ValueError, match="non-finite"):
        load_annual_mean_air_temp(Grid(), year=2023, cache_dir=str(tmp_path))
