"""Standardized benchmark metric tests."""
import numpy as np

from benchmark_metrics import (
    cell_area,
    latitude_band_mld_metrics,
    mixed_layer_depth,
    regional_masks,
    sea_ice_metrics,
)


def test_regional_masks_are_reproducible():
    lat = np.array([39.0, 45.0, 58.0])
    lon = np.array([300.0, 320.0, 0.0])
    ocean = np.ones((3, 3), dtype=bool)
    masks = regional_masks(lat, lon, ocean)
    assert masks["north_atlantic_40_60"].sum() == 4
    assert masks["near_wall_55_60"].sum() == 2


def test_sea_ice_metrics_counts_only_ocean_cells():
    sst = np.array([[-2.0, 20.0], [-3.0, 5.0]])
    ocean = np.array([[True, True], [False, True]])
    result = sea_ice_metrics(sst, ocean, freeze_temp=-1.8)
    assert result["n_cells"] == 1
    assert result["fraction"] == 1.0 / 3.0


def test_cell_area_scales_with_cosine_latitude():
    lat = np.array([-45.0, -44.0])
    lon = np.array([0.0, 1.0])
    area = cell_area(lat, lon)
    assert area.shape == (2, 2)
    assert np.all(area[:, 0] < area[:, 1] * 2)  # loose sanity, physical scale
    assert np.all(area > 0)


def test_mixed_layer_depth_uses_density_threshold():
    z = -np.array([0.0, 5.0, 15.0, 50.0, 100.0, 500.0, 1000.0])
    T = np.stack([np.full((2, 2), 20.0),
                  np.full((2, 2), 20.0),
                  np.full((2, 2), 20.0),
                  np.full((2, 2), 20.0),
                  np.full((2, 2), 19.5),
                  np.full((2, 2), 19.2),
                  np.full((2, 2), 5.0)], axis=-1)
    S = np.full(T.shape, 35.0)
    mld = mixed_layer_depth(T, S, z, ref_depth=0.0, density_delta=0.03)
    assert mld.shape == (2, 2)
    assert mld[0, 0] == 100.0
    assert mld[1, 1] == 100.0

def test_score_npz_reports_stable_fields(tmp_path):
    from benchmark_metrics import score_npz

    npz_path = tmp_path / "run.npz"
    days = np.array([0.0, 180.0, 365.0])
    T_top = np.stack([np.full((3, 3), 20.0),
                      np.full((3, 3), 21.0),
                      np.full((3, 3), 22.0)])
    np.savez(
        npz_path,
        days=days,
        T_top=T_top,
        T_init=np.full((3, 3, 1), 20.0),
        wet_mask=np.ones((3, 3), dtype=bool),
        lat=np.array([30.0, 45.0, 55.0]),
        lon=np.array([300.0, 320.0, 0.0]),
        max_u_peak=np.array(1.0),
        max_eta=np.array([0.1, 0.2, 0.3]),
        heat_content_J=np.array([1e12, 1e12, 1e12]),
        salt_content_kg=np.array([1e12, 1e12, 1e12]),
        verdict=np.array("PASS"))
    result = score_npz(npz_path, steady_days=90.0)
    assert result["verdict"] == "PASS"
    assert result["days_end"] == 365.0
    assert result["ice"]["n_cells"] == 0
    assert result["heat_drift_percent"] == 0.0


def test_cell_area_is_physical():
    lat = np.linspace(-65.0, 65.0, 260)
    lon = np.linspace(0.25, 359.75, 720)
    area = cell_area(lat, lon)
    assert area.shape == (720, 260)
    assert 1.2e9 < float(np.min(area)) < 2.0e9


def test_latitude_band_mld_metrics_separates_bands():

    lat = np.array([-45.0, 30.0, 55.0])
    ocean = np.ones((2, 3), dtype=bool)
    model = np.full((2, 3), 20.0)
    reference = np.full((2, 3), 10.0)
    result = latitude_band_mld_metrics(model, reference, lat, ocean)
    assert set(result) == {"-60_-40", "20_40", "40_60"}
    assert result["-60_-40"]["n"] == 2
    assert result["20_40"]["n"] == 2
    assert result["40_60"]["n"] == 2
    assert result["-60_-40"]["raw_bias_m"] == 10.0
