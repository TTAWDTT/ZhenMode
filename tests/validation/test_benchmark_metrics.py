"""Standardized benchmark metric tests."""
import numpy as np
import pytest

from zhenmode.evaluation.metrics import (
    cell_area,
    regional_masks,
    sea_ice_metrics,
)
from zhenmode.model.diagnostics.mixed_layer import mixed_layer_depth


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
    from zhenmode.evaluation.metrics import score_npz

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
    assert result["metric_definition"] == "area_weighted_angular_box_v2"
    assert result["reference_role"] == "initialization_field_not_independent_validation"
    assert result["temporal_averaging"] == "arithmetic_saved_records_not_time_bounds_weighted"
    assert result["budget_drift_scope"] == "endpoint_content_change_not_budget_residual"
    assert result["verdict_scope"] == "integration_watchdog_not_climate_accuracy"


def test_cell_area_is_physical():
    lat = np.linspace(-65.0, 65.0, 260)
    lon = np.linspace(0.25, 359.75, 720)
    area = cell_area(lat, lon)
    assert area.shape == (720, 260)
    assert 1.2e9 < float(np.min(area)) < 2.0e9


def test_internal_and_external_budget_drift_share_validation():
    from zhenmode.evaluation.external import _relative_drift as external_drift
    from zhenmode.evaluation.metrics import _relative_drift

    assert external_drift is _relative_drift
    assert np.isnan(_relative_drift(np.array([0., 1.])))
    assert np.isnan(_relative_drift(np.array([1., np.nan, 2.])))
    assert _relative_drift(np.array([100., 110.])) == 10.


def test_snapshot_scores_wet_area_not_number_of_cells():
    from zhenmode.evaluation.metrics import score_snapshot

    lat, lon = np.array([0., 60.]), np.array([0., 180.])
    reference = np.zeros((2, 2))
    model = np.array([[0., 3.], [0., 3.]])
    result = score_snapshot(model, reference, np.ones((2, 2), bool), lat, lon)
    assert result["global"]["raw_bias"] == pytest.approx(1.)
    assert result["global"]["raw_rmse"] == pytest.approx(np.sqrt(3.))
    assert result["metric_definition"] == "area_weighted_angular_box_v2"


def test_explicit_area_weights_apply_to_global_and_regional_scores():
    from zhenmode.evaluation.metrics import score_snapshot

    lat, lon = np.array([45., 58.]), np.array([310., 330.])
    area = np.array([[1., 2.], [3., 4.]])
    error = np.array([[1., 3.], [2., 5.]])
    result = score_snapshot(error, np.zeros_like(error), np.ones((2, 2), bool),
                            lat, lon, area=area)
    assert result["global"]["raw_bias"] == pytest.approx(3.3)
    assert result["north_atlantic_40_60"]["raw_bias"] == pytest.approx(3.3)
    assert result["near_wall_55_60"]["raw_rmse"] == pytest.approx(np.sqrt(118. / 6.))
    assert result["area_source"] == "provided_cell_area"


def test_angular_filter_uses_coordinates_and_excludes_land():
    from zhenmode.evaluation.metrics import smooth_2d_global

    lat, lon = np.array([0., 1.]), np.array([0., 1., 20., 359.])
    ocean = np.ones((4, 2), bool)
    ocean[1, 1] = False
    field = np.array([[0., 2.], [4., 1e12], [200., 300.], [6., 8.]])
    area = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    result = smooth_2d_global(field, lat, deg=2., lon=lon, ocean=ocean, area=area)
    for east, north in zip(*np.where(ocean), strict=True):
        longitude_distance = np.abs((lon - lon[east] + 180.) % 360. - 180.)
        selected = ocean & (longitude_distance[:, None] <= 2.) & (np.abs(lat - lat[north])[None, :] <= 2.)
        assert result[east, north] == pytest.approx(np.average(field[selected], weights=area[selected]))
    field[~ocean] = np.nan
    np.testing.assert_allclose(result[ocean], smooth_2d_global(field, lat, deg=2.,
                               lon=lon, ocean=ocean, area=area)[ocean])


def test_area_geometry_uses_spherical_edges_and_preserves_order():
    lat, lon = np.array([-30., 30.]), np.array([45., 135., 225., 315.])
    radius = 2.1e6
    area = cell_area(lat, lon, radius=radius)
    exact = radius ** 2 * np.pi / 2. * np.sin(np.pi / 3.)
    np.testing.assert_allclose(area, exact, rtol=1e-14)
    np.testing.assert_allclose(cell_area(lat[::-1], lon[::-1], radius=radius), area[::-1, ::-1])
    np.testing.assert_allclose(cell_area(lat, (lon + 270.) % 360., radius=radius), area)


@pytest.mark.parametrize("area", [np.zeros((2, 2)), np.full((2, 2), -1.),
                                 np.full((2, 2), np.nan), np.ones((2, 3))])
def test_spatial_score_rejects_invalid_wet_weights(area):
    from zhenmode.evaluation.metrics import score_snapshot

    with pytest.raises(ValueError, match="area|weight|shape"):
        score_snapshot(np.ones((2, 2)), np.zeros((2, 2)), np.ones((2, 2), bool),
                       np.array([0., 30.]), np.array([0., 180.]), area=area)


def test_wet_missing_value_is_not_removed_to_improve_score():
    from zhenmode.evaluation.metrics import score_snapshot

    model = np.array([[0., 0.], [0., np.nan]])
    result = score_snapshot(model, np.zeros((2, 2)), np.ones((2, 2), bool),
                            np.array([0., 30.]), np.array([0., 180.]))
    assert not result["coverage_complete"]
    assert np.isnan(result["global"]["raw_rmse"])
    assert result["global"]["n"] == 4


def test_weighted_pattern_matches_independent_small_grid():
    from zhenmode.evaluation.metrics import score_snapshot

    model = np.array([[1., 4.], [2., 8.]])
    reference = np.array([[2., 3.], [4., 7.]])
    area = np.array([[1., 2.], [3., 4.]])
    result = score_snapshot(model, reference, np.ones((2, 2), bool),
                            np.array([0., 60.]), np.array([0., 180.]), area=area)
    model_anomaly = model - np.average(model, weights=area)
    reference_anomaly = reference - np.average(reference, weights=area)
    exact = np.sum(area * model_anomaly * reference_anomaly) / np.sqrt(
        np.sum(area * model_anomaly ** 2) * np.sum(area * reference_anomaly ** 2))
    assert result["a2_corr"] == pytest.approx(exact)
    assert result["global_a2_rmse_c"] == pytest.approx(np.sqrt(np.average((model - reference) ** 2, weights=area)))


def test_snapshot_domain_hash_identifies_mask_and_area_changes():
    from zhenmode.evaluation.metrics import score_snapshot

    model, ocean, area = np.zeros((2, 2)), np.ones((2, 2), bool), np.ones((2, 2))
    arguments = (model, model, ocean, np.array([0., 30.]), np.array([0., 180.]))
    first = score_snapshot(*arguments, area=area)
    second = score_snapshot(*arguments, area=area * 2.)
    assert first["comparison_domain_sha256"] != second["comparison_domain_sha256"]
    ocean[0, 0] = False
    third = score_snapshot(*arguments, area=area)
    assert first["comparison_domain_sha256"] != third["comparison_domain_sha256"]


def test_snapshot_reference_hash_changes_only_with_scored_reference_values():
    from zhenmode.evaluation.metrics import score_snapshot

    model, reference, ocean = np.zeros((2, 2)), np.ones((2, 2)), np.ones((2, 2), bool)
    ocean[0, 0] = False
    arguments = (model, reference, ocean, np.array([0., 30.]), np.array([0., 180.]))
    first = score_snapshot(*arguments)
    reference[0, 0] = np.nan
    land_changed = score_snapshot(*arguments)
    assert first["comparison_reference_sha256"] == land_changed["comparison_reference_sha256"]
    reference[1, 1] += 1.
    wet_changed = score_snapshot(*arguments)
    assert first["comparison_reference_sha256"] != wet_changed["comparison_reference_sha256"]


def test_constant_patterns_do_not_get_a_perfect_roundoff_correlation():
    from zhenmode.evaluation.metrics import score_snapshot

    model, ocean = np.full((3, 3), 20.), np.ones((3, 3), bool)
    result = score_snapshot(model, model, ocean, np.array([30., 45., 55.]),
                            np.array([300., 320., 0.]))
    assert np.isnan(result["a1_corr"])
    assert np.isnan(result["a2_corr"])


def test_masked_wet_error_and_weight_fail_closed():
    from zhenmode.evaluation.metrics import score_snapshot

    ocean = np.ones((2, 2), bool)
    masked = np.ma.array(np.ones((2, 2)), mask=[[True, False], [False, False]])
    result = score_snapshot(masked, np.ones((2, 2)), ocean,
                            np.array([0., 30.]), np.array([0., 180.]))
    assert not result["coverage_complete"]
    with pytest.raises(ValueError, match="weight"):
        score_snapshot(np.ones((2, 2)), np.ones((2, 2)), ocean,
                       np.array([0., 30.]), np.array([0., 180.]), area=masked)
