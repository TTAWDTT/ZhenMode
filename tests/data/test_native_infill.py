"""Known Dirichlet answers, paired support and unanchored/fault controls."""

import numpy as np
import pytest

from zhenmode.model.inputs.initial_conditions import smooth_native_paired_holes


def test_symmetric_gap_has_known_harmonic_answer_and_retains_original_values():
    values = np.full((2, 4, 2), np.nan)
    values[:, 0] = [[0.0, 0.0], [34.0, 34.0]]
    values[:, 2] = [[10.0, 10.0], [36.0, 36.0]]
    wet = np.ones((4, 2))
    wet[3] = 0.0
    r = smooth_native_paired_holes(values, wet, [45.0, 135.0, 225.0, 315.0], [-45.0, 45.0])
    np.testing.assert_array_equal(r["fields"][:, [0, 2]], values[:, [0, 2]])
    np.testing.assert_allclose(r["fields"][:, 1], [[5.0, 5.0], [35.0, 35.0]], rtol=0, atol=1e-14)
    assert r["infill_mask"].sum() == 2 and not r["unresolved_mask"].any()
    assert r["linear_system_relative_residual"] < 1e-15
    assert np.all(r["nearest_original_anchor_flat_index"][r["infill_mask"]] >= 0)
    assert np.all(r["nearest_anchor_path_distance_m"][r["infill_mask"]] > 0.0)


def test_longitude_seam_is_open_but_land_and_latitude_wrap_are_closed():
    values = np.full((2, 4, 2), np.nan)
    wet = np.zeros((4, 2))
    wet[0, 0] = wet[3, 0] = wet[1, 1] = 1.0
    values[:, 3, 0] = [7.0, 35.0]
    r = smooth_native_paired_holes(values, wet, [45.0, 135.0, 225.0, 315.0], [-45.0, 45.0])
    np.testing.assert_allclose(r["fields"][:, 0, 0], [7.0, 35.0], rtol=0, atol=1e-14)
    assert r["unresolved_mask"][1, 1] and np.isnan(r["fields"][:, 1, 1]).all()
    assert r["unsupported_components"] == [{"native_flat_indices": [3]}]
    assert r["nearest_original_anchor_flat_index"][0, 0] == 6
    assert r["nearest_original_anchor_flat_index"][1, 1] == -1


def test_missing_one_field_never_uses_unpaired_temperature_as_an_anchor():
    values = np.full((2, 4, 2), np.nan)
    values[:, 0] = [[8.0, 8.0], [35.0, 35.0]]
    values[0, 1] = 900.0
    wet = np.zeros((4, 2))
    wet[:2] = 1.0
    r = smooth_native_paired_holes(values, wet, [45.0, 135.0, 225.0, 315.0], [-45.0, 45.0])
    np.testing.assert_allclose(r["fields"][:, 1], [[8.0, 8.0], [35.0, 35.0]], rtol=0, atol=1e-13)
    assert r["original_paired_mask"].sum() == 2


def test_no_anchor_never_becomes_water_via_a_global_mean():
    values = np.full((2, 4, 2), np.nan)
    r = smooth_native_paired_holes(
        values, np.ones((4, 2)), [45.0, 135.0, 225.0, 315.0], [-45.0, 45.0]
    )
    assert r["unresolved_mask"].sum() == 8
    assert np.isnan(r["fields"]).all() and not r["infill_mask"].any()


@pytest.mark.parametrize("defect", ["infinity", "longitude", "pole", "fractional", "complex"])
def test_invalid_support_or_coordinates_are_rejected(defect):
    values = np.ones((2, 4, 2))
    wet = np.ones((4, 2))
    lon = [45.0, 135.0, 225.0, 315.0]
    lat = [-45.0, 45.0]
    if defect == "infinity":
        values[0, 0, 0] = np.inf
    if defect == "longitude":
        lon[-1] = 316.0
    if defect == "pole":
        lat[-1] = 90.0
    if defect == "fractional":
        wet[0, 0] = 0.5
    if defect == "complex":
        values = values.astype(complex) + 1j
    with pytest.raises(ValueError):
        smooth_native_paired_holes(values, wet, lon, lat)
