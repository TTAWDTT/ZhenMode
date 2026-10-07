"""Known vertical profiles and refused gaps; no fitted climate tolerance."""

import numpy as np
import pytest

from zhenmode.model.inputs.initial_conditions import extend_native_bottom_pairs


def test_bottom_copies_same_column_and_preserves_all_resolved_values():
    fields = np.array([[[[18.0, 15.0, np.nan, np.nan]]], [[[34.0, 34.5, np.nan, np.nan]]]])
    result = extend_native_bottom_pairs(fields, np.ones((1, 1, 4)), [0.0, 5.0, 15.0, 30.0])
    np.testing.assert_array_equal(
        result["fields"], [[[[18.0, 15.0, 15.0, 15.0]]], [[[34.0, 34.5, 34.5, 34.5]]]]
    )
    np.testing.assert_array_equal(result["donor_native_level_index"], [[[-1, -1, 1, 1]]])
    np.testing.assert_array_equal(result["extension_distance_m"], [[[0.0, 0.0, 10.0, 25.0]]])
    np.testing.assert_allclose(
        result["linear_sensitivity_fields"],
        [[[[18.0, 15.0, 9.0, 0.0]]], [[[34.0, 34.5, 35.5, 37.0]]]],
    )
    assert not result["unresolved_mask"].any()
    assert np.isnan(fields[0, 0, 0, 2])  # Original source was not mutated.


def test_interior_surface_unanchored_and_dry_nodes_remain_missing():
    fields = np.full((2, 2, 1, 4), np.nan)
    fields[:, 0, 0, 1] = [12.0, 35.0]
    fields[:, 0, 0, 3] = [4.0, 36.0]
    wet = np.ones((2, 1, 4))
    wet[1, 0, 2:] = 0
    result = extend_native_bottom_pairs(fields, wet, [0.0, 5.0, 15.0, 30.0])
    assert not result["bottom_filled_mask"].any()
    np.testing.assert_array_equal(
        result["unresolved_mask"], [[[True, False, True, False]], [[True, True, False, False]]]
    )
    np.testing.assert_array_equal(result["fields"], fields)


def test_one_resolved_point_has_no_claimed_independent_linear_scenario():
    fields = np.array([[[[12.0, np.nan]]], [[[35.0, np.nan]]]])
    result = extend_native_bottom_pairs(fields, np.ones((1, 1, 2)), [0.0, 10.0])
    assert result["bottom_filled_mask"][0, 0, 1]
    assert not result["linear_sensitivity_support"].any()


def test_malformed_pair_and_wet_topology_are_refused():
    fields = np.full((2, 1, 1, 3), np.nan)
    fields[:, 0, 0, 0] = [12.0, 35.0]
    fields[0, 0, 0, 1] = 10.0
    with pytest.raises(ValueError, match="paired"):
        extend_native_bottom_pairs(fields, np.ones((1, 1, 3)), [0.0, 5.0, 15.0])
    fields[:, 0, 0, 1] = np.nan
    for wet, depth in (([[[1, 0, 1]]], [0.0, 5.0, 15.0]), ([[[1, 1, 1]]], [0.0, 15.0, 5.0])):
        with pytest.raises(ValueError, match="contiguous"):
            extend_native_bottom_pairs(fields, wet, depth)
