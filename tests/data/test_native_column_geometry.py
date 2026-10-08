"""Constructed column geometry controls; not FD operator/coupled-run certification."""

import numpy as np
import pytest

from zhenmode.model.solver.geometry.grid import fixed_reference_nodal_cells, nodal_control_thickness


def test_reference_columns_include_shallow_water_and_actual_bottom():
    z = np.array([0.0, -5.0, -15.0, -30.0])
    bed = np.array([[0.5, 7.0, 21.0], [27.0, 30.0, 0.0]])
    result = fixed_reference_nodal_cells(z, bed, bed > 0)
    # Independently declared widths, including the 22.5..27 gap an ordinary
    # clamp of the global midpoint boundaries would lose.
    expected = np.array(
        [
            [[0.5, 0.0, 0.0, 0.0], [2.5, 4.5, 0.0, 0.0], [2.5, 7.5, 11.0, 0.0]],
            [[2.5, 7.5, 17.0, 0.0], [2.5, 7.5, 12.5, 7.5], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    np.testing.assert_array_equal(result["thickness_m"], expected)
    np.testing.assert_array_equal(result["thickness_m"].sum(axis=-1), bed)
    np.testing.assert_array_equal(result["wet_node_mask"], expected > 0)
    np.testing.assert_array_equal(result["bottom_node_mask"].sum(axis=-1), bed > 0)
    assert np.all(result["cell_top_m"] <= result["cell_bottom_m"])
    assert np.all(result["cell_bottom_m"] <= bed[..., None])


def test_constant_inventory_and_destroyed_width_control():
    bed = np.array([[7.0, 27.0]])
    result = fixed_reference_nodal_cells([0.0, -5.0, -15.0, -30.0], bed, np.ones(bed.shape))
    area = np.array([[2.0, 3.0]])
    # Constant tracer=4: exact continuous inventory=4*(2*7+3*27)=380.
    assert float(np.sum(area[..., None] * result["thickness_m"] * 4.0)) == 380.0
    damaged = result["thickness_m"].copy()
    damaged[0, 1, 2] -= 4.5
    assert float(np.sum(area[..., None] * damaged * 4.0)) != 380.0


def test_existing_uniform_nodal_geometry_remains_unchanged():
    np.testing.assert_array_equal(
        nodal_control_thickness([0.0, -5.0, -15.0, -30.0]), [2.5, 7.5, 12.5, 7.5]
    )
    # Existing first-sample-below-surface convention is preserved.
    np.testing.assert_array_equal(nodal_control_thickness([-2.0, -6.0, -10.0]), [4.0, 4.0, 2.0])


@pytest.mark.parametrize(
    "bed,wet,z",
    [
        ([[31.0]], [[1]], [0.0, -5.0, -15.0, -30.0]),
        ([[1.0]], [[0]], [0.0, -5.0, -15.0, -30.0]),
        ([[0.0]], [[1]], [0.0, -5.0, -15.0, -30.0]),
        ([[np.nan]], [[1]], [0.0, -5.0, -15.0, -30.0]),
        ([[1.0]], [[0.5]], [0.0, -5.0, -15.0, -30.0]),
        ([[1.0]], [[1]], [-1.0, -5.0, -15.0, -30.0]),
        ([[1.0]], [[1]], [0.0, -5.0, -5.0, -30.0]),
        ([[1.0]], [[1]], [0.0, -5.0, np.nan, -30.0]),
    ],
)
def test_invalid_or_truncated_geometry_is_not_accepted(bed, wet, z):
    with pytest.raises(ValueError):
        fixed_reference_nodal_cells(z, bed, wet)
