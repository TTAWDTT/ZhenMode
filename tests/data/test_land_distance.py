
import numpy as np

from zhenmode.model.geometry.mesh import land_distance_from_land_mask


def test_land_distance_periodic_and_connectivity():
    ocean = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=bool)
    dist8 = land_distance_from_land_mask(ocean, connectivity=8)
    assert dist8[0, 0] == 1.0
    assert dist8[0, 1] == 1.0
    dist4 = land_distance_from_land_mask(ocean, connectivity=4)
    assert dist4[0, 0] == 2.0
    assert dist4[0, 1] == 1.0

    # Longitude wraps, so both cells adjacent to the edge are one step away.
    strip = np.array([[0, 1, 1, 1, 0]], dtype=bool)
    dist = land_distance_from_land_mask(strip, connectivity=4)
    np.testing.assert_array_equal(dist, [[0, 1, 2, 1, 0]])
