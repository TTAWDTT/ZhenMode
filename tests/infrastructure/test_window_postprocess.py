"""Small geometry contracts for snapshot postprocessing; no solver integration."""

import importlib.util

import numpy as np

from tests.support.paths import REPOSITORY_ROOT

spec = importlib.util.spec_from_file_location(
    'window_postprocess', REPOSITORY_ROOT / 'scripts/controlled_window/postprocess_6h.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_periodic_seam_connects_but_closed_latitude_does_not():
    wet = np.zeros((5, 5), dtype=bool)
    wet[0, 1] = wet[4, 1] = True
    wet[2, 0] = wet[2, 4] = True
    basins = module.connected_basins(wet, np.ones(wet.shape))
    assert len(basins) == 3
    assert basins['basin_1'][0, 1] and basins['basin_1'][4, 1]
    assert sorted(mask.sum() for mask in basins.values()) == [1, 1, 2]


def test_constant_pressure_gauge_and_land_values_do_not_force_water():
    wet = np.ones((4, 4, 2))
    wet[2, 2] = 0
    arrays = dict(wet_mask_z=wet, cos_lat=np.array([.5, .7, .8, .9]),
                  inv_dx=np.full((4, 4, 1), 1 / 1000))
    field = np.full(wet.shape, 10000.)
    field[2, 2] = -1e30
    gx, gy = module.gradient(field, arrays, {'inv_dy': 1 / 1000})
    assert np.array_equal(gx, np.zeros_like(gx))
    assert np.array_equal(gy, np.zeros_like(gy))


def test_meridional_linear_pressure_has_closed_wall_half_gradient():
    shape = (4, 4, 2)
    arrays = dict(wet_mask_z=np.ones(shape), cos_lat=np.ones(4),
                  inv_dx=np.full((4, 4, 1), 1 / 1000))
    field = np.broadcast_to(np.arange(4)[None, :, None], shape)
    gx, gy = module.gradient(field, arrays, {'inv_dy': 1 / 1000})
    np.testing.assert_array_equal(gx, np.zeros(shape))
    np.testing.assert_allclose(gy[:, 1:3], .001, rtol=0, atol=0)
    np.testing.assert_allclose(gy[:, [0, 3]], .0005, rtol=0, atol=0)
