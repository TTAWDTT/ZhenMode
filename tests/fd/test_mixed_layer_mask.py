"""Regression tests for spatially restricted mixed-layer heat capacity."""

from dataclasses import replace

import numpy as np

from ocean_solver.config.definitions import PhysicsConfig
from ocean_solver.model.factory import make_solver_global
from tests.support.grid import all_wet_grid


def _mask_params(mask, mixed_depth=50.0):
    grid = all_wet_grid(nx=16, ny=8, nz=4)
    physics = replace(PhysicsConfig(), nu_h=0.0, nu_bi=0.0,
                      nu_v=0.0, kappa_h=0.0, kappa_v=0.0,
                      kappa_conv=0.0, kappa_gm=0.0, kappa_redi=0.0)
    forcing = tuple(np.zeros((grid.nx, grid.ny)) for _ in range(3))
    _, _, _, params, _ = make_solver_global(
        grid, physics, 60.0, forcing=forcing, lambda_bulk=0.0,
        mixed_layer_depth_m=mixed_depth,
        mixed_layer_mask=mask,
        T_init=None, S_init=None,
        polar_cap_rows=0, polar_cap_taper=0,
        mode_split=False, dtype='float64', return_params=True)
    return params


def test_mixed_layer_mask_defaults_to_global():
    params = _mask_params(None)
    assert params.mixed_layer_mask_2d.shape == (16, 8)
    assert float(np.asarray(params.mixed_layer_mask_2d).min()) == 1.0


def test_solver_accepts_spatial_mixed_layer_mask():
    mask = np.zeros((16, 8), dtype=float)
    mask[:8, :] = 1.0
    params = _mask_params(mask, mixed_depth=20.0)
    assert params.mixed_layer_mask_2d.shape == (16, 8)
    assert float(np.asarray(params.mixed_layer_mask_2d).sum()) == 64.0



def test_solver_accepts_stratification_mld_2d():
    mask = np.ones((16, 8), dtype=float)
    depth_2d = np.full((16, 8), 20.0)
    depth_2d[8:, :] = 50.0
    params = _mask_params(mask, mixed_depth=30.0)
    assert params.mixed_layer_depth_2d is None
    # Rebuild only to exercise the new keyword path; compare the stored field.
    from tests.support.grid import all_wet_grid
    grid = all_wet_grid(nx=16, ny=8, nz=4)
    physics = replace(PhysicsConfig(), nu_h=0.0, nu_bi=0.0,
                      nu_v=0.0, kappa_h=0.0, kappa_v=0.0,
                      kappa_conv=0.0, kappa_gm=0.0, kappa_redi=0.0)
    forcing = tuple(np.zeros((grid.nx, grid.ny)) for _ in range(3))
    _, _, _, params, _ = make_solver_global(
        grid, physics, 60.0, forcing=forcing, lambda_bulk=0.0,
        mixed_layer_depth_m=30.0, mixed_layer_mask=mask,
        mixed_layer_depth_2d=depth_2d,
        T_init=None, S_init=None,
        polar_cap_rows=0, polar_cap_taper=0,
        mode_split=False, dtype='float64', return_params=True)
    assert np.allclose(np.asarray(params.mixed_layer_depth_2d), depth_2d)
