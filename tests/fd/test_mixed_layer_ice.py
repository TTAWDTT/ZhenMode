"""Regression tests for the minimal mixed-layer/sea-ice closure."""

import numpy as np

from zhenmode.model.physics.ice import MLIceConfig, mixed_layer_ice_step, surface_heat_flux

CFG = MLIceConfig(mixed_layer_depth_m=50.0, freeze_temp_c=-1.8,
                  bulk_lambda=80.0)


def test_cold_air_grows_ice_and_salt_flux_is_positive():
    T_new, ice_new, q, salt_flux = mixed_layer_ice_step(
        T_mld=5.0, ice_thickness_m=0.0, T_atm=-20.0, dt_s=864000.0, cfg=CFG)
    assert q < 0
    assert T_new == CFG.freeze_temp_c
    assert ice_new > 0.0
    assert salt_flux > 0.0


def test_warm_air_melts_existing_ice():
    T_new, ice_new, q, salt_flux = mixed_layer_ice_step(
        T_mld=0.0, ice_thickness_m=1.0, T_atm=10.0, dt_s=864000.0, cfg=CFG)
    assert q > 0
    assert T_new == CFG.freeze_temp_c
    assert ice_new < 1.0
    assert salt_flux < 0.0


def test_zero_flux_preserves_state():
    T_new, ice_new, q, salt_flux = mixed_layer_ice_step(
        T_mld=0.0, ice_thickness_m=1.0, T_atm=0.0, dt_s=60.0, cfg=CFG)
    assert q == 0.0
    assert T_new == 0.0
    assert ice_new == 1.0
    assert salt_flux == 0.0


def test_surface_flux_is_weakened_under_ice():
    open_water = surface_heat_flux(0.0, 10.0, 0.0, CFG)
    ice_covered = surface_heat_flux(0.0, 10.0, 2.0, CFG)
    assert ice_covered < open_water
    assert ice_covered > 0.0


def test_solver_accepts_mixed_layer_depth():
    from dataclasses import replace

    from tests.support.grid import all_wet_grid
    from zhenmode.model.config.definitions import PhysicsConfig
    from zhenmode.model.factory import make_solver_global

    grid = all_wet_grid(nx=16, ny=8, nz=4)
    physics = replace(PhysicsConfig(), nu_h=0.0, nu_bi=0.0,
                      nu_v=0.0, kappa_h=0.0, kappa_v=0.0,
                      kappa_conv=0.0, kappa_gm=0.0, kappa_redi=0.0)
    forcing = tuple(np.zeros((grid.nx, grid.ny)) for _ in range(3))
    _, _, _, params, _ = make_solver_global(
        grid, physics, 60.0, forcing=forcing, lambda_bulk=0.0,
        mixed_layer_depth_m=50.0, T_init=None, S_init=None,
        polar_cap_rows=0, polar_cap_taper=0,
        mode_split=False, dtype='float64', return_params=True)
    assert params.mixed_layer_depth_m == 50.0
