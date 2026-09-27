"""Regression tests for the minimal mixed-layer/sea-ice closure."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mixed_layer_ice import MLIceConfig, mixed_layer_ice_step, surface_heat_flux

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

    from _helpers import all_wet_grid

    from config import PhysicsConfig
    from jax_solver_global import make_solver_global

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
import jax.numpy as jnp
import numpy as np

from _helpers import all_wet_grid
from config import PhysicsConfig
from dataclasses import replace

from jax_solver_global import JaxStateG, _mixed_layer_depth_with_gate, make_solver_global


def _params(T_atm_value, cooling_gate=True):
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    physics = replace(PhysicsConfig(), nu_h=0.0, nu_bi=0.0,
                      nu_v=0.0, kappa_h=0.0, kappa_v=0.0,
                      kappa_conv=0.0, kappa_gm=0.0, kappa_redi=0.0)
    forcing = tuple(np.zeros((8, 8)) for _ in range(3))
    _, _, _, params, _ = make_solver_global(
        grid, physics, 60.0, forcing=forcing,
        T_atm=np.full((8, 8), T_atm_value), lambda_bulk=1.0,
        mixed_layer_depth_m=20.0,
        mixed_layer_cooling_gate=cooling_gate,
        polar_cap_rows=0, polar_cap_taper=0,
        mode_split=False, dtype='float64', return_params=True)
    return params


def _state(T_value):
    return JaxStateG(jnp.zeros((8, 8, 4)), jnp.zeros((8, 8, 4)),
                     jnp.full((8, 8, 4), T_value), jnp.full((8, 8, 4), 35.0),
                     jnp.zeros((8, 8)), jnp.zeros((8, 8)))


def test_mixed_layer_cooling_gate_keeps_deep_cooling():
    params = _params(T_atm_value=5.0, cooling_gate=True)
    depth = np.asarray(_mixed_layer_depth_with_gate(_state(10.0), params))
    assert np.allclose(depth, 20.0)


def test_mixed_layer_cooling_gate_falls_back_when_warming():
    params = _params(T_atm_value=15.0, cooling_gate=True)
    depth = np.asarray(_mixed_layer_depth_with_gate(_state(10.0), params))
    assert np.allclose(depth, params.dz_surface)


def test_mixed_layer_cooling_gate_off_keeps_depth():
    params = _params(T_atm_value=15.0, cooling_gate=False)
    depth = np.asarray(_mixed_layer_depth_with_gate(_state(10.0), params))
    assert np.allclose(depth, 20.0)



def _ice_state(T_value, ice_value):
    return JaxStateG(jnp.zeros((8, 8, 4)), jnp.zeros((8, 8, 4)),
                     jnp.full((8, 8, 4), T_value), jnp.full((8, 8, 4), 35.0),
                     jnp.zeros((8, 8)), jnp.full((8, 8), ice_value))


def test_mixed_layer_ice_gate_uses_depth_only_where_ice():
    params = _params(15.0, cooling_gate=False)._replace(mixed_layer_gate_mode="ice")
    open_water = _ice_state(10.0, ice_value=0.0)
    depth = np.asarray(_mixed_layer_depth_with_gate(open_water, params))
    assert np.allclose(depth, params.dz_surface)

    ice_water = _ice_state(10.0, ice_value=0.25)
    depth = np.asarray(_mixed_layer_depth_with_gate(ice_water, params))
    assert np.allclose(depth, 20.0)


def test_mixed_layer_cooling_ice_gate_combines_both():
    params = _params(15.0, cooling_gate=False)._replace(mixed_layer_gate_mode="cooling_ice")
    warm_open_water = _ice_state(10.0, ice_value=0.0)
    depth = np.asarray(_mixed_layer_depth_with_gate(warm_open_water, params))
    assert np.allclose(depth, params.dz_surface)

    cooling_open_water = _ice_state(20.0, ice_value=0.0)
    depth = np.asarray(_mixed_layer_depth_with_gate(cooling_open_water, params))
    assert np.allclose(depth, 20.0)

    warm_ice_water = _ice_state(10.0, ice_value=0.25)
    depth = np.asarray(_mixed_layer_depth_with_gate(warm_ice_water, params))
    assert np.allclose(depth, 20.0)


def test_mixed_layer_gate_mode_rejects_unknown_mode():
    params = _params(15.0, cooling_gate=False)._replace(mixed_layer_gate_mode="not_a_mode")
    try:
        _mixed_layer_depth_with_gate(_ice_state(10.0, ice_value=0.0), params)
    except ValueError as exc:
        assert "not_a_mode" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown gate mode")
