"""Regression tests for the minimal stateful mixed-layer/ice closure."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from _helpers import all_wet_grid

from config import PhysicsConfig
from jax_solver_global import (
    JaxStateG,
    _compute_tracer_tendency,
    _dynamic_ice_closure,
    make_solver_global,
)


def _params(T_atm_value=-20.0, lambda_bulk=80.0, dt=864000.0, ice_mask=None, dynamic_ice=True):
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    physics = replace(PhysicsConfig(), nu_h=0.0, nu_bi=0.0,
                      nu_v=0.0, kappa_h=0.0, kappa_v=0.0,
                      kappa_conv=0.0, kappa_gm=0.0, kappa_redi=0.0)
    forcing = tuple(np.zeros((grid.nx, grid.ny)) for _ in range(3))
    _, _, _, params, _ = make_solver_global(
        grid, physics, dt, forcing=forcing, T_atm=np.full((8, 8), T_atm_value),
        lambda_bulk=lambda_bulk, mixed_layer_depth_m=20.0,
        dynamic_ice=dynamic_ice, ice_mask=ice_mask,
        polar_cap_rows=0, polar_cap_taper=0,
        mode_split=False, dtype='float64', return_params=True)
    return grid, params


def test_dynamic_ice_grows_and_pins_surface_at_freezing():
    grid, params = _params(T_atm_value=-20.0)
    state = JaxStateG(
        jnp.zeros((8, 8, 4)), jnp.zeros((8, 8, 4)),
        jnp.full((8, 8, 4), 5.0), jnp.full((8, 8, 4), 35.0),
        jnp.zeros((8, 8)), jnp.zeros((8, 8)))
    new = _dynamic_ice_closure(state, params)
    ice = np.asarray(new.ice)
    assert np.all(ice > 0.0)
    assert np.allclose(np.asarray(new.T[:, :, 0]), -1.8)
    assert np.all(np.asarray(new.S[:, :, 0]) > 35.0)


def test_dynamic_ice_melts_under_warm_air_and_rejects_negative_salt():
    grid, params = _params(T_atm_value=10.0)
    state = JaxStateG(
        jnp.zeros((8, 8, 4)), jnp.zeros((8, 8, 4)),
        jnp.full((8, 8, 4), -1.8), jnp.full((8, 8, 4), 35.0),
        jnp.zeros((8, 8)), jnp.full((8, 8), 1.0))
    new = _dynamic_ice_closure(state, params)
    ice = np.asarray(new.ice)
    assert np.all(ice >= 0.0)
    assert np.all(ice < 1.0)
    assert np.all(np.asarray(new.S[:, :, 0]) < 35.0)


def test_dynamic_ice_respects_latitude_mask():
    grid, params = _params(T_atm_value=-20.0, ice_mask=np.zeros((8, 8)))
    state = JaxStateG(
        jnp.zeros((8, 8, 4)), jnp.zeros((8, 8, 4)),
        jnp.full((8, 8, 4), 5.0), jnp.full((8, 8, 4), 35.0),
        jnp.zeros((8, 8)), jnp.full((8, 8), 1.0))
    new = _dynamic_ice_closure(state, params)
    assert np.all(np.asarray(new.ice) == 0.0)
    assert np.allclose(np.asarray(new.T[:, :, 0]), 5.0)
    assert np.allclose(np.asarray(new.S[:, :, 0]), 35.0)



def test_dynamic_ice_mask_does_not_disable_surface_exchange():
    masked, _ = _params(T_atm_value=-20.0, ice_mask=np.zeros((8, 8)))
    _, _, _, params_on, _ = make_solver_global(
        masked, __import__('config').PhysicsConfig(), 864000.0,
        forcing=tuple(np.zeros((8, 8)) for _ in range(3)),
        T_atm=np.full((8, 8), -20.0), lambda_bulk=80.0,
        mixed_layer_depth_m=20.0, dynamic_ice=True,
        ice_mask=np.zeros((8, 8)), polar_cap_rows=0, polar_cap_taper=0,
        mode_split=False, dtype='float64', return_params=True)
    _, _, _, params_off, _ = make_solver_global(
        masked, __import__('config').PhysicsConfig(), 864000.0,
        forcing=tuple(np.zeros((8, 8)) for _ in range(3)),
        T_atm=np.full((8, 8), -20.0), lambda_bulk=80.0,
        mixed_layer_depth_m=20.0, dynamic_ice=False,
        polar_cap_rows=0, polar_cap_taper=0,
        mode_split=False, dtype='float64', return_params=True)
    state = JaxStateG(
        jnp.zeros((8, 8, 4)), jnp.zeros((8, 8, 4)),
        jnp.full((8, 8, 4), 5.0), jnp.full((8, 8, 4), 35.0),
        jnp.zeros((8, 8)), jnp.zeros((8, 8)))
    dTdt_on, dSdt_on = _compute_tracer_tendency(state, params_on)
    dTdt_off, dSdt_off = _compute_tracer_tendency(state, params_off)
    assert np.allclose(np.asarray(dTdt_on), np.asarray(dTdt_off))
    assert np.allclose(np.asarray(dSdt_on), np.asarray(dSdt_off))
