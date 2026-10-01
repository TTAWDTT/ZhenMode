"""Regression tests for the minimal stateful mixed-layer/ice closure."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from _helpers import all_wet_grid

from config import PhysicsConfig
from jax_solver_global import JaxStateG, _dynamic_ice_closure, make_solver_global


def _params(T_atm_value=-20.0, lambda_bulk=80.0, dt=864000.0):
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    grid = replace(grid, z=np.array([0., -5., -20., -50.]), dz=np.array([5., 15., 30.]))
    physics = replace(PhysicsConfig(), nu_h=0.0, nu_bi=0.0,
                      nu_v=0.0, kappa_h=0.0, kappa_v=0.0,
                      kappa_conv=0.0, kappa_gm=0.0, kappa_redi=0.0)
    forcing = tuple(np.zeros((grid.nx, grid.ny)) for _ in range(3))
    _, _, _, params, _ = make_solver_global(
        grid, physics, dt, forcing=forcing, T_atm=np.full((8, 8), T_atm_value),
        lambda_bulk=lambda_bulk, mixed_layer_depth_m=20.0,
        dynamic_ice=True,
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
