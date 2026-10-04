from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from tests.support.grid import all_wet_grid
from zhenmode.model.config.definitions import PhysicsConfig
from zhenmode.model.dynamics.processes import _compute_tracer_tendency
from zhenmode.model.factory import make_solver_global

jax.config.update('jax_enable_x64', True)


def _params_and_state(T_value, ice_salt_flux, freeze_temp):
    grid = all_wet_grid(nx=16, ny=8, nz=4)
    physics = replace(PhysicsConfig(), nu_h=0.0, nu_bi=0.0,
                      nu_v=0.0, kappa_h=0.0, kappa_v=0.0,
                      kappa_conv=0.0, kappa_gm=0.0, kappa_redi=0.0)
    forcing = tuple(np.zeros((grid.nx, grid.ny)) for _ in range(3))
    T = np.full((grid.nx, grid.ny, grid.nz), T_value, dtype=float)
    S = np.full_like(T, 35.0)
    _, init_fn, _, params, _ = make_solver_global(
        grid, physics, 60.0, forcing=forcing, lambda_bulk=0.0,
        ice_freeze_temp_c=freeze_temp, ice_salt_flux=ice_salt_flux,
        T_init=T, S_init=S,
        polar_cap_rows=0, polar_cap_taper=0,
        mode_split=False, dtype='float64', return_params=True)
    return init_fn(T_init=jnp.array(T), S_init=jnp.array(S)), params


def test_solver_applies_ice_salt_flux_only_below_freezing():
    state, params = _params_and_state(
        T_value=20.0, ice_salt_flux=1e-6, freeze_temp=20.0)
    _, dSdt = _compute_tracer_tendency(state, params)
    assert float(jnp.min(dSdt[:, :, 0])) == 1e-6
    assert float(jnp.max(jnp.abs(dSdt[:, :, 1:]))) == 0.0


def test_solver_ice_salt_flux_off_by_default():
    state, params = _params_and_state(
        T_value=20.0, ice_salt_flux=0.0, freeze_temp=20.0)
    _, dSdt = _compute_tracer_tendency(state, params)
    assert float(jnp.max(jnp.abs(dSdt))) == 0.0
