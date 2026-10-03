"""Dynamic monthly air-temperature target reaches the bulk-flux term."""


from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from ocean_solver.config.definitions import PhysicsConfig
from ocean_solver.model.factory import make_solver_global
from ocean_solver.runtime.entry import interp_monthly_field, interp_monthly_field_jit
from tests.support.grid import all_wet_grid


def test_monthly_field_blend_matches_calendar():
    fields = np.arange(12.0)[:, None, None] * np.ones((1, 2, 2))
    assert interp_monthly_field(fields, 15.0, blend_days=5.0)[0, 0] == pytest.approx(0.0)
    # At day 30 (January again after modulo), boundary blending points to December.
    assert interp_monthly_field_jit(fields, 29.5, blend_days=5.0)[0, 0] == pytest.approx(0.4)


def test_dynamic_bulk_target_changes_surface_temperature():
    grid = all_wet_grid(nx=12, ny=8, nz=4)
    physics = replace(
        PhysicsConfig(),
        nu_h=0.0, nu_bi=0.0, nu_v=0.0,
        kappa_h=0.0, kappa_v=0.0, kappa_conv=0.0,
        kappa_gm=0.0, kappa_redi=0.0,
    )
    T = np.full((grid.nx, grid.ny, grid.nz), 20.0)
    S = np.full((grid.nx, grid.ny, grid.nz), 35.0)
    forcing = tuple(np.zeros((grid.nx, grid.ny)) for _ in range(3))
    T_atm_base = np.full((grid.nx, grid.ny), 20.0)

    _, init_state, _, _, _, step_dyn = make_solver_global(
        grid, physics, 60.0, forcing=forcing,
        T_atm=T_atm_base, lambda_bulk=1.0e-4,
        T_init=T, S_init=S, polar_cap_rows=0, polar_cap_taper=0,
        mode_split=False, dtype='float64', dynamic_forcing=True,
        return_params=True)

    state0 = init_state(T_init=jnp.array(T), S_init=jnp.array(S))
    tau_x = jnp.zeros((grid.nx, grid.ny))
    tau_y = jnp.zeros((grid.nx, grid.ny))
    q_heat = jnp.zeros((grid.nx, grid.ny))
    warm = step_dyn(state0, tau_x, tau_y, q_heat,
                    T_atm_3d=jnp.full((grid.nx, grid.ny, 1), 21.0))
    cold = step_dyn(state0, tau_x, tau_y, q_heat,
                    T_atm_3d=jnp.array(19.0))
    warm_sst = float(np.asarray(warm.T[0, 0, 0]))
    cold_sst = float(np.asarray(cold.T[0, 0, 0]))
    assert warm_sst > 20.0
    assert cold_sst < 20.0
    assert warm_sst > cold_sst
