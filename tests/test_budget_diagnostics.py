"""Budget diagnostics: cheap global invariants for the run driver."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from config import PhysicsConfig
from _helpers import all_wet_grid
from diagnostics import compute_budget_diagnostics, node_thickness
from jax_solver_global import make_solver_global


def test_budget_diagnostics_on_uniform_state():
    grid = all_wet_grid(nx=16, ny=8, nz=4)
    T = np.full((grid.nx, grid.ny, grid.nz), 20.0)
    S = np.full((grid.nx, grid.ny, grid.nz), 35.0)
    state = type("State", (), {"T": T, "S": S})()
    d = compute_budget_diagnostics(state, grid)

    assert d.total_volume_m3 > 0.0
    assert d.mean_T == pytest.approx(20.0)
    assert d.mean_S == pytest.approx(35.0)
    assert d.mean_T_top == pytest.approx(20.0)
    assert d.mean_S_top == pytest.approx(35.0)
    expected_mean_depth = float(node_thickness(grid.z).sum())
    assert d.mean_depth_m == pytest.approx(expected_mean_depth)


def test_zero_forcing_run_conserves_global_heat_and_salt():
    """One zero-forcing step must not manufacture heat or salt."""
    grid = all_wet_grid(nx=16, ny=8, nz=4)
    physics = replace(
        PhysicsConfig(),
        nu_h=0.0, nu_bi=0.0, nu_v=0.0,
        kappa_h=0.0, kappa_v=0.0, kappa_conv=0.0,
        kappa_gm=0.0, kappa_redi=0.0,
    )
    T = np.full((grid.nx, grid.ny, grid.nz), 20.0)
    S = np.full((grid.nx, grid.ny, grid.nz), 35.0)
    forcing = tuple(np.zeros((grid.nx, grid.ny)) for _ in range(3))

    step, init_state, _ = make_solver_global(
        grid, physics, 60.0, forcing=forcing, lambda_bulk=0.0,
        T_init=T, S_init=S, polar_cap_rows=0, polar_cap_taper=0,
        mode_split=False, dtype='float64')

    state0 = init_state(T_init=jnp.array(T), S_init=jnp.array(S))
    state1 = step(state0)
    d0 = compute_budget_diagnostics(state0, grid)
    d1 = compute_budget_diagnostics(state1, grid)

    assert d1.total_volume_m3 == d0.total_volume_m3
    assert d1.heat_content_J == d0.heat_content_J
    assert d1.salt_content_kg == d0.salt_content_kg
