"""Discrete surface-energy and local-diffusion budget regressions."""
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from _helpers import all_wet_grid

from config import C_P, RHO_0, PhysicsConfig
from jax_solver_global import (
    JaxStateG,
    _compute_tracer_tendency,
    _dynamic_ice_closure,
    make_solver_global,
)


def _setup(depth=None, heat=100., ice=False, coastal_mask=None):
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    grid = replace(grid, z=np.array([0., -5., -20., -50.]), dz=np.array([5., 15., 30.]))
    physics = replace(PhysicsConfig(), nu_h=0., nu_bi=0., nu_v=0., kappa_h=0.,
                      kappa_v=0., kappa_conv=0., kappa_gm=0., kappa_redi=0.)
    forcing = (np.zeros((8, 8)), np.zeros((8, 8)), np.full((8, 8), heat))
    step, _, _, params, _ = make_solver_global(
        grid, physics, 3600., forcing=forcing, lambda_bulk=0.,
        mixed_layer_depth_m=depth, dynamic_ice=ice, polar_cap_rows=0, polar_cap_taper=0,
        coastal_kappa_h_mask=coastal_mask, coastal_kappa_h=100., return_params=True)
    shape = (8, 8, 4)
    state = JaxStateG(jnp.zeros(shape), jnp.zeros(shape), jnp.full(shape, -1.8),
                      jnp.full(shape, 35.), jnp.zeros((8, 8)), jnp.zeros((8, 8)))
    return grid, params, state, step


@pytest.mark.parametrize("depth", [None, 20., 50., 5000.])
def test_surface_heat_input_matches_column_budget(depth):
    _, params, state, _ = _setup(depth=depth)
    tendency, _ = _compute_tracer_tendency(state, params)
    flux = np.sum(np.asarray(tendency) * np.asarray(params.dz_node), axis=-1) * RHO_0 * C_P
    np.testing.assert_allclose(flux, 100., rtol=1e-12)
    if depth is not None and depth > 5.:
        assert float(tendency[0, 0, 1]) > 0.


def test_local_horizontal_diffusion_conserves_heat():
    mask = np.zeros((8, 8))
    mask[1, :] = 1.
    grid, params, state, _ = _setup(heat=0., coastal_mask=mask)
    state = state._replace(T=jnp.full((8, 8, 4), 10.).at[2, :, :].set(20.))
    tendency, _ = _compute_tracer_tendency(state, params)
    weights = grid.dx_2d[:, :, None] * grid.dy * np.asarray(params.dz_node)
    total = float(np.sum(np.asarray(tendency) * weights))
    absolute = float(np.sum(np.abs(np.asarray(tendency)) * weights))
    assert abs(total) <= 1e-12 * absolute
    assert float(tendency[1, 3, 0]) > 0.
    assert float(tendency[2, 3, 0]) < 0.


@pytest.mark.parametrize("initial_ice,heat", [(0., -100.), (1e-6, 100.), (1., -100.), (1., 100.)])
def test_ice_closure_conserves_water_ice_enthalpy(initial_ice, heat):
    _, params, state, _ = _setup(depth=20., heat=heat, ice=True)
    state = state._replace(ice=jnp.full((8, 8), initial_ice))
    result = _dynamic_ice_closure(state, params)
    water_change = (RHO_0 * C_P * np.sum(
        np.asarray(result.T - state.T) * np.asarray(params.dz_node), axis=-1))
    ice_change = -917. * 3.34e5 * np.asarray(result.ice - state.ice)
    expected = heat * params.dt / (1. + initial_ice / params.ice_insulation_scale_m)
    np.testing.assert_allclose(water_change + ice_change, expected, rtol=1e-10, atol=1e-6)


def test_complete_melt_salt_flux_is_bounded_by_available_ice():
    _, params, state, _ = _setup(depth=20., ice=True)
    initial_ice = 1e-6
    state = state._replace(ice=jnp.full((8, 8), initial_ice))
    result = _dynamic_ice_closure(state, params)
    salt_change = RHO_0 * np.sum(
        np.asarray(result.S - state.S) * np.asarray(params.dz_node), axis=-1)
    np.testing.assert_allclose(salt_change, -917. * 30. * initial_ice, rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(result.ice, 0.)
    assert float(result.T[0, 0, 0]) > -1.8


def test_full_step_applies_surface_heat_only_once_with_ice():
    _, params, state, step = _setup(heat=-100., ice=True)
    result = step(state)
    expected = 100. * params.dt / (917. * 3.34e5)
    np.testing.assert_allclose(result.ice, expected, rtol=1e-10)


def test_ice_checkpoint_restart_matches_continuous_steps(tmp_path):
    from run_long_integration_global import _load_checkpoint_state, _save_checkpoint

    grid, _, state, step = _setup(heat=-100., ice=True)
    first = step(state)
    path = tmp_path / "checkpoint.npz"
    _save_checkpoint(path, first, grid, 1, 0)
    with np.load(path) as saved:
        restart = _load_checkpoint_state(saved, grid, first.T.dtype, dynamic_ice=True)
    continuous = step(first)
    resumed = step(restart)
    for name in first._fields:
        np.testing.assert_array_equal(getattr(continuous, name), getattr(resumed, name))


def test_spatial_mixed_layer_conserves_bulk_heat_over_shallow_columns():
    _, params, state, _ = _setup(depth=20., heat=0.)
    depths = jnp.full((8, 8), 20.).at[4:, :].set(50.)
    wet = params.wet_mask_z.at[0, :, 1:].set(0.).at[1, :, :].set(0.)
    params = params._replace(
        mixed_layer_depth_2d=depths, wet_mask_z=wet, wet_mask=wet[:, :, 0],
        mixed_layer_mask_2d=jnp.ones((8, 8)).at[2, :].set(0.),
        lambda_bulk=80., T_atm_3d=jnp.full((8, 8, 1), 0.))
    tendency, _ = _compute_tracer_tendency(state, params)
    flux = RHO_0 * C_P * np.sum(np.asarray(tendency) * np.asarray(params.dz_node), axis=-1)
    np.testing.assert_allclose(flux, 80. * 1.8 * np.asarray(params.wet_mask), rtol=1e-12)
    np.testing.assert_array_equal(tendency[1], 0.)


def test_local_diffusion_budget_with_meridional_gradients_and_land():
    grid, params, state, _ = _setup(heat=0., coastal_mask=np.ones((8, 8)))
    random = np.random.default_rng(123)
    wet = params.wet_mask_z.at[0, 2:5, :].set(0.)
    params = params._replace(wet_mask_z=wet, wet_mask=wet[:, :, 0], kappa_h=50.,
                            coastal_kappa_h_2d=jnp.asarray(random.uniform(1., 100., (8, 8))))
    state = state._replace(T=jnp.asarray(random.uniform(5., 20., (8, 8, 4))))
    tendency, _ = _compute_tracer_tendency(state, params)
    weights = grid.dx_2d[:, :, None] * grid.dy * np.asarray(params.dz_node)
    total = float(np.sum(np.asarray(tendency) * weights))
    absolute = float(np.sum(np.abs(np.asarray(tendency)) * weights))
    assert abs(total) <= 1e-12 * absolute


def test_surface_operator_is_differentiable_away_from_phase_boundary():
    import jax

    _, params, state, _ = _setup(depth=20., ice=True)
    state = state._replace(T=jnp.full((8, 8, 4), 5.))

    def temperature_after_flux(flux):
        updated = _dynamic_ice_closure(state, params._replace(Q_heat_2d=jnp.full((8, 8), flux)))
        return updated.T[0, 0, 0]

    derivative = jax.grad(temperature_after_flux)(100.)
    expected = params.dt / (RHO_0 * C_P * 20.)
    assert float(derivative) == pytest.approx(expected)
