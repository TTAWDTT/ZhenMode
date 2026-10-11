"""Held-force inertial solutions and discrete geostrophic stage balance."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.grid import all_wet_grid
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.io.restart import load_restart, make_restart_contract, save_restart
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.model.solver.state import JaxStateG
from zhenmode.model.solver.timestepping.step import _linear_half_step, _rotate_baroclinic_shear


def make_case(scheme='joint_shear', f=1e-4, **changes):
    n = 16
    z = np.array([0., -5., -20., -50.])
    grid = replace(all_wet_grid(nx=n, ny=n, nz=4), z=z, dz=-np.diff(z),
                   depth=np.full((n, n), 50.), dx_2d=np.full((n, n), 1000.),
                   dy=1000., cos_lat=np.ones(n), f=np.full((n, n), f))
    physics = PhysicsConfig(**dict.fromkeys((
        'nu_h', 'nu_v', 'nu_bi', 'kappa_h', 'kappa_v', 'kappa_bi',
        'kappa_conv', 'kappa_gm', 'kappa_redi', 'r_bot', 'cd'), 0.))
    options = dict(return_params=True, column_geometry='nodal_dual_v1',
                   conservative_kv=True, localize_conv=True, polar_cap_rows=0,
                   polar_cap_taper=0, external_mode_scheme='symmetric',
                   meridional_boundary_scheme='closed_faces',
                   pressure_continuity_scheme='centered_fourth',
                   tracer_transport_scheme='product_fourth',
                   hydrostatic_surface_scheme='linear_density',
                   density_rotation_scheme=scheme) | changes
    return grid, make_solver_global(grid, physics, 50., **options)


@pytest.mark.parametrize('f', [0., 1e-4, -2e-4])
def test_forced_rotation_matches_independent_point_solution_and_keeps_mean(f):
    _, (_, initialize, _, p, _) = make_case(f=f)
    shape = np.array([1., .5, -.5, -1.])
    shape -= np.sum(shape * np.asarray(p.dz_node)) / 50
    u = jnp.broadcast_to(jnp.asarray(.01 + .001 * shape), initialize().u.shape)
    v = jnp.broadcast_to(jnp.asarray(-.02 + .002 * shape), u.shape)
    fx = jnp.broadcast_to(jnp.asarray(1e-8 * shape), u.shape)
    fy = -2 * fx
    duration = 25.
    actual_u, actual_v = _rotate_baroclinic_shear(u, v, p, duration, forcing=(fx, fy))
    theta = f * duration
    a = duration if f == 0 else np.sin(theta) / f
    b = 0. if f == 0 else (1 - np.cos(theta)) / f
    expected_u = .01 + .001 * shape * np.cos(theta) + .002 * shape * np.sin(theta) + (a - 2 * b) * 1e-8 * shape
    expected_v = -.02 - .001 * shape * np.sin(theta) + .002 * shape * np.cos(theta) + (-b - 2 * a) * 1e-8 * shape
    np.testing.assert_allclose(actual_u, np.broadcast_to(expected_u, u.shape), rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual_v, np.broadcast_to(expected_v, v.shape), rtol=0, atol=1e-14)
    np.testing.assert_allclose(jnp.sum(actual_u * p.dz_norm, axis=-1), .01, atol=3e-15)
    np.testing.assert_allclose(jnp.sum(actual_v * p.dz_norm, axis=-1), -.02, atol=3e-15)
    unforced = _rotate_baroclinic_shear(u, v, p, duration)
    zero_force = _rotate_baroclinic_shear(u, v, p, duration, forcing=(jnp.zeros_like(fx), jnp.zeros_like(fy)))
    for x, y in zip(unforced, zero_force, strict=True):
        np.testing.assert_array_equal(x, y)
    # Removing the nonzero force fails the same exact-solution comparison.
    assert np.max(abs(np.asarray(unforced[0]) - expected_u)) > 1e-8


def equilibrium(grid, initialize):
    phi = np.pi * (np.arange(16) + .5) / 16
    kd = np.sin(np.pi / 16) * (4 - np.cos(np.pi / 16)) / (3 * grid.dy)
    T = 15 + .5 * np.cos(phi)[None, :, None] + np.zeros((16, 16, 4))
    u = 9.81 * .0002 * .5 * kd * np.sin(phi)[None, :, None] * grid.z[None, None, :] / 1e-4
    return initialize()._replace(T=jnp.asarray(T), u=jnp.asarray(np.broadcast_to(u, (16, 16, 4))))


def test_discrete_geostrophic_equilibrium_survives_each_stage_and_full_step():
    grid, (advance, initialize, _, p, _) = make_case()
    state = equilibrium(grid, initialize)
    stage = _linear_half_step(state, p, 25.)
    assert np.max(abs(np.asarray(stage.v))) < 1e-12
    result = advance(state)
    assert np.max(abs(np.asarray(result.v))) < 1e-12
    np.testing.assert_allclose(result.T, state.T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.u, state.u, rtol=0, atol=1e-12)
    # Old source partition creates a large transient despite zero initial v.
    old_stage = _linear_half_step(state, p._replace(density_rotation_scheme='split_sources'), 25.)
    assert np.max(abs(np.asarray(old_stage.v))) > 1e-6


def test_density_force_is_removed_from_nonlinear_residual_once():
    from zhenmode.model.solver.dynamics.tendencies import _compute_momentum_residual

    grid, (_, initialize, _, p, _) = make_case()
    state = equilibrium(grid, initialize)
    residual = _compute_momentum_residual(state, p)
    assert max(np.max(abs(np.asarray(value))) for value in residual) < 5e-14
    old = _compute_momentum_residual(state, p._replace(density_rotation_scheme='split_sources'))
    assert np.max(abs(np.asarray(old[1]))) > 1e-6


def test_restart_keeps_every_field_and_rejects_other_source_partition(tmp_path):
    grid, (advance, initialize, _, p, _) = make_case()
    state = equilibrium(grid, initialize)._replace(eta=jnp.full((16, 16), 1e-4))
    first = advance(state)
    expected = advance(first)
    contract = make_restart_contract(grid, p, dtype='float64', forcing=None,
                                     controls={}, code_paths={}, execution={'backend': 'cpu'})
    path = tmp_path / 'state.npz'
    save_restart(path, first, contract, step=1, counters={}, cumulative={}, history={})
    saved = load_restart(path, contract)
    result = advance(JaxStateG(**{name: jnp.asarray(value) for name, value in saved.state.items()}))
    for name in state._fields:
        np.testing.assert_array_equal(getattr(result, name), getattr(expected, name))
    other = dict(contract, effective_params=dict(contract['effective_params'], density_rotation_scheme='split_sources'))
    with pytest.raises(ValueError, match='contract'):
        load_restart(path, other)


@pytest.mark.parametrize('change', [dict(density_rotation_scheme='unknown'),
    dict(hydrostatic_surface_scheme='reference_plane'), dict(sponge_days=1.), dict(dtype='float32')])
def test_unqualified_density_rotation_combinations_are_rejected(change):
    with pytest.raises(ValueError, match='density_rotation|joint_shear'):
        make_case(**change)
