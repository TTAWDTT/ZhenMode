"""Polynomial pressure integrals and mean-force ownership, separate from the scorer."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.grid import all_wet_grid
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.io.restart import load_restart, make_restart_contract, save_restart
from zhenmode.model.solver.dynamics.pressure import (
    _compute_bt_rho_pgf,
    _compute_hydrostatic_pressure,
)
from zhenmode.model.solver.dynamics.tendencies import _compute_momentum_residual
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.model.solver.state import JaxStateG


def make_case(scheme='linear_density', **changes):
    n = 8
    grid = all_wet_grid(nx=n, ny=n, nz=4)
    grid = replace(grid, z=np.array([0., -5., -20., -50.]), dz=np.array([5., 15., 30.]),
                   depth=np.full((n, n), 50.), dx_2d=np.full((n, n), 1000.), dy=1000.,
                   cos_lat=np.ones(n), f=np.full((n, n), 1e-4))
    physics = PhysicsConfig(**dict.fromkeys((
        'nu_h', 'nu_v', 'nu_bi', 'kappa_h', 'kappa_v', 'kappa_bi',
        'kappa_conv', 'kappa_gm', 'kappa_redi', 'r_bot', 'cd'), 0.))
    options = dict(return_params=True, column_geometry='nodal_dual_v1',
                   conservative_kv=True, localize_conv=True, polar_cap_rows=0,
                   polar_cap_taper=0, external_mode_scheme='symmetric',
                   meridional_boundary_scheme='closed_faces',
                   pressure_continuity_scheme='centered_fourth',
                   tracer_transport_scheme='product_fourth',
                   hydrostatic_surface_scheme=scheme) | changes
    return grid, make_solver_global(grid, physics, 1., **options)


@pytest.mark.parametrize('eta', [.3, -.4, 0.])
@pytest.mark.parametrize('a,b', [(0., 0.), (.41, 0.), (-.17, .009)])
def test_polynomial_hydrostatic_integral_including_signed_surface_interval(eta, a, b):
    grid, (_, initialize, _, p, _) = make_case()
    z = grid.z
    # Frozen linear-EOS constants independently invert the manufactured rho'.
    temperature = 15 - (a + b * z) / (1025 * .0002)
    state = initialize()._replace(T=jnp.broadcast_to(jnp.asarray(temperature), (8, 8, 4)),
                                  eta=jnp.full((8, 8), eta))
    expected = 9.81 * (1025 * eta + a * (eta - z) + .5 * b * (eta**2 - z**2))
    actual = np.asarray(_compute_hydrostatic_pressure(state, p))
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=0, atol=2e-11)
    old = np.asarray(_compute_hydrostatic_pressure(state, p._replace(hydrostatic_surface_scheme='reference_plane')))
    if eta == 0 or (a == 0 and b == 0):
        np.testing.assert_array_equal(actual, old)
    else:
        # Omitted and wrong-sign loads both fail the same fixed pressure test.
        assert np.max(abs(old - expected)) > 1e-3
        wrong_sign = old - (actual - old)
        assert np.max(abs(wrong_sign - expected)) > 1e-3


def test_analytic_thermal_state_has_constant_bottom_pressure_without_changing_initial_eta():
    grid, (_, initialize, _, p, _) = make_case()
    front = .5 * np.sin(2 * np.pi * (np.arange(8) + .5) / 8)
    alpha, gamma, depth = .0002, .02, 50.
    q = 1 - alpha * front
    eta = 2 * alpha * depth * front / (q + np.sqrt(q * q - 2 * alpha**2 * gamma * depth * front))
    temperature = 15 + gamma * grid.z[None, None, :] + front[None, :, None]
    state = initialize()._replace(T=jnp.asarray(np.broadcast_to(temperature, (8, 8, 4))),
                                  eta=jnp.asarray(np.broadcast_to(eta, (8, 8))))
    bottom = np.asarray(_compute_hydrostatic_pressure(state, p))[..., -1]
    assert np.ptp(bottom) < 1e-10
    old = np.asarray(_compute_hydrostatic_pressure(state, p._replace(hydrostatic_surface_scheme='reference_plane')))[..., -1]
    assert np.ptp(old) > 1e-3


def test_mean_load_retains_actual_eta_and_is_removed_once_from_nonlinear_residual():
    _, (advance, initialize, _, p, _) = make_case()
    phi = 2 * np.pi * (np.arange(8) + .5) / 8
    eta = jnp.broadcast_to(jnp.asarray(1e-4 * np.cos(phi))[:, None], (8, 8))
    state = initialize()._replace(eta=eta, T=jnp.full((8, 8, 4), 10.))
    old = p._replace(hydrostatic_surface_scheme='reference_plane')
    new_force = _compute_bt_rho_pgf(state, p)
    old_force = _compute_bt_rho_pgf(state, old)
    kd = np.sin(np.pi / 4) * (4 - np.cos(np.pi / 4)) / 3000
    expected = 9.81 * .001 * 1e-4 * kd * np.sin(phi)
    np.testing.assert_allclose(np.asarray(new_force[0] - old_force[0]),
                               np.broadcast_to(expected[:, None], (8, 8)), rtol=0, atol=5e-14)
    assert np.max(abs(expected)) > 1e-10
    zero_eta = _compute_bt_rho_pgf(state._replace(eta=jnp.zeros_like(eta)), p)[0]
    assert np.max(abs(np.asarray(zero_eta) - expected[:, None])) > 1e-10
    new_residual = _compute_momentum_residual(state, p)
    old_residual = _compute_momentum_residual(state, old)
    for a, b in zip(new_residual, old_residual, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=5e-14)
    # Complete L/N/L owns this extra mean force once, not in both L and N.
    from zhenmode.model.solver.timestepping.step import _step_impl

    updated = advance(state)
    previous = _step_impl(state, old)
    delta = np.asarray(updated.u - previous.u)
    np.testing.assert_allclose(delta, np.broadcast_to(expected[:, None, None], delta.shape),
                               rtol=0, atol=5e-14)
    assert np.max(abs(delta - 2 * expected[:, None, None])) > 1e-10


def test_actual_restart_reproduces_all_fields_and_rejects_the_other_surface_scheme(tmp_path):
    grid, (advance, initialize, _, p, _) = make_case()
    state = initialize()._replace(T=jnp.full((8, 8, 4), 14.),
        eta=jnp.broadcast_to(jnp.asarray(1e-4 * np.cos(2 * np.pi * np.arange(8) / 8))[:, None], (8, 8)))
    first = advance(state)
    expected = advance(first)
    contract = make_restart_contract(grid, p, dtype='float64', forcing=None,
                                     controls={}, code_paths={}, execution={'backend': 'cpu'})
    path = tmp_path / 'restart.npz'
    save_restart(path, first, contract, step=1, counters={}, cumulative={}, history={})
    saved = load_restart(path, contract)
    resumed = advance(JaxStateG(**{name: jnp.asarray(value) for name, value in saved.state.items()}))
    for name in state._fields:
        np.testing.assert_array_equal(getattr(expected, name), getattr(resumed, name))
    other = dict(contract, effective_params=dict(contract['effective_params'], hydrostatic_surface_scheme='reference_plane'))
    with pytest.raises(ValueError, match='contract'):
        load_restart(path, other)


def test_unknown_and_unqualified_surface_schemes_are_rejected():
    with pytest.raises(ValueError, match='hydrostatic_surface'):
        make_case('unknown')
    with pytest.raises(ValueError, match='linear_density'):
        make_case(pressure_continuity_scheme='centered_second')
