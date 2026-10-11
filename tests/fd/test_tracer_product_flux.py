"""Independent transport symbols, physical inventory and strict restart controls."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.grid import all_wet_grid
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.io.restart import load_restart, make_restart_contract, save_restart
from zhenmode.model.solver.dynamics.transport import (
    _advection_scalar,
    _horizontal_tracer_fluxes,
    _layer_face_transports,
    _vertical_transport_iface,
)
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.model.solver.state import JaxStateG


def channel(nx=16, **changes):
    grid = all_wet_grid(nx=nx, ny=8, nz=4)
    grid = replace(grid, z=np.array([0., -5., -20., -100.]), dz=np.array([5., 15., 80.]),
                   depth=np.full((nx, 8), 100.), dx_2d=np.full((nx, 8), 100000 / nx),
                   dy=1000., cos_lat=np.ones(8), f=np.zeros((nx, 8)))
    physics = PhysicsConfig(**dict.fromkeys((
        'nu_h', 'nu_v', 'nu_bi', 'kappa_h', 'kappa_v', 'kappa_bi',
        'kappa_conv', 'kappa_gm', 'kappa_redi', 'r_bot', 'cd'), 0.))
    options = dict(return_params=True, column_geometry='nodal_dual_v1',
                   conservative_kv=True, localize_conv=True, polar_cap_rows=0,
                   polar_cap_taper=0, external_mode_scheme='symmetric',
                   meridional_boundary_scheme='closed_faces',
                   pressure_continuity_scheme='centered_fourth',
                   tracer_transport_scheme='product_fourth') | changes
    return grid, make_solver_global(grid, physics, 10., **options)


def mode(nx):
    phi = 2 * np.pi * (np.arange(nx) + .5) / nx
    T = jnp.broadcast_to(jnp.asarray(15 + .1 * np.cos(phi))[:, None, None], (nx, 8, 4))
    return phi, T, jnp.full_like(T, .05), jnp.zeros_like(T)


def test_uniform_advection_has_fourth_order_symbol_and_spatial_convergence():
    errors = []
    for nx in (16, 32, 64):
        _, (_, _, _, p, _) = channel(nx)
        phi, T, u, v = mode(nx)
        rhs = np.asarray(_advection_scalar(T, u, v, _vertical_transport_iface(u, v, p), p))[:, 0, 0]
        dx = 100000 / nx
        kd = np.sin(2 * np.pi / nx) * (4 - np.cos(2 * np.pi / nx)) / (3 * dx)
        expected = .005 * kd * np.sin(phi)
        np.testing.assert_allclose(rhs, expected, atol=1e-18)
        continuum = .005 * 2 * np.pi / 100000 * np.sin(phi)
        errors.append(np.linalg.norm(rhs - continuum) / np.linalg.norm(continuum))
    assert all(15 < a / b < 17 for a, b in zip(errors[:-1], errors[1:], strict=True)), errors


def test_variable_velocity_differentiates_the_product_not_separate_reconstructions():
    nx = 32
    _, (_, _, _, p, _) = channel(nx)
    phi, T, _, v = mode(nx)
    u = jnp.broadcast_to(jnp.asarray(.05 + .02 * np.sin(phi))[:, None, None], T.shape)
    fx, _ = _horizontal_tracer_fluxes(T, u, v, p, None)
    derivative = np.asarray((fx - jnp.roll(fx, 1, axis=0)) * p.inv_dx)[:, 0, 0]
    theta = 2 * np.pi / nx
    dx = 100000 / nx
    kd1 = np.sin(theta) * (4 - np.cos(theta)) / (3 * dx)
    kd2 = np.sin(2 * theta) * (4 - np.cos(2 * theta)) / (3 * dx)
    # u*T = .75 + .3 sin(phi) + .005 cos(phi) + .001 sin(2phi).
    expected = .3 * kd1 * np.cos(phi) - .005 * kd1 * np.sin(phi) + .001 * kd2 * np.cos(2 * phi)
    np.testing.assert_allclose(derivative, expected, atol=2e-19)
    old, _ = _horizontal_tracer_fluxes(T, u, v, p._replace(tracer_transport_scheme='centered_second'), None)
    old_derivative = np.asarray((old - jnp.roll(old, 1, axis=0)) * p.inv_dx)[:, 0, 0]
    assert np.max(abs(old_derivative - expected)) > 1e-10


def test_corrected_volume_faces_preserve_constant_tracer_and_boundary_inventory():
    _, (_, _, _, p, _) = channel(8)
    rng = np.random.default_rng(111026)
    u, v = (jnp.asarray(rng.normal(size=(8, 8, 4)) * .01) for _ in range(2))
    fx, fy = _layer_face_transports(u, v, p)
    fx = fx + jnp.asarray(rng.normal(size=fx.shape) * .001)
    fy = (fy + jnp.asarray(rng.normal(size=fy.shape) * .001)).at[:, -1].set(0.)
    actual = (fx, fy)
    Fz = _vertical_transport_iface(u, v, p, face_transport=actual)
    constant = jnp.full_like(u, 15.)
    tendency = _advection_scalar(constant, u, v, Fz, p, face_transport=actual)
    np.testing.assert_allclose(tendency, 0., atol=2e-18)
    tracer = jnp.asarray(rng.normal(size=u.shape) + 15)
    rate, top = _advection_scalar(tracer, u, v, Fz, p, return_boundary=True, face_transport=actual)
    inventory_rate = float(jnp.sum(rate * p.dz_node))
    assert abs(inventory_rate - float(jnp.sum(top))) < 2e-16
    _, normal = _horizontal_tracer_fluxes(tracer, u, v, p, actual)
    assert np.max(abs(np.asarray(normal)[:, -1])) == 0


def test_full_core_matches_independent_rk2_transport_solution_and_restart(tmp_path):
    nx = 16
    grid, (advance, initialize, _, p, _) = channel(nx)
    phi, T, u, v = mode(nx)
    # Density-neutral tracer leaves the current unforced; all native equations run.
    initial = initialize()._replace(T=T, S=35 + (.205 / .779) * (T - 15), u=u, v=v)
    kd = np.sin(2 * np.pi / nx) * (4 - np.cos(2 * np.pi / nx)) / (3 * (100000 / nx))
    frequency = .05 * kd
    count = 16
    amplification = (1 - 1j * frequency * p.dt - .5 * (frequency * p.dt)**2)**count
    expected = 15 + .1 * np.real(np.exp(1j * phi) * amplification)
    state = initial
    for _ in range(count):
        state = advance(state)
    np.testing.assert_allclose(np.asarray(state.T)[:, 0, 0], expected, atol=2e-13)
    assert abs(float(jnp.mean(state.T)) - 15) < 5e-14
    assert np.max(abs(np.asarray(state.u) - .05)) < 1e-13
    contract = make_restart_contract(grid, p, dtype='float64', forcing=None,
                                     controls={}, code_paths={}, execution={'backend': 'cpu'})
    split = advance(initial)
    path = tmp_path / 'state.npz'
    save_restart(path, split, contract, step=1, counters={}, cumulative={}, history={})
    saved = load_restart(path, contract)
    resumed = advance(JaxStateG(**{name: jnp.asarray(value) for name, value in saved.state.items()}))
    expected_state = advance(split)
    for name in initial._fields:
        np.testing.assert_array_equal(getattr(resumed, name), getattr(expected_state, name))
    old = dict(contract, effective_params=dict(contract['effective_params'], tracer_transport_scheme='centered_second'))
    with pytest.raises(ValueError, match='contract'):
        load_restart(path, old)


@pytest.mark.parametrize('changes', [dict(monotone_adv=True), dict(fct_adv=True),
                                    dict(pressure_continuity_scheme='centered_second'),
                                    dict(tracer_transport_scheme='unknown')])
def test_unqualified_combinations_are_rejected(changes):
    with pytest.raises(ValueError, match='product_fourth|tracer_transport'):
        channel(8, **changes)
