"""Regression tests for the experimental flux-limited tracer transport.

The first bounded-flux prototype is intentionally small: it keeps the second-order
centered face value inside the min/max of the two donor cells. That is not a full
Zalesak multidimensional FCT limiter, but it should be conservative and must not
manufacture new extrema for a sharp passive tracer in a constant flow.
"""


from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from config import PhysicsConfig
from jax_solver_global import (
    _advection_scalar,
    _vertical_transport_iface,
    make_solver_global,
)
from tests.support.fd.advection import _synth_grid


def _make_params(grid, fct):
    nx, ny = grid.nx, grid.ny
    physics = replace(PhysicsConfig(), nu_h=100.0, nu_bi=0.0,
                      kappa_h=100.0, kappa_v=1e-5, kappa_conv=0.0)
    forcing = (np.zeros((nx, ny)), np.zeros((nx, ny)), np.zeros((nx, ny)))
    _, _, _, params, _ = make_solver_global(
        grid, physics, 60.0, forcing=forcing, lambda_bulk=0.0,
        sponge_days=0.0, sponge_cells=0, T_init=None, S_init=None,
        polar_cap_rows=0, polar_cap_taper=0, return_params=True,
        fct_adv=fct)
    return params


def test_params_default_preserves_centered_path():
    p = _make_params(_synth_grid(), fct=False)
    assert not p.fct_adv


def test_params_enable_fct_path():
    p = _make_params(_synth_grid(), fct=True)
    assert p.fct_adv


def test_fct_uniform_tracer_is_conserved_for_arbitrary_velocity():
    """Uniform T/S gives exactly zero tendency through horizontal + vertical flux."""
    grid = _synth_grid()
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    p = _make_params(grid, fct=True)

    lon = np.arange(nx)
    lat = np.arange(ny)
    u = 0.03 * np.cos(2.0 * np.pi * lon[:, None] / nx)[:, :, None] \
        + 0.01 * np.cos(np.pi * lat[None, :, None] / ny)
    v = 0.02 * np.sin(np.pi * lat[None, :, None] / ny)
    u = np.broadcast_to(u, (nx, ny, nz))
    v = np.broadcast_to(v, (nx, ny, nz))
    T = np.full((nx, ny, nz), 20.0)
    S = np.full((nx, ny, nz), 35.0)
    Fz = _vertical_transport_iface(jnp.asarray(u), jnp.asarray(v), p)

    aT = _advection_scalar(jnp.asarray(T), jnp.asarray(u), jnp.asarray(v), Fz, p)
    aS = _advection_scalar(jnp.asarray(S), jnp.asarray(u), jnp.asarray(v), Fz, p)
    np.testing.assert_allclose(np.asarray(aT), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(aS), 0.0, atol=1e-12)


def test_fct_bounded_flux_keeps_sharp_tracer_in_bounds():
    """For constant +x flow the bounded centered flux cannot overshoot."""
    grid = _synth_grid()
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    p = _make_params(grid, fct=True)
    dx_min = float(np.min(np.asarray(grid.dx_2d)))
    dt = 0.9 * dx_min / 1.0    # horizontal Courant number = 0.9

    T = np.full((nx, ny, nz), 10.0)
    T[nx // 2:, :, :] = 25.0
    u = np.ones((nx, ny, nz))
    v = np.zeros((nx, ny, nz))
    Fz = _vertical_transport_iface(jnp.asarray(u), jnp.asarray(v), p)

    aT = _advection_scalar(jnp.asarray(T), jnp.asarray(u), jnp.asarray(v), Fz, p)
    T_new = np.asarray(T + aT * dt)
    assert np.isfinite(T_new).all()
    assert T_new.min() >= 10.0 - 1e-12
    assert T_new.max() <= 25.0 + 1e-12
    # The front should still move east, not freeze.
    assert float(np.max(T_new[:nx // 2, :, :])) > 10.0 + 1e-8
    assert float(np.max(T_new[nx // 2:, :, :])) <= 25.0 + 1e-12


def test_fct_south_wall_does_not_read_north_wall():
    grid = _synth_grid()
    params = _make_params(grid, fct=True)
    shape = (grid.nx, grid.ny, grid.nz)
    velocity = jnp.zeros(shape).at[:, 1, :].set(1.)
    transport = jnp.zeros((grid.nx, grid.ny, grid.nz + 1))
    tracer = jnp.broadcast_to(10. + 2. * jnp.arange(grid.ny)[None, :, None], shape)
    first = _advection_scalar(tracer.at[:, -1, :].set(0.), jnp.zeros(shape),
                               velocity, transport, params)
    second = _advection_scalar(tracer.at[:, -1, :].set(30.), jnp.zeros(shape),
                                velocity, transport, params)
    np.testing.assert_array_equal(first[:, :2, :], second[:, :2, :])


def test_fct_wet_face_does_not_read_land_sentinel():
    grid = _synth_grid()
    params = _make_params(grid, fct=True)
    shape = (grid.nx, grid.ny, grid.nz)
    wet = jnp.ones(shape).at[0, :, :].set(0.)
    params = params._replace(wet_mask_z=wet, wet_mask=wet[:, :, 0])
    velocity = jnp.ones(shape).at[0, :, :].set(0.)
    transport = jnp.zeros((grid.nx, grid.ny, grid.nz + 1))
    tracer = jnp.full(shape, 20.).at[1, :, :].set(10.).at[2, :, :].set(12.)
    first = _advection_scalar(tracer.at[0, :, :].set(0.), velocity,
                               jnp.zeros(shape), transport, params)
    second = _advection_scalar(tracer.at[0, :, :].set(15.), velocity,
                                jnp.zeros(shape), transport, params)
    np.testing.assert_array_equal(first[1:3], second[1:3])
