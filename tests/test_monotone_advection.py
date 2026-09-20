"""Regression tests for the monotone tracer-advection switch.

The historical horizontal scalar flux is centered and therefore exactly
conservative but not monotone.  The new ``monotone_adv`` switch uses
first-order donor-cell horizontal face values; the vertical flux was already
donor-cell.  These tests check the two invariants that matter for a tracer
scheme: a uniform tracer cannot be created or destroyed, and a sharp passive
scalar cannot overshoot its initial extrema for one explicit donor-cell step.

Run: python -m pytest tests/test_monotone_advection.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp
import numpy as np
from dataclasses import replace

from config import PhysicsConfig
from grid import GlobalOceanGrid
from jax_solver_global import (
    _advection_scalar,
    _vertical_transport_iface,
    make_solver_global,
)


def _synth_grid():
    """Small, fully wet periodic grid suitable for CPU operator tests."""
    nx, ny, nz = 24, 8, 6
    dlon = 360.0 / nx
    lon = np.arange(nx) * dlon
    lat = np.linspace(-30.0, 30.0, ny)
    cos_lat = np.cos(np.radians(lat))
    dx_2d = np.broadcast_to(
        6.371e6 * cos_lat[None, :] * np.radians(dlon), (nx, ny)).copy()
    dy = 6.371e6 * np.radians(abs(lat[1] - lat[0]))
    f = np.broadcast_to(2.0 * 7.2921e-5 * np.sin(np.radians(lat)),
                        (nx, ny)).copy()
    z = -np.linspace(50.0, 4000.0, nz)
    dz = -np.diff(z)
    return GlobalOceanGrid(
        lon=lon, lat=lat, dx_2d=dx_2d, dy=float(dy), cos_lat=cos_lat, f=f,
        z=z, dz=dz, nz=nz,
        depth=np.full((nx, ny), 4000.0),
        wet_mask=np.ones((nx, ny)),
        ocean_mask=np.ones((nx, ny), dtype=bool),
        land_mask=np.zeros((nx, ny), dtype=bool),
        wet_mask_3d=np.ones((nx, ny, nz)),
        nx=nx, ny=ny)


def _make_params(grid, monotone):
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    physics = replace(PhysicsConfig(), nu_h=100.0, nu_bi=0.0,
                      kappa_h=100.0, kappa_v=1e-5, kappa_conv=0.0)
    forcing = (np.zeros((nx, ny)), np.zeros((nx, ny)), np.zeros((nx, ny)))
    _, _, _, params, _ = make_solver_global(
        grid, physics, 60.0, forcing=forcing, lambda_bulk=0.0,
        sponge_days=0.0, sponge_cells=0, T_init=None, S_init=None,
        polar_cap_rows=0, polar_cap_taper=0, return_params=True,
        monotone_adv=monotone)
    return params


def test_params_default_preserves_centered_path():
    p = _make_params(_synth_grid(), monotone=False)
    assert not p.monotone_adv


def test_params_enable_monotone_path():
    p = _make_params(_synth_grid(), monotone=True)
    assert p.monotone_adv


def test_uniform_tracer_is_conserved_for_arbitrary_velocity():
    """Uniform T/S gives exactly zero tendency through horizontal + vertical flux."""
    grid = _synth_grid()
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    p = _make_params(grid, monotone=True)

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


def test_donor_horizontal_flux_keeps_sharp_tracer_in_bounds():
    """For constant +x flow the donor update is a convex average if CFL<1."""
    grid = _synth_grid()
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    p = _make_params(grid, monotone=True)
    dx_min = float(np.min(np.asarray(grid.dx_2d)))
    dt = 0.9 * dx_min / 1.0    # horizontal Courant number = 0.9

    # A cold/west to warm/east discontinuity; the donor update must not
    # manufacture values outside the initial range.
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
    # The discontinuity should actually move east, not just remain frozen.
    assert float(np.max(T_new[:nx // 2, :, :])) > 10.0 + 1e-8
    assert float(np.max(T_new[nx // 2:, :, :])) <= 25.0 + 1e-12
