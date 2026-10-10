"""Analytic modes, random adjoint controls and transport/restart consistency."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.grid import all_wet_grid
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.io.restart import load_restart, make_restart_contract, save_restart
from zhenmode.model.solver.dynamics.transport import (
    _advection_scalar,
    _column_divergence,
    _face_transport_divergence,
    _layer_face_transports,
    _vertical_transport_iface,
)
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.model.solver.numerics.horizontal import _gradient_conservative_3d
from zhenmode.model.solver.state import JaxStateG


def make_channel(n=16, **changes):
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
                   pressure_continuity_scheme='centered_fourth') | changes
    return grid, make_solver_global(grid, physics, 1., **options)


@pytest.mark.parametrize('n', [8, 16, 32])
def test_discrete_channel_symbol_includes_wall_centres(n):
    _, (_, _, _, p, _) = make_channel(n)
    phi = np.pi * (np.arange(n) + .5) / n
    scalar = jnp.broadcast_to(jnp.cos(phi)[None, :, None], (n, n, 4))
    normal = jnp.broadcast_to(jnp.sin(phi)[None, :, None], (n, n, 4))
    kd = np.sin(np.pi / n) * (4 - np.cos(np.pi / n)) / (3 * p.dy)
    _, gradient = _gradient_conservative_3d(scalar, p)
    np.testing.assert_allclose(gradient, -kd * np.asarray(normal), atol=2e-18)
    divergence = _column_divergence(jnp.zeros_like(normal), normal, p)
    np.testing.assert_allclose(divergence, 50 * kd * np.asarray(scalar[..., 0]), atol=2e-17)
    faces = _layer_face_transports(jnp.zeros_like(normal), normal, p)
    assert np.max(abs(np.asarray(faces[1])[:, -1])) == 0


def test_smooth_wall_mode_has_fourth_order_spatial_derivative_not_whole_model_order():
    errors = []
    for n in (8, 16, 32):
        _, (_, _, _, p, _) = make_channel(n)
        phi = np.pi * (np.arange(n) + .5) / n
        field = jnp.broadcast_to(jnp.cos(phi)[None, :, None], (n, n, 4))
        _, gradient = _gradient_conservative_3d(field, p)
        expected = -np.pi / (n * p.dy) * np.sin(phi)
        errors.append(np.linalg.norm(np.asarray(gradient)[0, :, 0] - expected) / np.linalg.norm(expected))
    assert all(15 < a / b < 17 for a, b in zip(errors[:-1], errors[1:], strict=True)), errors


def test_pair_preserves_mass_work_identity_and_actual_constant_tracer():
    _, (_, _, _, p, _) = make_channel(8)
    rng = np.random.default_rng(10102026)
    scalar, u, v = (jnp.asarray(rng.normal(size=(8, 8, 4))) for _ in range(3))
    gx, gy = _gradient_conservative_3d(scalar, p)
    fx, fy = _layer_face_transports(u, v, p)
    divergence = _face_transport_divergence(fx, fy, p)
    work = float(jnp.sum(scalar * divergence + (gx * u + gy * v) * p.dz_node))
    assert abs(work) < 2e-15
    assert abs(float(jnp.sum(divergence))) < 2e-15
    mismatch = _column_divergence(u, v, p._replace(pressure_continuity_scheme='centered_second'))
    # Separate uniform-in-depth pressure makes this a genuine mismatched pair.
    eta = jnp.broadcast_to(scalar[..., :1], scalar.shape)
    ex, ey = _gradient_conservative_3d(eta, p)
    wrong_work = float(jnp.sum(eta[..., 0] * mismatch) + jnp.sum((ex * u + ey * v) * p.dz_node))
    assert abs(wrong_work) > 1e-6
    constant = jnp.ones_like(u) * 15
    Fz = _vertical_transport_iface(u, v, p)
    tendency = _advection_scalar(constant, u, v, Fz, p)
    rate = 15 * (float(jnp.max(jnp.abs(u))) / 1000 + float(jnp.max(jnp.abs(v))) / p.dy)
    np.testing.assert_allclose(tendency, 0., atol=64 * np.finfo(float).eps * rate)
    old_faces = _layer_face_transports(u, v, p._replace(pressure_continuity_scheme='centered_second'))
    wrong_tracer = _advection_scalar(constant, u, v, Fz, p, face_transport=old_faces)
    assert np.max(abs(np.asarray(wrong_tracer))) > 1e-4


def test_periodic_pressure_symbol_and_unforced_constant_field():
    _, (_, _, _, p, _) = make_channel(16)
    phi = 2 * np.pi * (np.arange(16) + .5) / 16
    field = jnp.broadcast_to(jnp.cos(phi)[:, None, None], (16, 16, 4))
    gx, gy = _gradient_conservative_3d(field, p)
    kd = np.sin(2 * np.pi / 16) * (4 - np.cos(2 * np.pi / 16)) / 3000
    np.testing.assert_allclose(np.asarray(gx)[:, 0, 0], -kd * np.sin(phi), atol=2e-18)
    np.testing.assert_allclose(gy, 0., atol=2e-18)
    assert all(np.max(abs(np.asarray(g))) == 0 for g in _gradient_conservative_3d(jnp.ones_like(field), p))


def test_gradient_annihilates_directional_constants_with_large_pressure_offset():
    _, (_, _, _, p, _) = make_channel(8)
    rng = np.random.default_rng(610)
    x_only = jnp.asarray(np.broadcast_to(1e6 + rng.normal(size=(8, 1, 4)), (8, 8, 4)))
    y_only = jnp.asarray(np.broadcast_to(1e6 + rng.normal(size=(1, 8, 4)), (8, 8, 4)))
    _, gy = _gradient_conservative_3d(x_only, p)
    gx, _ = _gradient_conservative_3d(y_only, p)
    assert np.max(abs(np.asarray(gy))) == 0
    assert np.max(abs(np.asarray(gx))) == 0


def test_full_step_restart_and_wrong_spatial_pair_rejection(tmp_path):
    grid, (advance, initialize, _, p, _) = make_channel(8)
    phi = np.pi * (np.arange(8) + .5) / 8
    initial = initialize()._replace(eta=jnp.broadcast_to(jnp.cos(phi)[None, :] * 1e-5, (8, 8)))
    split = advance(initial)
    expected = advance(split)
    contract = make_restart_contract(grid, p, dtype='float64', forcing=None,
                                     controls={}, code_paths={}, execution={'backend': 'cpu'})
    path = tmp_path / 'state.npz'
    save_restart(path, split, contract, step=1, counters={}, cumulative={}, history={})
    saved = load_restart(path, contract)
    resumed = advance(JaxStateG(**{name: jnp.asarray(value) for name, value in saved.state.items()}))
    for name in initial._fields:
        np.testing.assert_array_equal(getattr(expected, name), getattr(resumed, name))
    old = dict(contract, effective_params=dict(contract['effective_params'], pressure_continuity_scheme='centered_second'))
    with pytest.raises(ValueError, match='contract'):
        load_restart(path, old)


@pytest.mark.parametrize('change', [dict(project_adv_vel=True),
                                   dict(meridional_boundary_scheme='clamped_nodes'),
                                   dict(pressure_continuity_scheme='unknown')])
def test_unqualified_combinations_are_rejected(change):
    with pytest.raises(ValueError, match='centered_fourth|pressure_continuity'):
        make_channel(8, **change)
