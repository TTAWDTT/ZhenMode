"""Closed-form channel modes and strict restart controls for face walls.

Expected stencils below are analytic sine/cosine identities, not imports from
the linear diagnosis or the scorer. These are development controls, not a
certification of global coastlines, mixing or the complete model's order.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zhenmode.model.io.restart import load_restart, make_restart_contract, save_restart
from zhenmode.model.solver.dynamics.transport import (
    _column_divergence,
    _face_transport_divergence,
    _layer_face_transports,
)
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.model.solver.numerics.horizontal import (
    _gradient_conservative_3d,
    _gradient_face_gated_3d,
    _mirror_latitude,
)
from zhenmode.model.solver.state import JaxStateG


def candidate():
    # Reconstruct through the factory: replacing the effective mask in a test
    # would miss validation and the public switch's actual propagation.
    from tests.support.fd.reference_geometry import _fixture

    grid, _ = _fixture()
    grid = replace(grid, cos_lat=np.ones(8), dx_2d=np.full((8, 8), 1000.),
                   dy=1000., f=np.full((8, 8), 1e-4))
    from zhenmode.model.config import PhysicsConfig

    physics = PhysicsConfig(**dict.fromkeys((
        'nu_h', 'nu_v', 'nu_bi', 'kappa_h', 'kappa_v', 'kappa_bi',
        'kappa_conv', 'kappa_gm', 'kappa_redi', 'r_bot', 'cd'), 0.))
    options = dict(return_params=True, column_geometry='nodal_dual_v1',
                   conservative_kv=True, localize_conv=True,
                   polar_cap_rows=0, polar_cap_taper=0, external_mode_scheme='symmetric',
                   meridional_boundary_scheme='closed_faces')
    return grid, physics, options, make_solver_global(grid, physics, 1., **options)


def test_closed_faces_keeps_centres_and_matches_channel_eigenmode_at_both_walls():
    _, _, _, (_, initialize, _, p, _) = candidate()
    y = (np.arange(8) + .5) * p.dy
    k = np.pi / (8 * p.dy)
    eta = jnp.broadcast_to(jnp.cos(k * y)[None, :], (8, 8))
    v = jnp.broadcast_to(jnp.sin(k * y)[None, :, None], (8, 8, 4))
    kd = np.sin(k * p.dy) / p.dy
    np.testing.assert_allclose(_column_divergence(jnp.zeros_like(v), v, p),
                               50 * kd * np.asarray(eta), atol=1e-16)
    _, gradient = _gradient_conservative_3d(eta[..., None], p)
    np.testing.assert_allclose(gradient, -kd * np.asarray(v), atol=1e-18)
    assert np.all(np.asarray(p.interior_mask_z) == 1)
    faces = _layer_face_transports(initialize().u, v, p)[1]
    assert np.all(np.asarray(faces)[:, -1] == 0)
    # Odd ghosts close the south face, too, without erasing the first centre.
    ghosts = _mirror_latitude(v, normal=True)
    assert np.all(np.asarray(ghosts[:, 0] + ghosts[:, 1]) == 0)
    assert np.all(np.asarray(v)[:, [0, -1]] != 0)
    # Negative control: the historical extra centre clamp breaks this identity.
    clamped = v.at[:, [0, -1]].set(0.)
    assert np.max(abs(np.asarray(_column_divergence(jnp.zeros_like(v), clamped, p))
                      - 50 * kd * np.asarray(eta))) > 1e-3


def test_pressure_continuity_are_adjoint_and_close_mass_for_random_velocity():
    _, _, _, (_, _, _, p, _) = candidate()
    rng = np.random.default_rng(8102026)
    eta, u, v = (jnp.asarray(rng.normal(size=shape)) for shape in
                 ((8, 8), (8, 8, 4), (8, 8, 4)))
    gx, gy = _gradient_conservative_3d(eta[..., None], p)
    divergence = _column_divergence(u, v, p)
    work = np.sum(np.asarray(eta * divergence))
    work += np.sum(np.asarray((u * gx + v * gy) * p.dz_node))
    assert abs(work) < 2e-15
    assert abs(float(jnp.sum(divergence))) < 1e-15
    fx, fy = _layer_face_transports(u, v, p)
    leaking = fy.at[:, -1].set(1.)
    assert abs(float(jnp.sum(_face_transport_divergence(fx, leaking, p)))) > .01


def test_normal_momentum_derivative_uses_odd_ghost_not_scalar_ghost():
    _, _, _, (_, _, _, p, _) = candidate()
    # A linear normal velocity near the south wall has exact derivative there.
    field = jnp.broadcast_to(jnp.asarray(np.arange(8) + .5)[None, :, None], (8, 8, 4))
    _, derivative = _gradient_face_gated_3d(field, p, normal=True)
    _, scalar = _gradient_face_gated_3d(field, p)
    np.testing.assert_allclose(derivative[:, 0], 1 / p.dy, atol=1e-18)
    np.testing.assert_allclose(scalar[:, 0], .5 / p.dy, atol=1e-18)
    ghosts = np.asarray(_mirror_latitude(field, normal=True, width=2))
    np.testing.assert_array_equal(ghosts[:, :2], -np.asarray(field)[:, 1::-1])
    np.testing.assert_array_equal(ghosts[:, -2:], -np.asarray(field)[:, :-3:-1])


def test_rotating_external_mode_converges_to_independent_discrete_channel_solution():
    from zhenmode.model.solver.dynamics.barotropic import _free_surface_step_fd

    _, _, _, (_, initialize, _, p, _) = candidate()
    phi = np.pi * (np.arange(8) + .5) / 8
    kd = np.sin(np.pi / 8) / p.dy
    f, g, depth, end, amplitude = 1e-4, 9.81, 50., 128., 1e-7
    omega = np.sqrt(f * f + g * depth * kd * kd)
    eta = amplitude * np.cos(phi) * (f * f / omega**2 + (1 - f * f / omega**2) * np.cos(omega * end))
    u = amplitude * g * kd * f / omega**2 * (1 - np.cos(omega * end)) * np.sin(phi)
    v = amplitude * g * kd / omega * np.sin(omega * end) * np.sin(phi)
    expected = np.stack((eta, u * np.sqrt(depth / g), v * np.sqrt(depth / g)))
    state = initialize()._replace(eta=jnp.broadcast_to(jnp.asarray(amplitude * np.cos(phi)), (8, 8)))
    errors = []
    for dt in (4., 2., 1.):
        @jax.jit
        def evolve(values):
            return jax.lax.fori_loop(0, int(end / dt), lambda _, s:
                _free_surface_step_fd(*s, p, dt_half=dt), values)

        actual = evolve((state.eta, state.u, state.v))
        fields = np.stack((np.asarray(actual[0])[0],
                           np.asarray(actual[1])[0, :, 0] * np.sqrt(depth / g),
                           np.asarray(actual[2])[0, :, 0] * np.sqrt(depth / g)))
        errors.append(np.max(abs(fields - expected)) / amplitude)
    assert all(3.8 < a / b < 4.2 for a, b in zip(errors[:-1], errors[1:], strict=True)), errors


def test_complete_step_keeps_wall_shear_and_restart_is_bit_exact(tmp_path):
    grid, _, _, (advance, initialize, _, p, _) = candidate()
    shear = np.array([1., .5, -.5, -1.])
    shear -= np.sum(shear * np.asarray(p.dz_node)) / 50
    initial = initialize()._replace(u=jnp.broadcast_to(jnp.asarray(1e-5 * shear), (8, 8, 4)))
    contract = make_restart_contract(grid, p, dtype='float64', forcing=None,
                                     controls={}, code_paths={}, execution={'backend': jax.default_backend()})
    current = initial
    for _ in range(4):
        current = advance(current)
    uninterrupted = current
    split = advance(advance(initial))
    path = tmp_path / 'state.npz'
    save_restart(path, split, contract, step=2, counters={}, cumulative={}, history={})
    loaded = load_restart(path, contract)
    split = JaxStateG(**{name: jnp.asarray(value) for name, value in loaded.state.items()})
    resumed = advance(advance(split))
    for name in initial._fields:
        np.testing.assert_array_equal(getattr(resumed, name), getattr(uninterrupted, name))
    assert np.max(abs(np.asarray(resumed.v)[:, [0, -1]])) > 1e-10
    # Wrong scheme is rejected by the effective-parameter restart contract.
    old = dict(contract)
    old['effective_params'] = dict(contract['effective_params'], meridional_boundary_scheme='clamped_nodes')
    with pytest.raises(ValueError, match='contract'):
        load_restart(path, old)


def test_factory_limits_new_scheme_without_changing_old_default():
    grid, physics, options, _ = candidate()
    changes = [dict(external_mode_scheme='forward_backward'), dict(polar_cap_rows=1)]
    for change in changes:
        with pytest.raises(ValueError, match='closed_faces'):
            make_solver_global(grid, physics, 1., **(options | change))
    with pytest.raises(ValueError, match='all-wet flat Cartesian'):
        make_solver_global(replace(grid, depth=np.full((8, 8), 40.)), physics, 1., **options)
    with pytest.raises(ValueError, match='mixing'):
        make_solver_global(grid, replace(physics, nu_h=1.), 1., **options)
    defaults = {name: value for name, value in options.items()
                if name != 'meridional_boundary_scheme'}
    _, _, _, old, _ = make_solver_global(grid, physics, 1., **defaults)
    assert old.meridional_boundary_scheme == 'clamped_nodes'
    assert np.all(np.asarray(old.interior_mask_z)[:, [0, -1]] == 0)
