"""Polynomial moments, actual faces, source ownership and unchanged capacities."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.uniform_tracer_channel import column_case
from zhenmode.model.solver.dynamics.pressure import _compute_bt_rho_pgf, _density_pressure_gradient
from zhenmode.model.solver.dynamics.transport import (
    _advection_scalar,
    _barotropic_velocity,
    _column_divergence,
    _face_transport_divergence,
    _layer_face_transports,
    _vertical_transport_iface,
)
from zhenmode.model.solver.numerics.column import column_mean, cubic_uniform_mean_weights
from zhenmode.model.solver.timestepping.step import _rotate_baroclinic_shear


@pytest.mark.parametrize('nodes', (6, 7, 16, 32))
@pytest.mark.parametrize('degree', (0, 1, 2, 3))
def test_point_quadrature_matches_independent_polynomial_primitive(nodes, degree):
    x = np.linspace(0., 1., nodes)
    weights = cubic_uniform_mean_weights(nodes)
    np.testing.assert_allclose(weights @ x**degree, 1/(degree+1), rtol=0, atol=2e-15)
    assert np.min(weights) > 0.
    if degree == 2:
        trap = np.ones(nodes)/(nodes-1)
        trap[[0, -1]] /= 2
        assert abs(trap @ x**degree-1/3) > 1e-5


def test_actual_faces_constant_scalar_and_inventory_share_the_same_transport():
    grid, (_, initialize, _, p, _) = column_case()
    rng = np.random.default_rng(12122026)
    u, v = [jnp.asarray(rng.normal(size=(8, 8, 16))*.001) for _ in range(2)]
    faces = _layer_face_transports(u, v, p)
    mean_x, mean_y = _barotropic_velocity(u, v, p)
    mean_faces = _layer_face_transports(mean_x[..., None], mean_y[..., None], p)
    for actual, expected in zip(faces, mean_faces, strict=True):
        np.testing.assert_allclose(jnp.sum(actual, axis=-1), jnp.sum(expected, axis=-1), rtol=0, atol=2e-16)
    F = _vertical_transport_iface(u, v, p)
    divergence = _face_transport_divergence(*faces, p)
    np.testing.assert_allclose(F[..., 0], _column_divergence(u, v, p), rtol=0, atol=2e-20)
    np.testing.assert_allclose(divergence+(F[..., 1:]-F[..., :-1]), 0., rtol=0, atol=2e-20)
    np.testing.assert_allclose(_advection_scalar(jnp.full_like(u, 35.), u, v, F, p), 0., rtol=0, atol=1e-17)
    q = jnp.asarray(15+rng.normal(size=u.shape))
    rate, boundary = _advection_scalar(q, u, v, F, p, return_boundary=True)
    np.testing.assert_allclose(jnp.sum(rate*p.dz_node), jnp.sum(boundary), rtol=0, atol=2e-15)
    _, (_, _, _, old, _) = column_case(column_mean_scheme='nodal_capacity')
    np.testing.assert_array_equal(p.dz_node, old.dz_node)
    np.testing.assert_array_equal(p.dz_norm, old.dz_norm)
    assert initialize().T.shape == (grid.nx, grid.ny, grid.nz)


def test_pressure_and_shear_use_the_same_column_mean_once():
    _, (_, initialize, _, p, _) = column_case()
    rng = np.random.default_rng(12122027)
    state = initialize()._replace(T=jnp.asarray(15+rng.normal(size=(8, 8, 16))*.01))
    force = _density_pressure_gradient(state, p)
    mean = _compute_bt_rho_pgf(state, p)
    for f, m in zip(force, mean, strict=True):
        np.testing.assert_allclose(column_mean(f, p), m, rtol=0, atol=1e-18)
        np.testing.assert_allclose(column_mean(f-m[..., None], p), 0., rtol=0, atol=1e-18)
    u, v = [jnp.asarray(rng.normal(size=(8, 8, 16))*.001) for _ in range(2)]
    updated = _rotate_baroclinic_shear(u, v, p, 6.25,
                                     forcing=tuple(f-m[..., None] for f, m in zip(force, mean, strict=True)))
    for a, b in zip(_barotropic_velocity(*updated, p), _barotropic_velocity(u, v, p), strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=2e-18)


@pytest.mark.parametrize('changes', (
    {'hydrostatic_integral_scheme': 'trapezoidal'}, {'density_rotation_scheme': 'split_sources'},
    {'column_mean_scheme': 'unrecognized'}, {'project_adv_vel': True},
))
def test_factory_rejects_unqualified_mean_owners(changes):
    with pytest.raises(ValueError):
        column_case(**changes)
