"""Analytic primitive and exact scalar isolation for momentum selector."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.uniform_tracer_channel import column_case
from zhenmode.model.solver.dynamics.transport import (
    _advection_scalar,
    _compute_vertical_velocity,
    _vertical_transport_iface,
)


def fixture():
    return column_case(column_mean_scheme='nodal_capacity',
                       tracer_flux_arithmetic='integrated_reference_v1',
                       momentum_vertical_velocity_scheme='point_primitive_v1')


@pytest.mark.parametrize('degree', range(4))
def test_velocity_owner_degree_zero_to_three(degree):
    grid, (_, init, _, p, _) = fixture()
    depth = -grid.z
    phase = np.pi*(np.arange(8)+.5)/8
    v = jnp.asarray(np.broadcast_to(.001*np.sin(phase)[None, :, None]
                                  *(depth/100)**degree, (8, 8, 16)).copy())
    u = jnp.zeros_like(v)
    state = init()._replace(u=u, v=v)
    mass = jax.jit(lambda a, b: _vertical_transport_iface(a, b, p))(u, v)
    # Last node D=1 gives the independent horizontal derivative coefficient.
    # Integrating that coefficient times D^degree analytically tests physical
    # nodes and the real velocity owner, not a copied production quadrature.
    coefficient = -jnp.diff(mass, axis=-1)[..., -1]/p.dz_node[..., -1]
    exact_w = -coefficient[..., None]*100*(1-(depth/100)**(degree+1))/(degree+1)
    point = jax.jit(lambda s: _compute_vertical_velocity(s, p))(state)
    np.testing.assert_allclose(point[..., 1:], exact_w[..., 1:], rtol=0., atol=1e-14)
    np.testing.assert_array_equal(point[..., 0], -mass[..., 0])
    np.testing.assert_array_equal(point[..., -1], 0.)


def test_zero_divergence_and_explicit_nonmaterial_tracer():
    _, (_, init, _, p, _) = fixture()
    assert p.vertical_tracer_scheme == 'muscl_band_v3'
    state = init()._replace(u=jnp.zeros((8, 8, 16)), v=jnp.zeros((8, 8, 16)))
    point = jax.jit(lambda s: _compute_vertical_velocity(s, p))(state)
    np.testing.assert_array_equal(point, 0.)
    _, (_, _, _, default, _) = column_case()
    assert default.momentum_vertical_velocity_scheme == 'dual_face_v1'


@pytest.mark.parametrize('constant,badmass', ((False, False), (True, False), (True, True)))
def test_selector_keeps_scalar_face_mass_residual_owner_exact(constant, badmass):
    grid, (_, _, _, p, _) = fixture()
    baseline = p._replace(momentum_vertical_velocity_scheme='dual_face_v1')
    depth = -grid.z
    phase = np.pi*(np.arange(8)+.5)/8
    velocity = jnp.asarray(np.broadcast_to(.001*np.sin(phase)[None, :, None]
                                         *(1-2*depth/100), (8, 8, 16)).copy())
    u = jnp.zeros_like(velocity)
    mass = _vertical_transport_iface(u, velocity, p)
    if badmass:
        mass = mass.at[..., 8].add(1e-5)
    tracer = jnp.full_like(velocity, 35.) if constant else (
        14.+.01*jnp.asarray(depth)[None, None]+.2*jnp.cos(jnp.asarray(phase))[None, :, None])
    old = jax.jit(lambda q: _advection_scalar(q, u, velocity, mass, baseline))(tracer)
    new = jax.jit(lambda q: _advection_scalar(q, u, velocity, mass, p))(tracer)
    np.testing.assert_array_equal(old, new)
    if constant and badmass:
        assert float(jnp.max(jnp.abs(new))) > 1e-12


def test_nonzero_constant_flow_retains_actual_surface_mass_residual():
    _, (_, init, _, p, _) = fixture()
    state = init()._replace(u=jnp.full((8, 8, 16), .01), v=jnp.zeros((8, 8, 16)))

    @jax.jit
    def observe(s):
        return _vertical_transport_iface(s.u, s.v, p), _compute_vertical_velocity(s, p)

    mass, point = observe(state)
    # Nominal continuum divergence zero does not imply compiled discrete zero.
    # Preserve the actual surface residual exactly, including on backends that
    # evaluate this same flux expression as an exact zero. No epsilon clipping.
    np.testing.assert_array_equal(point[..., 0], -mass[..., 0])
    np.testing.assert_array_equal(point[..., -1], 0.)
