"""Exact nonzero constant controls and explicit compressible mass residuals."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.uniform_tracer_channel import column_case
from zhenmode.model.solver.dynamics.transport import (
    _advection_scalar,
    _face_transport_divergence,
    _layer_face_transports,
    _vertical_transport_iface,
)
from zhenmode.model.solver.numerics.vertical import (
    _conservative_endpoint_flux,
    _vertical_tracer_face_values,
    integrated_reference_scalar_rate,
)


def setup(nodes):
    d = np.linspace(0., 1., nodes)
    h = np.full(nodes, 1/(nodes-1))
    h[[0, -1]] /= 2
    flow = np.r_[0., .25*np.sin(np.pi*(d[:-1]+d[1:])/2), 0.]
    mass_numerator = -np.diff(flow)
    p = SimpleNamespace(dz_node=jnp.asarray(h)[None, None], dz_iface=jnp.asarray(np.diff(d))[None, None],
                        wet_mask_z=jnp.ones((1, 1, nodes)), dt=.002, adv_nsub=1,
                        vertical_tracer_scheme='muscl_band_v3')
    return d, h, jnp.asarray(flow)[None, None], jnp.asarray(mass_numerator)[None, None], p


@pytest.mark.parametrize('nodes', (21, 41, 81))
@pytest.mark.parametrize('concentration', (1., 35., -7.))
def test_integrated_nonzero_constant_survives_original_absolute_gate(nodes, concentration):
    _, _, flow, mass, p = setup(nodes)
    tracer = jnp.full((1, 1, nodes), concentration)
    rate, _ = jax.jit(lambda q: integrated_reference_scalar_rate(q, flow, jnp.zeros_like(q), mass, p))(tracer)
    assert np.max(abs(np.asarray(rate))) <= 1e-17


def test_reference_component_retains_inconsistent_mass_flux_instead_of_clipping():
    _, h, flow, mass, p = setup(41)
    tracer = jnp.full((1, 1, 41), 35.)
    rate, _ = integrated_reference_scalar_rate(tracer, -flow, jnp.zeros_like(tracer), mass, p)
    expected = -35*(np.asarray(mass)+(np.asarray(-flow)[..., 1:]-np.asarray(-flow)[..., :-1]))/h
    np.testing.assert_allclose(rate, expected, rtol=0, atol=2e-14)
    assert np.max(abs(np.asarray(rate))) > 1e-12


def test_inventory_telescope_and_temperature_translation_use_the_full_mass_term():
    d, h, flow, mass, p = setup(41)
    original = 10+.2*np.exp(d)+.3*np.cos(2*np.pi*d)
    rates = []
    for offset in (0., 273.15):
        tracer = jnp.asarray(original+offset)[None, None]
        shifted = tracer-tracer.reshape(-1)[0]
        rate, boundary = integrated_reference_scalar_rate(tracer, flow, shifted*mass, mass, p)
        horizontal_full = np.asarray(tracer*mass)
        np.testing.assert_allclose(jnp.sum(rate*p.dz_node),
                                   -np.sum(horizontal_full)+np.sum(np.asarray(boundary)), rtol=0, atol=2e-14)
        rates.append(np.asarray(rate))
    np.testing.assert_allclose(*rates, rtol=0, atol=2e-12)


def test_shifted_salt_floor_is_the_same_absolute_zero():
    d, _, flow, mass, p = setup(41)
    tracer = jnp.asarray(2*(1-d))[None, None]
    shifted = tracer-tracer.reshape(-1)[0]
    rate, _ = integrated_reference_scalar_rate(tracer, flow, shifted*mass, mass, p, minimum=0.)
    assert np.min(np.asarray(tracer+p.dt*rate)) >= -1e-14


def test_actual_solver_owner_assembles_mass_before_capacity_division():
    _, (_, _, _, p, _) = column_case(column_mean_scheme='nodal_capacity',
                                    hydrostatic_integral_scheme='trapezoidal',
                                    tracer_flux_arithmetic='integrated_reference_v1')
    rng = np.random.default_rng(12122028)
    u, v = [jnp.asarray(rng.normal(size=(8, 8, 16))*.001) for _ in range(2)]
    faces = _layer_face_transports(u, v, p)
    mass = _face_transport_divergence(*faces, p)
    F = _vertical_transport_iface(u, v, p)
    expected = jnp.cumsum(mass[..., ::-1], axis=-1)[..., ::-1]
    np.testing.assert_array_equal(F[..., :-1], expected)
    constant = jnp.full_like(u, 35.)
    np.testing.assert_allclose(_advection_scalar(constant, u, v, F, p), 0., rtol=0, atol=1e-17)
    reversed_rate = _advection_scalar(constant, u, v, -F, p)
    expected_bad = -35*(mass+(-F[..., 1:]+F[..., :-1]))/p.dz_node
    np.testing.assert_allclose(reversed_rate, expected_bad, rtol=0, atol=2e-17)
    assert np.max(abs(np.asarray(reversed_rate))) > 1e-12


def test_nonzero_global_reference_activates_absolute_zero_salt_floor():
    tracer = jnp.asarray([[[2., 1., 0., 0., 0., 0.], [0., 1., 2., 3., 4., 5.]]])
    flow = jnp.broadcast_to(jnp.asarray([.1, .09, .07, .05, .03, .01, 0.]), (1, 2, 7))
    h = jnp.asarray([.5, 1., 1., 1., 1., .5])[None, None]
    mass = -(flow[..., 1:]-flow[..., :-1])
    p = SimpleNamespace(dz_node=h, dz_iface=jnp.ones((1, 2, 5)), wet_mask_z=jnp.ones_like(tracer),
                        dt=.01, adv_nsub=1, vertical_tracer_scheme='muscl_band_v3')
    shifted = tracer-tracer.reshape(-1)[0]
    rate, _ = integrated_reference_scalar_rate(tracer, flow, shifted*mass, mass, p, minimum=0.)
    unbounded, _ = integrated_reference_scalar_rate(tracer, flow, shifted*mass, mass, p)
    assert float((tracer+p.dt*rate)[0, 1, 0]) >= -1e-14
    assert float((tracer+p.dt*unbounded)[0, 1, 0]) < -1e-6
    a, b = _vertical_tracer_face_values(tracer, p)
    inside = flow[..., 1:-1]*jnp.where(flow[..., 1:-1] > 0., a, b)
    flux = jnp.concatenate((flow[..., :1]*tracer[..., :1], inside, jnp.zeros_like(flow[..., :1])), axis=-1)
    old = _conservative_endpoint_flux(tracer, flow, flux, tracer*mass/h, p, minimum=0.)
    np.testing.assert_allclose(rate, -(tracer*mass+(old[..., 1:]-old[..., :-1]))/h,
                               rtol=0, atol=2e-13)
