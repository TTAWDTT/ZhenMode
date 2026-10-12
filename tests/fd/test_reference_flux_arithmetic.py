"""Exact nonzero constant controls and explicit compressible mass residuals."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zhenmode.model.solver.numerics.vertical import integrated_reference_scalar_rate


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
