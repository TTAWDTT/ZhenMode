"""Declared derivative, owner isolation, gauge and parity controls."""
from copy import deepcopy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.uniform_tracer_channel import column_case, transport_fields
from zhenmode.benchmarks.internal_wave import development_contract, validate_contract
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.model.solver.dynamics.transport import (
    _advection_scalar,
    _horizontal_tracer_fluxes,
    _layer_face_transports,
    _material_point_scalar_rate,
    _vertical_transport_iface,
)
from zhenmode.model.solver.numerics.vertical import fourth_uniform_tracer_gradient


def case(**changes):
    options = dict(column_mean_scheme='nodal_capacity', vertical_tracer_scheme='material_point_v4',
                   tracer_flux_arithmetic='integrated_reference_v1',
                   material_point_gradient_scheme='fourth_uniform_v1') | changes
    return column_case(**options)


@pytest.mark.parametrize('degree', range(5))
def test_polynomial_derivative_at_all_physical_nodes(degree):
    depth = np.linspace(0., 100., 19)
    p = SimpleNamespace(dz_iface=jnp.asarray(np.diff(depth))[None, None, :])
    q = jnp.asarray(7.+(depth/100.)**degree)[None, None, :]
    exact = np.zeros(19) if degree == 0 else degree*(depth/100.)**(degree-1)/100.
    result = jax.jit(lambda values: fourth_uniform_tracer_gradient(values, p))(q)
    np.testing.assert_allclose(result[0, 0], exact, rtol=0., atol=1e-14)
    if degree == 0:
        np.testing.assert_array_equal(result, 0.)


@pytest.mark.parametrize('changes', (
    {'material_point_gradient_scheme': 'unknown'},
    {'vertical_tracer_scheme': 'muscl_band_v3'},
    {'dtype': 'float32'},
    {'tracer_flux_arithmetic': 'legacy'},
    {'column_mean_scheme': 'cubic_consistent_v1'},
))
def test_factory_rejects_unsupported_gradient(changes):
    with pytest.raises(ValueError):
        case(**changes)


def test_default_contract_and_serialized_selector():
    options = dict(scheme='material_point_v4', integral='cubic_uniform_v1',
                   arithmetic='integrated_reference_v1')
    original = development_contract(**options)
    assert digest(original) == '6be0816e669dd5b2110f3565ceeabd7dd880469392831aca207eaa3458e5e890'
    assert original == development_contract(**options, gradient='minmod_v1')
    revised = development_contract(**options, gradient='fourth_uniform_v1')
    validate_contract(revised)
    altered = deepcopy(revised)
    altered['ocean_options']['material_point_gradient_scheme'] = 'minmod_v1'
    with pytest.raises(ValueError, match='differs'):
        validate_contract(altered)


def test_real_owner_constant_mass_residual_gauge_inventory_and_baseline():
    grid, (_, _, _, p, _) = case()
    u, v, q = transport_fields(grid)
    F = _vertical_transport_iface(u, v, p)
    old = p._replace(material_point_gradient_scheme='minmod_v1')
    np.testing.assert_array_equal(F, _vertical_transport_iface(u, v, old))
    for constant in (1., 35., -7.):
        rate = jax.jit(lambda a: _advection_scalar(a, u, v, F, p))(jnp.full_like(q, constant))
        np.testing.assert_allclose(rate, 0., rtol=0., atol=1e-17)
    wrong = _advection_scalar(jnp.full_like(q, 35.), u, v, -F, p)
    assert float(jnp.max(abs(wrong))) > 1e-12
    apply = jax.jit(lambda a: _advection_scalar(a, u, v, F, p, return_boundary=True))
    first, boundary = apply(q)
    second, _ = apply(q+273.15)
    np.testing.assert_allclose(first, second, rtol=0., atol=1e-13)
    assert abs(float(jnp.sum(first*p.dz_node)-jnp.sum(boundary))) < 1e-13
    mass = _layer_face_transports(u, v, p)
    fx, fy = _horizontal_tracer_fluxes(q-q.reshape(-1)[0], u, v, p, mass)
    for params in (p, old):
        _, _, diagnostics = _material_point_scalar_rate(q, F, mass, (fx*p.dz_node, fy*p.dz_node),
                                                        params, return_diagnostics=True)
        if params is p:
            baseline = diagnostics['low_rate']
        else:
            np.testing.assert_array_equal(baseline, diagnostics['low_rate'])


def test_unlimited_target_preserves_bilinear_amplitude_parity():
    grid, (_, _, _, p, _) = case()
    d = -grid.z[None, None, :]
    Y = np.pi*(np.arange(grid.ny)[None, :, None]+.5)/grid.ny
    shape = (grid.nx, grid.ny, grid.nz)
    bg = jnp.asarray(np.broadcast_to(15.-.02*d, shape))

    @jax.jit
    def target(amplitude, wave):
        u = jnp.zeros(shape)
        v = jnp.broadcast_to(amplitude*.0002*jnp.sin(Y)*jnp.cos(jnp.pi*d/100), shape)
        q = bg+jnp.broadcast_to(wave*amplitude*.02*jnp.cos(Y)*jnp.sin(jnp.pi*d/100), shape)
        mass = _layer_face_transports(u, v, p)
        F = _vertical_transport_iface(u, v, p, face_transport=mass)
        fx, fy = _horizontal_tracer_fluxes(q-q.reshape(-1)[0], u, v, p, mass)
        _, _, di = _material_point_scalar_rate(q, F, mass, (fx*p.dz_node, fy*p.dz_node),
                                               p, return_diagnostics=True)
        return di['desired']

    odd = (target(.01, 1.)-target(-.01, 1.)-target(.01, 0.)+target(-.01, 0.))/2
    np.testing.assert_allclose(odd, 0., rtol=0., atol=1e-20)
