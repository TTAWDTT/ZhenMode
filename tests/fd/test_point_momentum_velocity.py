"""Explicit physical-node momentum velocity; scalar face owner stays unchanged."""
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.uniform_tracer_channel import column_case
from zhenmode.benchmarks.internal_wave import development_contract, validate_contract
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.model.solver.dynamics.tendencies import _compute_momentum_tendency
from zhenmode.model.solver.dynamics.transport import (
    _advection_flux_form,
    _compute_vertical_velocity,
    _vertical_transport_iface,
)


def point_case(**changes):
    return column_case(column_mean_scheme='nodal_capacity',
                       tracer_flux_arithmetic='integrated_reference_v1',
                       momentum_vertical_velocity_scheme='point_primitive_v1', **changes)


def test_point_velocity_analytic_nodes_and_momentum_owner():
    grid, (_, init, _, p, _) = point_case()
    nx, ny, nz = 8, 8, 16
    depth = -grid.z
    phase = np.pi*(np.arange(ny)+.5)/ny
    u = jnp.asarray(np.broadcast_to(.01+.001*depth, (nx, ny, nz)).copy())
    v = jnp.asarray(np.broadcast_to(.16/100*np.sin(phase)[None, :, None]
                                  *(1-2*depth/100), (nx, ny, nz)).copy())
    state = init()._replace(u=u, v=v)
    baseline = p._replace(momentum_vertical_velocity_scheme='dual_face_v1')
    mass = jax.jit(lambda a, b: _vertical_transport_iface(a, b, p))(u, v)
    mass_old = jax.jit(lambda a, b: _vertical_transport_iface(a, b, baseline))(u, v)
    np.testing.assert_array_equal(mass, mass_old)
    old_w = jax.jit(lambda s: _compute_vertical_velocity(s, baseline))(state)
    np.testing.assert_array_equal(old_w, -mass[..., :-1]*p.wet_mask_z)
    point_w = jax.jit(lambda s: _compute_vertical_velocity(s, p))(state)
    coefficient = -jnp.diff(mass, axis=-1)[..., 0]/p.dz_node[..., 0]
    exact_w = coefficient[..., None]*depth*(1-depth/100)
    np.testing.assert_allclose(point_w[..., 1:], exact_w[..., 1:], rtol=0., atol=1e-14)
    np.testing.assert_array_equal(point_w[..., 0], -mass[..., 0])
    np.testing.assert_array_equal(point_w[..., -1], 0.)
    assert float(jnp.max(jnp.abs(old_w[..., -1]))) > 1e-10
    # The real owner must consume this w; Coriolis, pressure and diffusion are
    # common and cancel. Scalar face fluxes remain the original exact inverse.
    old_adv = jax.jit(lambda: _advection_flux_form(u, v, old_w, baseline))()
    new_adv = jax.jit(lambda: _advection_flux_form(u, v, point_w, p))()
    old_rhs = jax.jit(lambda s: _compute_momentum_tendency(s, baseline))(state)
    new_rhs = jax.jit(lambda s: _compute_momentum_tendency(s, p))(state)
    for actual_new, actual_old, adv_new, adv_old in zip(new_rhs, old_rhs, new_adv, old_adv, strict=True):
        np.testing.assert_allclose(actual_new-actual_old, adv_new-adv_old, rtol=0., atol=1e-14)


@pytest.mark.parametrize('changes', (
    {'momentum_vertical_velocity_scheme': 'unknown'},
    {'tracer_flux_arithmetic': 'legacy'},
    {'column_mean_scheme': 'cubic_consistent_v1'},
    {'pressure_continuity_scheme': 'centered_second'},
    {'dtype': 'float32'},
))
def test_factory_rejects_unqualified_point_velocity(changes):
    options = dict(column_mean_scheme='nodal_capacity',
                   tracer_flux_arithmetic='integrated_reference_v1',
                   momentum_vertical_velocity_scheme='point_primitive_v1') | changes
    with pytest.raises(ValueError):
        column_case(**options)


def test_default_contract_digest_preserved_and_point_declaration_checked():
    options = dict(scheme='material_point_v4', integral='cubic_uniform_v1',
                   arithmetic='integrated_reference_v1')
    original = development_contract(**options)
    assert digest(original) == '6be0816e669dd5b2110f3565ceeabd7dd880469392831aca207eaa3458e5e890'
    assert original == development_contract(**options, momentum='dual_face_v1')
    revised = development_contract(**options, momentum='point_primitive_v1')
    assert revised['development']['momentum_vertical_velocity_scheme'] == 'point_primitive_v1'
    assert revised['ocean_options']['momentum_vertical_velocity_scheme'] == 'point_primitive_v1'
    validate_contract(revised)
    tampered = deepcopy(revised)
    tampered['ocean_options']['momentum_vertical_velocity_scheme'] = 'dual_face_v1'
    with pytest.raises(ValueError, match='differs'):
        validate_contract(tampered)
