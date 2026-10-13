"""Independent symbols, algebraic witnesses and owner controls for paired faces.

These tests make no full-model order claim. In particular, the quadratic-depth
compatibility defect is an expected retained result, not a tolerance to repair.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.uniform_tracer_channel import column_case, transport_fields
from zhenmode.benchmarks.internal_wave import development_contract, validate_contract
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.solver.dynamics.transport import (
    _advection_scalar,
    _horizontal_tracer_fluxes,
    _layer_face_transports,
    _material_point_scalar_rate,
    _vertical_transport_iface,
)
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.model.solver.numerics.channel import (
    channel_paired_product_fluxes,
    channel_tracer_fluxes,
)


def case(*, paired=True, **changes):
    options = dict(column_mean_scheme='nodal_capacity', vertical_tracer_scheme='material_point_v4',
                   tracer_flux_arithmetic='integrated_reference_v1',
                   material_point_gradient_scheme='fourth_uniform_v1')
    if paired:
        options['tracer_transport_scheme'] = 'paired_product_fourth'
    return column_case(**(options | changes))


@pytest.fixture(scope='module')
def owner_case():
    return case()


def analytic_columns():
    """Dyadic capacities isolate the horizontal and quadratic-depth identities."""
    depth = np.arange(9, dtype=float)
    h = np.ones(9)
    h[[0, -1]] = .5
    p = SimpleNamespace(dz_node=jnp.asarray(h)[None, None, :],
                        dz_iface=jnp.ones((1, 1, 8)), column_geometry='nodal_dual_v1',
                        pressure_continuity_scheme='centered_fourth')
    x = 2*np.pi*(np.arange(16)[:, None, None]+.5)/16
    y = np.pi*(np.arange(8)[None, :, None]+.5)/8
    return p, x, y, depth[None, None, :], (16, 8, 9)


def test_depth_constant_variable_velocity_product_symbol_and_column_identity():
    p, x, _, d, shape = analytic_columns()
    q = jnp.asarray(np.broadcast_to(np.cos(x), shape))
    u = jnp.asarray(np.broadcast_to((.05+.02*np.sin(x))*(1+d/8), shape))
    v = jnp.zeros(shape)
    mass = _layer_face_transports(u, v, p)
    faces, moments = channel_paired_product_fluxes(q, u, v, p, mass)
    dx, theta = 2000., 2*np.pi/shape[0]
    kd1 = np.sin(theta)*(4-np.cos(theta))/(3*dx)
    kd2 = np.sin(2*theta)*(4-np.cos(2*theta))/(3*dx)
    # U*q = (1+d/8)*(.05*cos(x)+.01*sin(2*x)), without separate interpolation.
    expected = np.broadcast_to((1+d/8)*(-.05*kd1*np.sin(x)+.01*kd2*np.cos(2*x)), shape)
    derivative = (np.asarray(faces[0])-np.roll(np.asarray(faces[0]), 1, axis=0))/dx
    np.testing.assert_allclose(derivative/np.asarray(p.dz_node), expected, rtol=0., atol=1e-13)
    np.testing.assert_allclose(moments[0], np.sum(faces[0], axis=-1), rtol=0., atol=1e-13)
    np.testing.assert_array_equal(faces[1], 0.)
    np.testing.assert_array_equal(moments[1], 0.)


def test_exact_zero_mass_product_counterexample_is_a_barrier():
    p, _, _, _, shape = analytic_columns()
    u0, q0 = 1/32, 1/4
    u = jnp.zeros(shape).at[0].set(7*u0).at[1].set(u0)
    q = jnp.zeros(shape).at[1].set(q0)
    v = jnp.zeros(shape)
    mass = _layer_face_transports(u, v, p)
    np.testing.assert_array_equal(mass[0][1], 0.)
    old_faces = channel_tracer_fluxes(q, u, v, p, mass)
    np.testing.assert_allclose(old_faces[0][1], 7*u0*q0/12, rtol=0., atol=1e-17)
    faces, moments = channel_paired_product_fluxes(q, u, v, p, mass)
    np.testing.assert_array_equal(faces[0][1], 0.)
    np.testing.assert_array_equal(moments[0][1], 0.)
    # A single stagnant layer also stays closed when the rest of its column flows.
    actual = (mass[0].at[2, 3, 4].set(0.), mass[1])
    faces, _ = channel_paired_product_fluxes(q, u, v, p, actual)
    assert float(faces[0][2, 3, 4]) == 0.


def test_constant_and_gauge_flux_laws_use_actual_residual_mass():
    p, x, y, d, shape = analytic_columns()
    u = jnp.asarray(np.broadcast_to((.03+.01*np.sin(x))*(1+d/8), shape))
    v = jnp.asarray(np.broadcast_to(.02*np.sin(y)*(1+d/8), shape))
    q = jnp.asarray(np.broadcast_to(.2*np.cos(x)+.1*np.cos(y)+d/64, shape))
    mx, my = _layer_face_transports(u, v, p)
    actual = ((mx+.003).at[2, 3, 4].set(0.), (my+.002).at[:, -1].set(0.))
    constant_faces, constant_moments = channel_paired_product_fluxes(
        jnp.full(shape, 7.), u, v, p, actual)
    first_faces, first_moments = channel_paired_product_fluxes(q, u, v, p, actual)
    next_faces, next_moments = channel_paired_product_fluxes(q+7., u, v, p, actual)
    for mass, cf, cm, first, following, jm, following_jm in zip(
            actual, constant_faces, constant_moments, first_faces, next_faces,
            first_moments, next_moments, strict=True):
        np.testing.assert_allclose(cf, 7*mass, rtol=0., atol=1e-13)
        np.testing.assert_allclose(cm, 7*jnp.sum(mass, axis=-1), rtol=0., atol=1e-13)
        np.testing.assert_allclose(following-first, 7*mass, rtol=0., atol=1e-13)
        np.testing.assert_allclose(following_jm-jm, 7*jnp.sum(mass, axis=-1),
                                   rtol=0., atol=1e-13)


def test_closed_y_reflection_has_product_symbol_and_closed_inventory():
    p, _, y, d, shape = analytic_columns()
    u = jnp.zeros(shape)
    v = jnp.asarray(np.broadcast_to(np.sin(y)*(1+d/8)/32, shape))
    q = jnp.asarray(np.broadcast_to(np.cos(y), shape))
    mass = _layer_face_transports(u, v, p)
    faces, moments = channel_paired_product_fluxes(q, u, v, p, mass)
    outgoing = np.asarray(faces[1])
    incoming = np.concatenate((np.zeros_like(outgoing[:, :1]), outgoing[:, :-1]), axis=1)
    dy, theta = 2000., np.pi/shape[1]
    kd2 = np.sin(2*theta)*(4-np.cos(2*theta))/(3*dy)
    expected = np.broadcast_to((1+d/8)*kd2*np.cos(2*y)/64, shape)
    np.testing.assert_allclose((outgoing-incoming)/(dy*np.asarray(p.dz_node)), expected,
                               rtol=0., atol=1e-13)
    np.testing.assert_array_equal(outgoing[:, -1], 0.)
    np.testing.assert_array_equal(moments[1][:, -1], 0.)
    assert abs(float(np.sum(outgoing-incoming))) < 1e-13


def test_quadratic_depth_column_defect_is_retained():
    p, x, _, d, shape = analytic_columns()
    u0, height, dz = 1/32, 8., 1.
    u, v = jnp.full(shape, u0), jnp.zeros(shape)
    q = jnp.asarray(np.broadcast_to((1+np.cos(x))*d**2, shape))
    mass = _layer_face_transports(u, v, p)
    faces, moments = channel_paired_product_fluxes(q, u, v, p, mass)
    theta = 2*np.pi/shape[0]
    face_b = 1+(7*np.cos(theta/2)-np.cos(3*theta/2))/6*np.cos(x+theta/2)
    # Exact trapezoid identity E=-u0*b*H*dz^2/2; J-sum(G)=L4(E).
    expected = np.broadcast_to(-u0*height*dz**2*face_b[..., 0]/2, shape[:2])
    defect = np.asarray(moments[0])-np.sum(np.asarray(faces[0]), axis=-1)
    np.testing.assert_allclose(defect, expected, rtol=0., atol=1e-13)
    assert np.any(expected != 0.)


def test_real_owner_constant_bad_mass_gauge_inventory_and_pins(owner_case):
    grid, (_, _, _, p, _) = owner_case
    u, v, q = transport_fields(grid)
    mx, my = _layer_face_transports(u, v, p)
    actual = ((mx+.00001).at[2, 3, 5].set(0.), (my+.00002).at[:, -1].set(0.))
    F = _vertical_transport_iface(u, v, p, face_transport=actual)
    apply = jax.jit(lambda values, vertical: _advection_scalar(
        values, u, v, vertical, p, return_boundary=True, face_transport=actual))
    for constant in (1., 35., -7.):
        rate, _ = apply(jnp.full_like(q, constant), F)
        np.testing.assert_allclose(rate, 0., rtol=0., atol=1e-17)
    wrong, _ = apply(jnp.full_like(q, 35.), -F)
    assert float(jnp.max(abs(wrong))) > 1e-12
    rate, boundary = apply(q, F)
    shifted, _ = apply(q+273.15, F)
    np.testing.assert_allclose(rate, shifted, rtol=0., atol=1e-13)
    assert abs(float(jnp.sum(rate*p.dz_node)-jnp.sum(boundary))) < 1e-13
    faces, moments = channel_paired_product_fluxes(q-q.reshape(-1)[0], u, v, p, actual)
    _, _, diagnostics = _material_point_scalar_rate(
        q, F, actual, faces, p, return_diagnostics=True, column_moments=moments)
    assert float(diagnostics['anti_horizontal'][0][2, 3, 5]) == 0.
    np.testing.assert_array_equal(diagnostics['anti_horizontal'][1][:, -1], 0.)
    vertical = np.asarray(diagnostics['anti_vertical'])
    np.testing.assert_array_equal(vertical[np.asarray(F) == 0.], 0.)
    np.testing.assert_array_equal(vertical[..., [0, -1]], 0.)


def test_invalid_supplied_wall_mass_signals_nonfinite_and_infeasible(owner_case):
    grid, (_, _, _, p, _) = owner_case
    u, v, q = transport_fields(grid)
    mx, my = _layer_face_transports(u, v, p)
    actual = (mx, my.at[:, -1].set(.001))
    F = _vertical_transport_iface(u, v, p, face_transport=actual)

    @jax.jit
    def apply(values):
        faces, moments = channel_paired_product_fluxes(values-values.reshape(-1)[0],
                                                      u, v, p, actual)
        rate, _, diagnostics = _material_point_scalar_rate(
            values, F, actual, faces, p, return_diagnostics=True, column_moments=moments)
        return rate, diagnostics['feasible']

    rate, feasible = apply(q)
    assert not np.isfinite(np.asarray(rate)).all()
    assert not bool(feasible)


def test_missing_moments_and_unpaired_interface_reject(owner_case):
    grid, (_, _, _, p, _) = owner_case
    q = jnp.zeros((grid.nx, grid.ny, grid.nz))
    faces = (q, q)
    F = jnp.zeros(q.shape[:-1]+(q.shape[-1]+1,))
    with pytest.raises(ValueError, match='integrated faces and paired column moments'):
        _horizontal_tracer_fluxes(q, q, q, p, faces)
    with pytest.raises(ValueError, match='actual mass faces'):
        channel_paired_product_fluxes(q, q, q, p, None)
    for moments in (None, (q[..., 0],), (q, q)):
        with pytest.raises(ValueError, match='column moments'):
            _material_point_scalar_rate(q, F, faces, faces, p, column_moments=moments)
    with pytest.raises(ValueError, match='require paired_product_fourth'):
        _material_point_scalar_rate(q, F, faces, faces,
                                    p._replace(tracer_transport_scheme='centered_second'),
                                    column_moments=(q[..., 0], q[..., 0]))


def test_helper_rejects_broadcast_shapes_and_nonbinary64_operands():
    p, _, _, _, shape = analytic_columns()
    q = jnp.zeros(shape)
    arguments = dict(tracer=q, u=q, v=q, params=p, face_transport=(q, q))
    for name in ('tracer', 'u', 'v'):
        for invalid in (q[:1], q.astype(jnp.float32)):
            with pytest.raises(ValueError, match='paired_product_fourth'):
                channel_paired_product_fluxes(**(arguments | {name: invalid}))
    for axis in (0, 1):
        for invalid in (q[:1], q.astype(jnp.float32)):
            mass = [q, q]
            mass[axis] = invalid
            with pytest.raises(ValueError, match='paired_product_fourth'):
                channel_paired_product_fluxes(**(arguments | {'face_transport': tuple(mass)}))
    for name in ('dz_node', 'dz_iface'):
        original = getattr(p, name)
        for invalid in (jnp.broadcast_to(original, (shape[0], 1, original.shape[-1])),
                        original.astype(jnp.float32)):
            malformed = SimpleNamespace(**(vars(p) | {name: invalid}))
            with pytest.raises(ValueError, match='paired_product_fourth'):
                channel_paired_product_fluxes(**(arguments | {'params': malformed}))


def test_owner_rejects_broadcast_shapes_and_nonbinary64_operands(owner_case):
    grid, (_, _, _, p, _) = owner_case
    q = jnp.zeros((grid.nx, grid.ny, grid.nz))
    F = jnp.zeros(q.shape[:-1]+(q.shape[-1]+1,))
    column = q[..., 0]
    arguments = dict(T=q, Fz=F, mass_faces=(q, q), shifted_faces=(q, q), p=p,
                     column_moments=(column, column))
    for name, original in (('T', q), ('Fz', F)):
        for invalid in (original[:1], original.astype(jnp.float32)):
            with pytest.raises(ValueError, match='paired_product_fourth'):
                _material_point_scalar_rate(**(arguments | {name: invalid}))
    for name, original in (('mass_faces', q), ('shifted_faces', q),
                           ('column_moments', column)):
        for axis in (0, 1):
            for invalid in (original[:1], original.astype(jnp.float32)):
                pair = [original, original]
                pair[axis] = invalid
                with pytest.raises(ValueError, match='paired_product_fourth'):
                    _material_point_scalar_rate(**(arguments | {name: tuple(pair)}))
    for name in ('dz_node', 'dz_iface'):
        original = getattr(p, name)
        for invalid in (jnp.broadcast_to(original, (grid.nx, 1, original.shape[-1])),
                        original.astype(jnp.float32)):
            with pytest.raises(ValueError, match='paired_product_fourth'):
                _material_point_scalar_rate(**(arguments | {'p': p._replace(**{name: invalid})}))


def test_factory_identity_and_legacy_default_symbol(owner_case, monkeypatch):
    from zhenmode.model.solver.dynamics import transport

    assert owner_case[1][3].tracer_transport_scheme == 'paired_product_fourth'
    grid, (_, _, _, p, _) = case(paired=False)
    assert p.tracer_transport_scheme == 'centered_second'

    def forbidden(*args, **kwargs):
        raise AssertionError('legacy dispatch entered paired preparation')

    monkeypatch.setattr(transport, 'channel_paired_product_fluxes', forbidden)
    x = 2*np.pi*(np.arange(grid.nx)[:, None, None]+.5)/grid.nx
    shape = (grid.nx, grid.ny, grid.nz)
    q = jnp.asarray(np.broadcast_to(np.cos(x), shape))
    u, v = jnp.full(shape, .05), jnp.zeros(shape)
    F = _vertical_transport_iface(u, v, p)
    rate = _advection_scalar(q, u, v, F, p)
    centered_symbol = np.sin(2*np.pi/grid.nx)/2000.
    expected = np.broadcast_to(.05*centered_symbol*np.sin(x), shape)
    np.testing.assert_allclose(rate, expected, rtol=0., atol=1e-17)


def test_real_owner_depth_constant_fourth_symbol(owner_case):
    grid, (_, _, _, p, _) = owner_case
    x = 2*np.pi*(np.arange(grid.nx)[:, None, None]+.5)/grid.nx
    shape = (grid.nx, grid.ny, grid.nz)
    q = jnp.asarray(np.broadcast_to(np.cos(x), shape))
    u, v = jnp.full(shape, .05), jnp.zeros(shape)
    F = jnp.zeros(shape[:-1]+(shape[-1]+1,))
    rate = jax.jit(lambda values: _advection_scalar(values, u, v, F, p))(q)
    theta = 2*np.pi/grid.nx
    fourth_symbol = np.sin(theta)*(4-np.cos(theta))/(3*2000.)
    expected = np.broadcast_to(.05*fourth_symbol*np.sin(x), shape)
    # This end-to-end symbol would fail if the owner restored its old A2 moment.
    np.testing.assert_allclose(rate, expected, rtol=0., atol=1e-17)


def test_development_opt_in_metadata_and_legacy_digest():
    options = dict(scheme='material_point_v4', integral='cubic_uniform_v1',
                   arithmetic='integrated_reference_v1')
    original = development_contract(**options)
    assert digest(original) == '6be0816e669dd5b2110f3565ceeabd7dd880469392831aca207eaa3458e5e890'
    assert original == development_contract(**options, horizontal='centered_second')
    assert 'tracer_transport_scheme' not in original['development']
    expected = development_contract(**options, gradient='fourth_uniform_v1')
    expected['ocean_options']['tracer_transport_scheme'] = 'paired_product_fourth'
    expected['development']['tracer_transport_scheme'] = 'paired_product_fourth'
    candidate = development_contract(**options, gradient='fourth_uniform_v1',
                                     horizontal='paired_product_fourth')
    assert candidate == expected
    validate_contract(candidate)
    altered = deepcopy(candidate)
    altered['development'].pop('tracer_transport_scheme')
    with pytest.raises(ValueError, match='differs'):
        validate_contract(altered)
    with pytest.raises(ValueError, match='paired_product_fourth'):
        development_contract(**options, horizontal='paired_product_fourth')


def test_factory_rejects_nonuniform_integrated_columns(owner_case):
    grid = owner_case[0]
    z = grid.z.copy()
    z[3] += .1
    nonuniform = replace(grid, z=z, dz=-np.diff(z))
    physics = PhysicsConfig(**dict.fromkeys((
        'nu_h', 'nu_v', 'nu_bi', 'kappa_h', 'kappa_v', 'kappa_bi',
        'kappa_conv', 'kappa_gm', 'kappa_redi', 'r_bot', 'cd'), 0.))
    # The existing integrated-arithmetic guard rejects this before the paired guard.
    with pytest.raises(ValueError, match='integrated_reference_v1'):
        make_solver_global(nonuniform, physics, 12.5, column_geometry='nodal_dual_v1',
                           external_mode_scheme='symmetric', meridional_boundary_scheme='closed_faces',
                           pressure_continuity_scheme='centered_fourth',
                           hydrostatic_integral_scheme='trapezoidal', column_mean_scheme='nodal_capacity',
                           vertical_tracer_scheme='material_point_v4',
                           material_point_gradient_scheme='fourth_uniform_v1',
                           tracer_flux_arithmetic='integrated_reference_v1',
                           tracer_transport_scheme='paired_product_fourth',
                           polar_cap_rows=0, polar_cap_taper=0)


@pytest.mark.parametrize('changes', (
    {'tracer_transport_scheme': 'product_fourth'},
    {'vertical_tracer_scheme': 'muscl_band_v3'},
    {'material_point_gradient_scheme': 'minmod_v1'},
    {'dtype': 'float32'},
    {'tracer_flux_arithmetic': 'legacy'},
    {'column_mean_scheme': 'cubic_consistent_v1'},
    {'column_geometry': 'legacy'},
    {'pressure_continuity_scheme': 'centered_second'},
    {'monotone_adv': True},
    {'fct_adv': True},
    {'project_adv_vel': True},
    {'mode_split': True},
    {'momentum_vertical_velocity_scheme': 'point_primitive_v1'},
    {'momentum_tangent_filter_scheme': 'gram_fourth_v1'},
))
def test_factory_rejects_unqualified_combinations(changes):
    with pytest.raises(ValueError):
        case(**changes)
