"""Fixed-uniform Ft matrix, selective momentum owner and configuration guards."""
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
    _advection_flux_form,
    _advection_scalar,
    _compute_vertical_velocity,
    _vertical_transport_iface,
)
from zhenmode.model.solver.factory import make_solver_global
from zhenmode.model.solver.numerics.horizontal import (
    _dealias_h_fd,
    _filter_tangent_gram_fd,
    _gradient_face_gated_3d,
)
from zhenmode.model.solver.numerics.vertical import _d_dz, _fill_ghost_bottom


def options(**changes):
    return dict(column_mean_scheme='nodal_capacity', vertical_tracer_scheme='material_point_v4',
                tracer_flux_arithmetic='integrated_reference_v1',
                material_point_gradient_scheme='fourth_uniform_v1',
                momentum_tangent_filter_scheme='gram_fourth_v1') | changes


@pytest.mark.parametrize('n', [4, 5, 8, 17])
def test_dense_matrix_fixed_norm_mean_and_affine(n):
    p = SimpleNamespace(dealias_lon_mask=jnp.ones((8, 1, 1)))
    R = np.zeros((n-2, n))
    for i in range(n-2):
        R[i, i:i+3] = [1., -2., 1.]
    F = np.eye(n)-R.T@R/16
    for q in (np.zeros(n), np.full(n, 7.), np.arange(n, dtype=float),
              np.arange(n, dtype=float)**2, (-1.)**np.arange(n), np.eye(n)[0]):
        raw = np.broadcast_to(q[None, :, None], (8, n, 2)).copy()
        out = np.asarray(jax.jit(lambda a: _filter_tangent_gram_fd(a, p))(jnp.asarray(raw)))
        exact = np.broadcast_to((F@q)[None, :, None], raw.shape)
        eps = np.finfo(float).eps
        np.testing.assert_allclose(out, exact, rtol=0., atol=128*eps*np.sum(abs(raw)))
        assert np.sum(out*out) <= np.sum(raw*raw)*(1+512*eps)
        assert abs(np.sum(out-raw)) <= 512*eps*np.sum(abs(raw))
        if np.max(abs(R@q)) == 0.:
            np.testing.assert_array_equal(out, raw)
    if n == 4:
        np.testing.assert_array_equal(F@np.array([0., 1., 4., 9.]), [-1/8, 9/8, 33/8, 71/8])
        impulse = F[:, 0]
        assert impulse.min() == -1/16
        assert np.sum(abs(np.diff(impulse))) == 17/16
        plant = (np.eye(n)-R.T@R/4)@np.array([1., -1., 1., -1.])
        assert np.sum(plant*plant) > 4.


def test_longitude_projection_and_minimum_shape():
    p = SimpleNamespace(dealias_lon_mask=jnp.asarray([1, 1, 1, 0, 0, 0, 1, 1])[:, None, None])
    high = np.broadcast_to((-1.)**np.arange(8)[:, None, None], (8, 4, 1))
    np.testing.assert_array_equal(_filter_tangent_gram_fd(jnp.asarray(high), p), 0.)
    low = np.broadcast_to(np.cos(2*np.pi*np.arange(8)/8)[:, None, None], (8, 4, 1))
    np.testing.assert_allclose(_filter_tangent_gram_fd(jnp.asarray(low), p), low, rtol=0., atol=1e-14)
    with pytest.raises(ValueError):
        _filter_tangent_gram_fd(jnp.ones((8, 3, 1)), p)


@pytest.mark.parametrize('change', (
    {'momentum_tangent_filter_scheme': 'unknown'}, {'dtype': 'float32'},
    {'vertical_tracer_scheme': 'muscl_band_v3'}, {'material_point_gradient_scheme': 'minmod_v1'},
    {'tracer_flux_arithmetic': 'legacy'}, {'hydrostatic_integral_scheme': 'trapezoidal'},
    {'column_mean_scheme': 'cubic_consistent_v1'}, {'mode_split': True},
    {'process_time_scheme': 'consistent_split_v1'}, {'project_adv_vel': True},
    {'momentum_vertical_velocity_scheme': 'point_primitive_v1'},
    {'pressure_continuity_scheme': 'centered_second'},
))
def test_unsupported_family_is_rejected(change):
    with pytest.raises(ValueError):
        column_case(**options(**change))


@pytest.mark.parametrize('metric', ['cos_lat', 'dx_2d'])
def test_nonuniform_horizontal_metric_rejected(metric):
    grid, _ = column_case(**options(momentum_tangent_filter_scheme='binomial_v1'))
    values = np.asarray(getattr(grid, metric)).copy()
    values.flat[0] *= .99
    grid = replace(grid, **{metric: values})
    physics = PhysicsConfig(**dict.fromkeys(('nu_h', 'nu_v', 'nu_bi', 'kappa_h', 'kappa_v',
                         'kappa_bi', 'kappa_conv', 'kappa_gm', 'kappa_redi', 'r_bot', 'cd'), 0.))
    with pytest.raises(ValueError):
        make_solver_global(grid, physics, 12.5, return_params=True, column_geometry='nodal_dual_v1',
                           external_mode_scheme='symmetric', meridional_boundary_scheme='closed_faces',
                           pressure_continuity_scheme='centered_fourth', density_rotation_scheme='joint_shear',
                           hydrostatic_surface_scheme='linear_density', hydrostatic_integral_scheme='cubic_uniform_v1',
                           polar_cap_rows=0, polar_cap_taper=0, **options())


def test_default_contract_fingerprint_and_nondefault_tamper():
    args = dict(scheme='material_point_v4', integral='cubic_uniform_v1',
                arithmetic='integrated_reference_v1', gradient='fourth_uniform_v1')
    original = development_contract(**args)
    assert digest(original) == 'b6fa93015619fc3759e45365338f58531aa81da892c287bda1ce9818a2d0267b'
    assert original == development_contract(**args, tangent_filter='binomial_v1')
    changed = development_contract(**args, tangent_filter='gram_fourth_v1')
    validate_contract(changed)
    broken = deepcopy(changed)
    broken['ocean_options']['momentum_tangent_filter_scheme'] = 'binomial_v1'
    with pytest.raises(ValueError, match='differs'):
        validate_contract(broken)


def test_only_tangent_filter_changes_actual_transport_owner():
    grid, (_, init, _, p, _) = column_case(**options())
    old = p._replace(momentum_tangent_filter_scheme='binomial_v1')
    u, v, q = transport_fields(grid)
    s = init()._replace(u=u, v=v, T=q, S=jnp.full_like(q, 35.))
    F = _vertical_transport_iface(u, v, p)
    np.testing.assert_array_equal(F, _vertical_transport_iface(u, v, old))
    w = _compute_vertical_velocity(s, p)
    np.testing.assert_array_equal(w, _compute_vertical_velocity(s, old))
    np.testing.assert_array_equal(_advection_scalar(q, u, v, F, p), _advection_scalar(q, u, v, F, old))
    revised = jax.jit(lambda a, b, c: _advection_flux_form(a, b, c, p))(u, v, w)
    baseline = jax.jit(lambda a, b, c: _advection_flux_form(a, b, c, old))(u, v, w)
    np.testing.assert_array_equal(revised[1], baseline[1])
    gx, gy = _gradient_face_gated_3d(u, p)
    gz = _d_dz(_fill_ghost_bottom(u, p), p)
    raw = -(u*gx+v*gy+w*gz)
    np.testing.assert_allclose(revised[0], _filter_tangent_gram_fd(raw, p)*p.wet_mask_z, rtol=0., atol=2e-12)
    np.testing.assert_allclose(baseline[0], _dealias_h_fd(raw, old)*old.wet_mask_z, rtol=0., atol=2e-12)
    np.testing.assert_array_equal(_dealias_h_fd(q, p), _dealias_h_fd(q, old))
    zero = _advection_flux_form(jnp.zeros_like(u), jnp.zeros_like(v), jnp.zeros_like(w), p)
    np.testing.assert_array_equal(zero[0], 0.)
    np.testing.assert_array_equal(zero[1], 0.)
