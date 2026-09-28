"""Projection must constrain native wet-face continuity, not a surface-mask proxy."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_horizontal_tracer_diffusion import _parameters

from jax_solver_global import (
    _advection_scalar,
    _column_divergence,
    _gradient_conservative_3d,
    _project_column_divergence,
    _step_impl,
    _vertical_transport_iface,
)
from stage_budgets import make_budget_step


def _velocities(params, seed=2718):
    random = np.random.default_rng(seed)
    shape = params.wet_mask_z.shape
    return tuple(jnp.asarray(random.normal(size=shape)) * params.wet_mask_z for _ in range(2))


@pytest.mark.parametrize("land", [False, True])
@pytest.mark.parametrize("seed", [2718, 3141, 1618])
def test_column_constraint_matches_actual_top_transport(land, seed):
    _, params, _ = _parameters(65., land=land)
    velocity_x, velocity_y = _velocities(params, seed)
    actual = np.asarray(_vertical_transport_iface(velocity_x, velocity_y, params)[..., 0])
    constraint = np.asarray(_column_divergence(velocity_x, velocity_y, params))
    assert np.max(np.abs(actual - constraint)) <= 5e-13 * np.max(np.abs(actual))


@pytest.mark.parametrize("land", [False, True])
def test_native_column_constraint_has_volume_weighted_negative_adjoint(land):
    _, params, volume = _parameters(65., land=land)
    velocity_x, velocity_y = _velocities(params)
    random = np.random.default_rng(3141)
    potential = jnp.asarray(random.normal(size=params.wet_mask.shape))
    other = jnp.asarray(random.normal(size=params.wet_mask.shape))
    gradient_x, gradient_y = _gradient_conservative_3d(potential[..., None], params)
    area = np.asarray(params.dx_2d) * params.dy * np.asarray(params.wet_mask)
    column_product = np.asarray(potential * _column_divergence(velocity_x, velocity_y, params)) * area
    gradient_product = np.asarray(gradient_x * velocity_x + gradient_y * velocity_y) * volume
    assert abs(column_product.sum() + gradient_product.sum()) <= 5e-13 * (np.abs(column_product).sum() + np.abs(gradient_product).sum())
    other_x, other_y = _gradient_conservative_3d(other[..., None], params)
    forward = np.asarray(potential * _column_divergence(other_x, other_y, params)) * area
    backward = np.asarray(other * _column_divergence(gradient_x, gradient_y, params)) * area
    assert abs(forward.sum() - backward.sum()) <= 5e-13 * (np.abs(forward).sum() + np.abs(backward).sum())
    assert np.sum(np.asarray(potential * _column_divergence(gradient_x, gradient_y, params)) * area) <= 0.


@pytest.mark.parametrize("land", [False, True])
@pytest.mark.parametrize("preconditioner", ["none", "jacobi"])
def test_projection_removes_native_transport_without_increasing_energy(land, preconditioner):
    _, params, volume = _parameters(65., land=land)
    params = params._replace(projection_preconditioner=preconditioner)
    velocity_x, velocity_y = _velocities(params)
    corrected_x, corrected_y = jax.jit(lambda velocity_x, velocity_y: _project_column_divergence(
        velocity_x, velocity_y, params, params.dt, n_iter=1000))(velocity_x, velocity_y)
    before = np.asarray(_vertical_transport_iface(velocity_x, velocity_y, params)[..., 0])
    after = np.asarray(_vertical_transport_iface(corrected_x, corrected_y, params)[..., 0])
    assert np.linalg.norm(after) <= 1e-9 * np.linalg.norm(before)
    energy_before = np.sum(np.asarray(velocity_x ** 2 + velocity_y ** 2) * volume)
    energy_after = np.sum(np.asarray(corrected_x ** 2 + corrected_y ** 2) * volume)
    assert energy_after <= energy_before * (1. + 1e-12)


def test_dry_velocity_sentinels_do_not_affect_wet_projection():
    _, params, _ = _parameters(65., land=True)
    velocity_x, velocity_y = _velocities(params)
    project = jax.jit(lambda velocity_x, velocity_y: _project_column_divergence(
        velocity_x, velocity_y, params, params.dt, n_iter=1000))
    first = project(velocity_x, velocity_y)
    dry = 1. - params.wet_mask_z
    second = project(velocity_x + 100. * dry, velocity_y - 100. * dry)
    wet = np.asarray(params.wet_mask_z) > 0.5
    for original, altered in zip(first, second, strict=True):
        np.testing.assert_array_equal(np.asarray(original)[wet], np.asarray(altered)[wet])


def test_zero_rhs_projection_is_finite_zero():
    _, params, _ = _parameters(65., land=True)
    zero = jnp.zeros_like(params.wet_mask_z)
    corrected = jax.jit(lambda velocity_x, velocity_y: _project_column_divergence(
        velocity_x, velocity_y, params, params.dt, n_iter=1000))(zero, zero)
    for field in corrected:
        np.testing.assert_array_equal(field, 0.)


def test_projected_velocity_preserves_constant_tracer_pointwise():
    _, params, _ = _parameters(65., land=True)
    corrected_x, corrected_y = _project_column_divergence(*_velocities(params), params, params.dt, n_iter=1000)
    transport = _vertical_transport_iface(corrected_x, corrected_y, params)
    constant = jnp.full_like(params.wet_mask_z, 15.)
    tendency = _advection_scalar(constant, corrected_x, corrected_y, transport, params)
    assert np.max(np.abs(tendency)) < 1e-15


def test_projection_jvp_matches_independent_finite_difference():
    _, params, _ = _parameters(65., land=True, nx=12, ny=12)
    velocities = _velocities(params)
    direction = _velocities(params, 3141)
    project = jax.jit(lambda velocity_x, velocity_y: _project_column_divergence(
        velocity_x, velocity_y, params, params.dt, n_iter=1000))
    _, tangent = jax.jvp(project, velocities, direction)
    epsilon = 1e-4
    positive = project(*(velocity + epsilon * delta for velocity, delta in zip(velocities, direction, strict=True)))
    negative = project(*(velocity - epsilon * delta for velocity, delta in zip(velocities, direction, strict=True)))
    for actual, plus, minus in zip(tangent, positive, negative, strict=True):
        np.testing.assert_allclose(actual, (plus - minus) / (2. * epsilon), rtol=5e-7, atol=5e-8)


def test_float32_projection_resolves_native_transport_with_dtype_tolerance():
    _, params, _ = _parameters(65., land=True)
    params = params._replace(**{name: value.astype(jnp.float32) for name, value in params._asdict().items()
                              if isinstance(value, jnp.ndarray) and jnp.issubdtype(value.dtype, jnp.floating)})
    velocities = tuple(field.astype(jnp.float32) for field in _velocities(params))
    corrected = jax.jit(lambda velocity_x, velocity_y: _project_column_divergence(
        velocity_x, velocity_y, params, params.dt, n_iter=1000))(*velocities)
    before = np.asarray(_vertical_transport_iface(*velocities, params)[..., 0])
    after = np.asarray(_vertical_transport_iface(*corrected, params)[..., 0])
    assert np.linalg.norm(after) <= 2e-5 * np.linalg.norm(before)
    assert all(field.dtype == jnp.float32 for field in corrected)


def test_actual_projection_monitor_does_not_change_state_or_external_sources():
    from test_nonlinear_budgets import _state

    _, params, _ = _parameters(65., land=True)
    params = params._replace(kappa_h=0., dt=1., project_adv_vel=True)
    state = _state(params)
    normal = jax.jit(lambda current: _step_impl(current, params))(state)
    audited, report = make_budget_step(params)(state)
    for actual, expected in zip(audited, normal, strict=True):
        np.testing.assert_array_equal(actual, expected)
    norms = np.asarray(report["projection_transport_norm_squared"])
    assert norms[0] > 0. and 0. <= norms[1] < norms[0]
    np.testing.assert_array_equal(report["source_inputs"], 0.)
