"""Actual wet-face residuals, not recursively estimated Krylov residuals."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_horizontal_tracer_diffusion import _parameters

from jax_solver_global import (
    _gradient_conservative_3d,
    _project_column_divergence,
    _vertical_transport_iface,
    make_solver_global,
    projection_config,
)


def _smooth_predictor():
    _, params, volume = _parameters(65., land=True, nx=128, ny=96)
    params = params._replace(**{name: value.astype(jnp.float32)
                              for name, value in params._asdict().items()
                              if isinstance(value, jnp.ndarray) and jnp.issubdtype(value.dtype, jnp.floating)})
    params = params._replace(projection_niter=600, projection_preconditioner="jacobi")
    phase_x = jnp.arange(params.nx, dtype=jnp.float32)[:, None] * 2. * jnp.pi / params.nx
    phase_y = jnp.arange(params.ny, dtype=jnp.float32)[None, :] * jnp.pi / (params.ny - 1)
    potential = jnp.sin(phase_x) * jnp.cos(phase_y) * 1e6
    velocities = _gradient_conservative_3d(potential[..., None], params)
    return params, velocities, volume


def _native_norm(velocities, params):
    transport = np.asarray(_vertical_transport_iface(*velocities, params)[..., 0], dtype=np.float64)
    area = np.asarray(params.dx_2d, dtype=np.float64) * params.dy * np.asarray(params.wet_mask)
    return np.sqrt(np.sum(transport ** 2 * area))


def test_float32_smooth_pressure_projection_closes_actual_transport():
    params, velocities, volume = _smooth_predictor()
    corrected = jax.jit(lambda velocity_x, velocity_y: _project_column_divergence(
        velocity_x, velocity_y, params, params.dt))(*velocities)
    assert _native_norm(corrected, params) <= 5e-5 * _native_norm(velocities, params)
    energy_before = sum(np.sum(np.asarray(field, dtype=np.float64) ** 2 * volume) for field in velocities)
    energy_after = sum(np.sum(np.asarray(field, dtype=np.float64) ** 2 * volume) for field in corrected)
    assert energy_after <= energy_before * (1. + 5e-6)
    assert all(field.dtype == jnp.float32 for field in corrected)


def test_disabling_refinement_retains_the_known_failure():
    params, velocities, _ = _smooth_predictor()
    original = params._replace(projection_max_refinements=0)
    uncorrected = jax.jit(lambda velocity_x, velocity_y: _project_column_divergence(
        velocity_x, velocity_y, original, original.dt))(*velocities)
    assert _native_norm(uncorrected, original) > 5e-5 * _native_norm(velocities, original)
    once = params._replace(projection_max_refinements=1)
    corrected = jax.jit(lambda velocity_x, velocity_y: _project_column_divergence(
        velocity_x, velocity_y, once, once.dt))(*velocities)
    assert _native_norm(corrected, once) <= 5e-5 * _native_norm(velocities, once)


@pytest.mark.parametrize("limit", [-1, 3, True, 1.5, np.nan])
def test_invalid_refinement_limits_fail_at_construction(limit):
    from config import PhysicsConfig

    grid, _, _ = _parameters(nx=12, ny=12)
    with pytest.raises(ValueError, match="projection_max_refinements"):
        make_solver_global(grid, PhysicsConfig(), 60., projection_max_refinements=limit)


def test_refinement_settings_record_effective_dtype_and_scale():
    params, _, _ = _smooth_predictor()
    settings = projection_config(params)
    assert settings["max_refinements"] == 2
    assert settings["transport_rtol"] == 32. * np.finfo(np.float32).eps
    assert settings["refinement_atol_scale"] == "original_rhs_l2"


def test_active_float32_refinement_has_consistent_jvp_and_vjp():
    params, velocities, volume = _smooth_predictor()
    random = np.random.default_rng(3141)
    direction = tuple(jnp.asarray(random.normal(size=volume.shape), dtype=jnp.float32) * params.wet_mask_z
                      for component in range(2))
    cotangent = tuple(jnp.asarray(random.normal(size=volume.shape), dtype=jnp.float32) * params.wet_mask_z
                      for component in range(2))
    project = jax.jit(lambda velocity_x, velocity_y: _project_column_divergence(
        velocity_x, velocity_y, params, params.dt))
    _, tangent = jax.jvp(project, velocities, direction)
    epsilon = 1e-2
    positive = project(*(field + epsilon * delta for field, delta in zip(velocities, direction, strict=True)))
    negative = project(*(field - epsilon * delta for field, delta in zip(velocities, direction, strict=True)))
    difference = tuple((plus - minus) / (2. * epsilon)
                       for plus, minus in zip(positive, negative, strict=True))
    error = sum(np.sum((np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)) ** 2 * volume)
                for actual, expected in zip(tangent, difference, strict=True))
    scale = sum(np.sum(np.asarray(field, dtype=np.float64) ** 2 * volume) for field in difference)
    assert np.sqrt(error / scale) < 5e-4
    _, pullback = jax.vjp(project, *velocities)
    reverse = pullback(cotangent)
    forward_product = sum(float(jnp.sum(first.astype(jnp.float64) * second))
                          for first, second in zip(tangent, cotangent, strict=True))
    reverse_product = sum(float(jnp.sum(first.astype(jnp.float64) * second))
                          for first, second in zip(direction, reverse, strict=True))
    normalization = sum(np.linalg.norm(np.asarray(first, dtype=np.float64)) * np.linalg.norm(np.asarray(second, dtype=np.float64))
                        for first, second in zip(tangent, cotangent, strict=True))
    assert abs(forward_product - reverse_product) <= 5e-4 * normalization
