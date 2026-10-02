"""Immutable solve settings and independently assembled Jacobi diagonal."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from config import PhysicsConfig
from jax_solver_global import (
    _column_divergence,
    _column_projection_diagonal,
    _gradient_conservative_3d,
    _project_column_divergence,
    make_solver_global,
    projection_config,
)
from stage_budgets import accumulate_budget, empty_budget
from tests.support.fd.column_projection import _velocities
from tests.support.fd.horizontal_diffusion import _parameters


def _build(grid, **settings):
    physics = replace(PhysicsConfig(), nu_h=0., nu_bi=0., kappa_conv=0.)
    return make_solver_global(grid, physics, 60., return_params=True, **settings)[3]


@pytest.mark.parametrize("land", [False, True])
def test_jacobi_diagonal_matches_independent_dense_operator(land):
    _, params, _ = _parameters(65., land=land, nx=18, ny=12)
    area = params.dx_2d * params.dy * params.wet_mask
    def operator(potential):
        gradient = _gradient_conservative_3d(potential[..., None], params)
        return -area * _column_divergence(*gradient, params)

    basis = jnp.eye(params.nx * params.ny).reshape(-1, params.nx, params.ny)
    matrix = np.asarray(jax.jit(jax.vmap(operator))(basis)).reshape(len(basis), -1)
    diagonal = np.asarray(_column_projection_diagonal(params)).ravel()
    np.testing.assert_allclose(diagonal, np.diag(matrix), rtol=5e-13, atol=1e-14)
    assert np.all(diagonal >= 0.)


def test_jacobi_is_safe_for_dry_and_isolated_rows():
    grid, _, _ = _parameters(65., nx=12, ny=12)
    wet = np.zeros_like(grid.wet_mask_3d)
    wet[3, 4, :] = 1.
    grid = replace(grid, wet_mask_3d=wet, wet_mask=wet[..., 0])
    params = _build(grid, projection_preconditioner="jacobi")
    np.testing.assert_array_equal(_column_projection_diagonal(params), 0.)
    assert np.all(np.isfinite(params.projection_inv_diagonal))
    np.testing.assert_array_equal(params.projection_inv_diagonal, 1.)
    zero = jnp.zeros_like(params.wet_mask_z)
    result = _project_column_divergence(zero, zero, params, params.dt)
    for field in result:
        np.testing.assert_array_equal(field, 0.)


def test_projection_settings_are_resolved_before_trace(monkeypatch):
    grid, _, _ = _parameters(65., nx=12, ny=12)
    monkeypatch.setenv("OCEAN_PAV_NITER", "700")
    params = _build(grid, projection_preconditioner="jacobi")
    assert params.projection_niter == 700
    assert projection_config(params)["niter_source"] == "environment:OCEAN_PAV_NITER"
    velocities = _velocities(params)
    monkeypatch.setenv("OCEAN_PAV_NITER", "not-an-integer")
    actual = jax.jit(lambda velocity_x, velocity_y: _project_column_divergence(
        velocity_x, velocity_y, params, params.dt))(*velocities)
    expected = _project_column_divergence(*velocities, params, params.dt, n_iter=700)
    for first, second in zip(actual, expected, strict=True):
        np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)
    explicit = _build(grid, projection_niter=300)
    assert projection_config(explicit)["niter_source"] == "explicit"
    assert explicit.projection_niter == 300


@pytest.mark.parametrize("settings", [
    {"projection_niter": 0}, {"projection_niter": -1}, {"projection_niter": 1.5},
    {"projection_niter": True}, {"projection_rtol": 0.}, {"projection_rtol": -1.},
    {"projection_rtol": np.nan}, {"projection_rtol": np.inf},
    {"projection_preconditioner": "not-a-preconditioner"},
])
def test_invalid_projection_settings_fail_at_construction(settings):
    grid, _, _ = _parameters(nx=12, ny=12)
    with pytest.raises(ValueError, match="projection"):
        _build(grid, **settings)


def test_invalid_legacy_environment_fails_at_construction(monkeypatch):
    grid, _, _ = _parameters(nx=12, ny=12)
    monkeypatch.setenv("OCEAN_PAV_NITER", "1.5")
    with pytest.raises(ValueError, match="projection"):
        _build(grid)


def test_float32_effective_tolerance_and_diagonal_are_recorded(monkeypatch):
    monkeypatch.delenv("OCEAN_PAV_NITER", raising=False)
    grid, _, _ = _parameters(nx=12, ny=12)
    params = _build(grid, dtype="float32", projection_rtol=1e-12,
                    projection_preconditioner="jacobi")
    settings = projection_config(params)
    assert settings["rtol"] == 32. * np.finfo(np.float32).eps
    assert settings["niter"] == 150
    assert settings["niter_source"] == "default"
    assert settings["preconditioner"] == "jacobi"
    assert params.projection_inv_diagonal.dtype == jnp.float32


def test_peak_projection_residual_is_maximized_not_summed():
    first, second = empty_budget(), empty_budget()
    first["projection_relative_residual_max"] = jnp.asarray(0.25)
    second["projection_relative_residual_max"] = jnp.asarray(0.5)
    first["projection_transport_norm_squared"] = jnp.asarray([4., 1.])
    second["projection_transport_norm_squared"] = jnp.asarray([6., 2.])
    combined = accumulate_budget(first, second)
    assert combined["projection_relative_residual_max"] == 0.5
    np.testing.assert_array_equal(combined["projection_transport_norm_squared"], [10., 3.])


def test_jacobi_jvp_vjp_and_volume_adjoint_match_converged_projection():
    _, params, volume = _parameters(65., land=True, nx=12, ny=12)
    params = params._replace(projection_preconditioner="jacobi", projection_niter=1000)
    velocities = _velocities(params)
    direction = _velocities(params, 3141)
    project = jax.jit(lambda velocity_x, velocity_y: _project_column_divergence(
        velocity_x, velocity_y, params, params.dt))
    projected, tangent = jax.jvp(project, velocities, direction)
    epsilon = 1e-4
    positive = project(*(velocity + epsilon * delta for velocity, delta in zip(velocities, direction, strict=True)))
    negative = project(*(velocity - epsilon * delta for velocity, delta in zip(velocities, direction, strict=True)))
    for actual, plus, minus in zip(tangent, positive, negative, strict=True):
        np.testing.assert_allclose(actual, (plus - minus) / (2. * epsilon), rtol=5e-7, atol=5e-8)
    cotangent = tuple(field * volume for field in velocities)
    _, pullback = jax.vjp(project, *velocities)
    reverse = pullback(cotangent)
    for actual, expected in zip(reverse, projected, strict=True):
        np.testing.assert_allclose(actual, expected * volume, rtol=5e-8, atol=1e-7 * np.max(volume))
    forward_product = sum(float(jnp.sum(first * second)) for first, second in zip(tangent, cotangent, strict=True))
    reverse_product = sum(float(jnp.sum(first * second)) for first, second in zip(direction, reverse, strict=True))
    assert forward_product == pytest.approx(reverse_product, rel=5e-10)


def test_actual_monitor_detects_iteration_cap_failure():
    from stage_budgets import make_budget_step
    from tests.support.fd.nonlinear_budgets import _state

    _, params, _ = _parameters(65., land=True)
    params = params._replace(kappa_h=0., dt=1., project_adv_vel=True, projection_niter=1)
    _, report = make_budget_step(params)(_state(params))
    assert float(report["projection_relative_residual_max"]) > 1e-3
