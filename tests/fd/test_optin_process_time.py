"""Independent M3 process oracles; isolated order is not whole-model order."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.support.fd.process_time import (
    _candidate,
    _inertial_source_path,
    _numpy_vertical_matrix,
)
from tests.support.fd.reference_geometry import WIDTHS, _fixture
from zhenmode.model.config.definitions import C_P, G_EARTH, RHO_0, PhysicsConfig
from zhenmode.model.dynamics.barotropic import _free_surface_step_fd
from zhenmode.model.dynamics.pressure import _reference_depth_gradient
from zhenmode.model.dynamics.processes import _compute_tracer_residual, _rotate_baroclinic_shear
from zhenmode.model.dynamics.transport import _barotropic_velocity
from zhenmode.model.factory import make_solver_global
from zhenmode.model.timestepping.integration import _step_impl


@pytest.mark.parametrize("subcycles", [1, 2, 4, 8])
def test_convection_retains_physical_coefficient_in_every_substep(subcycles):
    _, (_, initialize, _, params, _) = _candidate()
    params = params._replace(kappa_conv=0.01, conv_nsub=subcycles)
    state = initialize()._replace(T=jnp.broadcast_to(jnp.arange(4., dtype=jnp.float64), (8, 8, 4)))
    operator = 0.01 * _numpy_vertical_matrix()
    expected = np.arange(4., dtype=float)
    expected_tendency = np.zeros(4)
    for _ in range(subcycles):
        current_tendency = operator @ expected
        expected_tendency += current_tendency / subcycles
        expected += params.dt / subcycles * current_tendency
    tendency, salt_tendency = _compute_tracer_residual(state, params)
    np.testing.assert_allclose(tendency, np.broadcast_to(expected_tendency, state.T.shape),
                               rtol=1e-12, atol=1e-17)
    np.testing.assert_allclose(salt_tendency, 0., rtol=0., atol=1e-17)
    np.testing.assert_allclose(np.sum(np.asarray(tendency) * WIDTHS, axis=-1), 0., atol=1e-17)

def test_legacy_convection_remains_an_explicit_negative_control():
    _, (_, initialize, _, params, _) = _candidate()
    params = params._replace(kappa_conv=0.01, conv_nsub=2, process_time_scheme="legacy")
    state = initialize()._replace(T=jnp.broadcast_to(jnp.arange(4., dtype=jnp.float64), (8, 8, 4)))
    operator = 0.005 * _numpy_vertical_matrix()
    expected = np.arange(4., dtype=float)
    for _ in range(2):
        expected += params.dt / 2. * (operator @ expected)
    actual = np.asarray(_compute_tracer_residual(state, params)[0])[0, 0]
    np.testing.assert_allclose(actual, (expected - np.arange(4.)) / params.dt, atol=1e-17)
    physical = _compute_tracer_residual(state, params._replace(process_time_scheme="consistent_split_v1"))[0]
    assert 0.49 < actual[0] / float(physical[0, 0, 0]) < 0.51

@pytest.mark.parametrize("stairs", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_slow_rotation_preserves_wet_mean_and_rotates_only_shear(stairs, dtype):
    _, (_, initialize, _, params, _) = _candidate(stairs=stairs, dtype=dtype)
    params = params._replace(f=jnp.full_like(params.f, 1e-3))
    generator = np.random.default_rng(329)
    state = initialize()
    velocity_x = jnp.asarray(generator.normal(size=state.u.shape), dtype=state.u.dtype)
    velocity_y = jnp.asarray(generator.normal(size=state.v.shape), dtype=state.v.dtype)
    wet = np.asarray(params.wet_mask_z)
    weights = wet * WIDTHS
    total_width = np.maximum(np.sum(weights, axis=-1), 1.)
    mean_x = np.sum(np.asarray(velocity_x) * weights, axis=-1) / total_width
    mean_y = np.sum(np.asarray(velocity_y) * weights, axis=-1) / total_width
    duration = 123.
    angle = np.asarray(params.f)[..., None] * duration
    shear_x = (np.asarray(velocity_x) - mean_x[..., None]) * wet
    shear_y = (np.asarray(velocity_y) - mean_y[..., None]) * wet
    expected_x = np.asarray(velocity_x) + np.cos(angle) * shear_x + np.sin(angle) * shear_y - shear_x
    expected_y = np.asarray(velocity_y) - np.sin(angle) * shear_x + np.cos(angle) * shear_y - shear_y
    actual_x, actual_y = _rotate_baroclinic_shear(velocity_x, velocity_y, params, duration)
    tolerance = 3e-7 if dtype == "float32" else 2e-15
    np.testing.assert_allclose(actual_x, expected_x, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual_y, expected_y, rtol=tolerance, atol=tolerance)
    actual_mean = _barotropic_velocity(actual_x, actual_y, params)
    np.testing.assert_allclose(actual_mean[0], mean_x, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual_mean[1], mean_y, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(np.asarray(actual_x)[wet == 0], np.asarray(velocity_x)[wet == 0])
    np.testing.assert_array_equal(np.asarray(actual_y)[wet == 0], np.asarray(velocity_y)[wet == 0])

def test_inertial_source_is_counted_once_and_midpoint_phase_is_second_order():
    _, (_, initialize, _, params, _) = _candidate()
    state = initialize()._replace(u=jnp.ones((8, 8, 4)), v=jnp.zeros((8, 8, 4)))
    duration = 400.
    errors = []
    for subcycles in (2, 4, 8):
        actual = _inertial_source_path(params, state, duration, subcycles)
        angle = 2. * subcycles * np.arctan(1e-3 * duration / (2. * subcycles))
        np.testing.assert_allclose(actual.u, np.cos(angle), atol=1e-14, rtol=0.)
        np.testing.assert_allclose(actual.v, -np.sin(angle), atol=1e-14, rtol=0.)
        np.testing.assert_allclose(np.asarray(actual.u) ** 2 + np.asarray(actual.v) ** 2, 1., atol=1e-14, rtol=0.)
        errors.append(abs(angle - 1e-3 * duration))
    assert all(3.9 < coarse / fine < 4.1 for coarse, fine in zip(errors[:-1], errors[1:], strict=True))
    legacy = _inertial_source_path(params._replace(process_time_scheme="legacy"), state, duration, 8)
    legacy_angle = -np.arctan2(float(legacy.v[0, 0, 0]), float(legacy.u[0, 0, 0]))
    assert 0.79 < legacy_angle < 0.81
    assert float(legacy.u[0, 0, 0] ** 2 + legacy.v[0, 0, 0] ** 2) < 0.99

def test_midpoint_fast_step_preserves_discrete_geostrophic_equilibrium():
    _, (_, initialize, _, params, _) = _candidate()
    params = params._replace(f=jnp.full_like(params.f, 1e-4))
    eta = jnp.broadcast_to(jnp.array([0., 0.02, 0.08, 0.13, 0.09, 0.03, 0.01, 0.]), (8, 8))
    gradient_x, gradient_y = _reference_depth_gradient(eta, params)
    mean_x, mean_y = -G_EARTH * gradient_y / params.f, G_EARTH * gradient_x / params.f
    actual_eta, actual_x, actual_y = _free_surface_step_fd(eta, mean_x, mean_y, params, dt_half=200.)
    np.testing.assert_allclose(actual_eta, eta, atol=1e-16, rtol=0.)
    np.testing.assert_allclose(actual_x, mean_x, atol=1e-16, rtol=0.)
    np.testing.assert_allclose(actual_y, mean_y, atol=1e-16, rtol=0.)

@pytest.mark.parametrize("outer_dt", [4., 2., 1.])
def test_actual_step_matches_independent_linear_forward_backward_wave(outer_dt):
    _, (_, initialize, _, params, _) = _candidate(use_scan=True)
    params = params._replace(dt=outer_dt, dt_bt=outer_dt / 2., n_subcyc=2,
                             dx_2d=jnp.full_like(params.dx_2d, 1000.), dy=1000.,
                             inv_dx=jnp.full_like(params.inv_dx, 0.001), inv_dy=0.001,
                             inv_dx2=jnp.full_like(params.inv_dx2, 1e-6), inv_dy2=1e-6,
                             cos_lat=jnp.ones_like(params.cos_lat), adv_nsub=1)
    amplitude = 1e-7
    wave = 2. * np.pi * np.arange(8) / 8.
    state = initialize()._replace(eta=jnp.broadcast_to(jnp.asarray(amplitude * np.cos(wave))[:, None], (8, 8)))
    steps = int(64. / outer_dt)
    advance = jax.jit(lambda initial: jax.lax.fori_loop(0, steps, lambda index, current: _step_impl(current, params), initial))
    actual = advance(state)
    eta = np.asarray(state.eta).copy()
    velocity_x = np.zeros_like(eta)
    for _ in range(steps * 2):
        divergence = 50. * (np.roll(velocity_x, -1, axis=0) - np.roll(velocity_x, 1, axis=0)) / 2000.
        eta -= params.dt_bt * divergence
        gradient = (np.roll(eta, -1, axis=0) - np.roll(eta, 1, axis=0)) / 2000.
        velocity_x -= params.dt_bt * G_EARTH * gradient
    np.testing.assert_allclose(actual.eta, eta, atol=2e-15, rtol=1e-7)
    np.testing.assert_allclose(actual.u, np.broadcast_to(velocity_x[..., None], actual.u.shape), atol=2e-15, rtol=1e-7)

def test_shear_rotation_gradient_matches_finite_difference():
    _, (_, initialize, _, params, _) = _candidate(stairs=True)
    params = params._replace(f=jnp.full_like(params.f, 1e-3))
    state = initialize()
    generator = np.random.default_rng(993)
    velocity = jnp.asarray(generator.normal(size=state.u.shape))
    direction = jnp.asarray(generator.normal(size=state.u.shape))
    probe = jnp.asarray(generator.normal(size=state.u.shape))

    def objective(values):
        rotated, _ = _rotate_baroclinic_shear(values, state.v, params, 17.)
        return jnp.sum(rotated * probe)

    derivative = jnp.vdot(jax.grad(objective)(velocity), direction)
    increment = 1e-5
    reference = (objective(velocity + increment * direction) - objective(velocity - increment * direction)) / (2. * increment)
    np.testing.assert_allclose(derivative, reference, rtol=2e-10, atol=2e-9)

@pytest.mark.parametrize("scan", [False, True])
def test_actual_candidate_step_wind_and_heat_are_counted_once(scan):
    forcing = (np.full((8, 8), 0.05), np.zeros((8, 8)), np.full((8, 8), 100.))
    _, (step, initialize, _, _, _) = _candidate(forcing=forcing, use_scan=scan)
    before = initialize()
    after = step(before)
    momentum = np.sum(np.asarray(after.u - before.u) * WIDTHS, axis=-1) * RHO_0
    heat = np.sum(np.asarray(after.T - before.T) * WIDTHS, axis=-1) * RHO_0 * C_P
    np.testing.assert_allclose(momentum, 0.05 * 10., rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(heat, 100. * 10., rtol=1e-11, atol=1e-6)

def test_short_actual_step_trajectory_gradient_matches_finite_difference():
    _, (step, initialize, _, params, _) = _candidate(stairs=True, use_scan=True)
    state = initialize()
    generator = np.random.default_rng(763)
    direction = jnp.asarray(generator.normal(size=state.T.shape)) * params.wet_mask_z
    probe = jnp.asarray(generator.normal(size=state.T.shape)) * params.wet_mask_z

    @jax.jit
    def objective(scale):
        current = state._replace(T=state.T + scale * direction)
        for _ in range(3):
            current = step(current)
        return jnp.sum(current.T * probe) + 0.01 * jnp.sum(current.eta)

    center = 0.02
    derivative = jax.grad(objective)(center)
    increment = 1e-4
    reference = (objective(center + increment) - objective(center - increment)) / (2. * increment)
    assert np.isfinite(float(derivative))
    np.testing.assert_allclose(derivative, reference, rtol=1e-7, atol=1e-7)

@pytest.mark.parametrize("scheme,match", [("unknown", True), ("consistent_split_v1", False)])
def test_process_scheme_rejects_invalid_or_unmatched_construction(scheme, match):
    grid, _ = _fixture()
    with pytest.raises(ValueError, match="process_time_scheme|consistent_split_v1"):
        make_solver_global(grid, PhysicsConfig(), 10.,
                           column_geometry="nodal_dual_v1", mode_split=True,
                           conservative_kv=True, localize_conv=True,
                           match_barotropic_transport=match, process_time_scheme=scheme)
