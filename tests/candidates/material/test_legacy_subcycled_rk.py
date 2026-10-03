"""Independent nonlinear time oracles, not a whole-model second-order claim."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocean_solver.audit.stages import _StageRecorder, make_budget_step
from ocean_solver.config.definitions import C_P, RHO_0
from ocean_solver.timestepping.integration import _step_impl, _tracer_step_with_transport
from tests.support.material.process_time import _numpy_vertical_matrix
from tests.support.material.reference_geometry import WIDTHS, _fixture
from tests.support.material.subcycled_rk import (
    _candidate,
    _transport_case,
    advection_bulk_order,
    convection_order,
    quadratic_drag_order,
)


@pytest.mark.parametrize("oracle,error_field", [(convection_order, "rms_K"),
                                               (quadratic_drag_order, "error_m_per_s")])
def test_actual_nonlinear_process_second_order_and_v1_negative_control(oracle, error_field):
    candidate = oracle("subcycled_rk2_v2")
    control = oracle("consistent_split_v1")
    candidate_ratios = [coarse[error_field] / fine[error_field]
                        for coarse, fine in zip(candidate[:-1], candidate[1:], strict=True)]
    control_ratios = [coarse[error_field] / fine[error_field]
                      for coarse, fine in zip(control[:-1], control[1:], strict=True)]
    assert all(ratio >= 3.3 for ratio in candidate_ratios), candidate_ratios
    assert all(1.5 < ratio < 2.8 for ratio in control_ratios), control_ratios

def test_fixed_transport_advection_and_bulk_match_independent_fourier_exact_solution():
    rows = advection_bulk_order()
    ratios = [coarse["rms_K"] / fine["rms_K"] for coarse, fine in zip(rows[:-1], rows[1:], strict=True)]
    assert all(ratio >= 3.3 for ratio in ratios), ratios

@pytest.mark.parametrize("subcycles", [1, 2, 4, 8])
@pytest.mark.parametrize("scan", [False, True])
def test_convection_matches_numpy_heun_with_physical_coefficient_and_preserves_inventory(subcycles, scan):
    _, (_, initialize, _, params, _) = _candidate(use_scan=scan)
    params = params._replace(kappa_conv=0.1, conv_nsub=subcycles)
    initial_profile = np.arange(4., dtype=float)
    reference = initial_profile.copy()
    operator = 0.1 * _numpy_vertical_matrix()
    for _ in range(subcycles):
        first = operator @ reference
        second = operator @ (reference + params.dt / subcycles * first)
        reference += 0.5 * params.dt / subcycles * (first + second)
    initial = initialize()._replace(T=jnp.broadcast_to(jnp.asarray(initial_profile), (8, 8, 4)))
    actual = _step_impl(initial, params)
    np.testing.assert_allclose(actual.T, np.broadcast_to(reference, initial.T.shape), atol=2e-15, rtol=1e-14)
    np.testing.assert_allclose(np.sum(np.asarray(actual.T - initial.T) * WIDTHS, axis=-1), 0., atol=2e-14)
    assert np.sum(WIDTHS * (reference - np.sum(WIDTHS * reference) / 50.) ** 2) <= np.sum(
        WIDTHS * (initial_profile - np.sum(WIDTHS * initial_profile) / 50.) ** 2)

@pytest.mark.parametrize("scan", [False, True])
def test_actual_substep_sources_and_transport_accounting_match_numpy_heun(scan):
    initial, params, faces = _transport_case(400., subcycles=3)
    params = params._replace(use_scan=scan)
    reference = np.asarray(initial.T).copy()
    bulk_total = np.zeros_like(reference)
    rate = 400. / (RHO_0 * C_P * WIDTHS[0])

    def rhs(temperature):
        advection = -0.2 / 1000. * (temperature - np.roll(temperature, 1, axis=0))
        bulk = np.zeros_like(temperature)
        bulk[..., 0] = rate * (15. - temperature[..., 0])
        return advection + bulk, bulk

    for _ in range(3):
        first, first_bulk = rhs(reference)
        second, second_bulk = rhs(reference + params.dt / 3. * first)
        reference += 0.5 * params.dt / 3. * (first + second)
        bulk_total += 0.5 * params.dt / 3. * (first_bulk + second_bulk)
    recorder = _StageRecorder(params)
    actual = jax.jit(lambda state: _tracer_step_with_transport(state, params, faces))(initial)
    audited = _tracer_step_with_transport(initial, params, faces, budget=recorder)
    np.testing.assert_allclose(actual.T, reference, atol=4e-15, rtol=1e-15)
    np.testing.assert_allclose(audited.T, actual.T, atol=4e-15, rtol=1e-15)
    expected_heat = np.sum(bulk_total * WIDTHS * 1e6) * RHO_0 * C_P
    np.testing.assert_allclose(recorder.sources["bulk_heat"][0], expected_heat, rtol=1e-13, atol=1.)
    np.testing.assert_allclose(recorder.tracer_face_mean[0], 10., rtol=0., atol=2e-15)
    np.testing.assert_allclose(recorder.tracer_face_mean[1], 0., rtol=0., atol=0.)
    np.testing.assert_allclose(recorder.advection_boundary, 0., rtol=0., atol=0.)
    observed = float(recorder.difference(initial, audited)[0][0])
    np.testing.assert_allclose(observed, expected_heat, rtol=2e-12, atol=2.)

@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("subcycles", [1, 3])
def test_actual_step_counts_heat_wind_once_and_audit_does_not_change_state(scan, subcycles):
    forcing = (np.full((8, 8), 0.05), np.zeros((8, 8)), np.full((8, 8), 100.))
    _, (_, initialize, _, params, _) = _candidate(forcing=forcing, use_scan=scan)
    params = params._replace(adv_nsub=subcycles, conv_nsub=subcycles,
                             dx_2d=jnp.full_like(params.dx_2d, 1000.), dy=1000.,
                             inv_dx=jnp.full_like(params.inv_dx, 0.001), inv_dy=0.001,
                             inv_dx2=jnp.full_like(params.inv_dx2, 1e-6), inv_dy2=1e-6,
                             cos_lat=jnp.ones_like(params.cos_lat))
    step = jax.jit(lambda state: _step_impl(state, params))
    before = initialize()._replace(T=jnp.zeros((8, 8, 4)))
    actual = step(before)
    audited, ledger = make_budget_step(params)(before)
    for normal_field, audited_field in zip(actual, audited, strict=True):
        np.testing.assert_allclose(normal_field, audited_field, atol=4e-15, rtol=1e-15)
    momentum = np.sum(np.asarray(actual.u - before.u) * WIDTHS, axis=-1) * RHO_0
    heat = np.sum(np.asarray(actual.T - before.T) * WIDTHS, axis=-1) * RHO_0 * C_P
    np.testing.assert_allclose(momentum, 0.05 * 10., rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(heat, 100. * 10., rtol=1e-11, atol=1e-6)
    np.testing.assert_allclose(ledger["budget_residual"], 0., rtol=0., atol=0.1)
    assert float(ledger["transport_consistency_max"][2]) < 1e-12

@pytest.mark.parametrize("scan", [False, True])
def test_dry_nodes_are_held_and_constant_tracers_remain_constant(scan):
    _, (_, initialize, _, params, _) = _candidate(stairs=True, use_scan=scan)
    params = params._replace(adv_nsub=3, conv_nsub=2)
    step = jax.jit(lambda state: _step_impl(state, params))
    initial = initialize()._replace(T=jnp.full((8, 8, 4), 15.), S=jnp.full((8, 8, 4), 35.))
    actual = step(initial)
    for actual_field, initial_field in zip(actual, initial, strict=True):
        np.testing.assert_array_equal(actual_field, initial_field)
    random = np.random.default_rng(454)
    wet = np.asarray(params.wet_mask_z)
    profile = np.array([18., 16., 14., 12.])[None, None, :]
    initial = initial._replace(T=jnp.asarray(profile + random.normal(scale=0.01, size=initial.T.shape)))
    actual = step(initial)
    for actual_field, initial_field in zip(actual[:4], initial[:4], strict=True):
        np.testing.assert_array_equal(np.asarray(actual_field)[wet == 0], np.asarray(initial_field)[wet == 0])

def test_short_actual_trajectory_gradient_matches_finite_difference():
    _, (_, initialize, _, params, _) = _candidate(stairs=True, use_scan=True)
    params = params._replace(adv_nsub=3, conv_nsub=2)
    step = jax.jit(lambda state: _step_impl(state, params))
    state = initialize()
    generator = np.random.default_rng(765)
    direction = jnp.asarray(generator.normal(scale=0.01, size=state.T.shape)) * params.wet_mask_z
    probe = jnp.asarray(generator.normal(size=state.T.shape)) * params.wet_mask_z

    @jax.jit
    def objective(scale):
        current = state._replace(T=state.T + scale * direction)
        for _ in range(3):
            current = step(current)
        return jnp.sum(current.T * probe) + 0.01 * jnp.sum(current.eta)

    derivative = jax.grad(objective)(0.02)
    increment = 1e-4
    reference = (objective(0.02 + increment) - objective(0.02 - increment)) / (2. * increment)
    assert np.isfinite(float(derivative))
    np.testing.assert_allclose(derivative, reference, rtol=2e-7, atol=2e-7)

def test_new_scheme_requires_matched_barotropic_transport():
    with pytest.raises(ValueError, match="subcycled_rk2_v2 requires"):
        _fixture(process_time_scheme="subcycled_rk2_v2")

@pytest.mark.parametrize("scan", [False, True])
def test_nonzero_surface_transport_and_absolute_stage_scales_are_not_discarded(scan):
    initial, params, _ = _transport_case(100., subcycles=3)
    params = params._replace(use_scan=scan, lambda_bulk=0.)
    phase = 2. * np.pi * np.arange(8) / 8.
    column_flux = np.broadcast_to((5. + 2. * np.sin(phase))[:, None], (8, 8))
    layer_flux = column_flux[..., None] * WIDTHS / 50.
    layer_divergence = (layer_flux - np.roll(layer_flux, 1, axis=0)) / 1000.
    vertical_flux = np.concatenate((np.cumsum(layer_divergence[..., ::-1], axis=-1)[..., ::-1],
                                    np.zeros((8, 8, 1))), axis=-1)
    initial = initial._replace(T=initial.T + jnp.asarray([0., 1., 2., 3.]), u=jnp.zeros_like(initial.u))

    def rhs(temperature):
        horizontal_flux = layer_flux * temperature
        internal_flux = vertical_flux[..., 1:-1] * np.where(
            vertical_flux[..., 1:-1] > 0., temperature[..., :-1], temperature[..., 1:])
        top = vertical_flux[..., 0] * temperature[..., 0]
        upper_flux = np.concatenate((top[..., None], internal_flux), axis=-1)
        lower_flux = np.concatenate((internal_flux, np.zeros((8, 8, 1))), axis=-1)
        tendency = -((horizontal_flux - np.roll(horizontal_flux, 1, axis=0)) / 1000.
                     + lower_flux - upper_flux) / WIDTHS
        return tendency, top

    reference = np.asarray(initial.T).copy()
    boundary_integral = 0.
    absolute_integral = 0.
    for _ in range(3):
        first, first_top = rhs(reference)
        second, second_top = rhs(reference + params.dt / 3. * first)
        reference += 0.5 * params.dt / 3. * (first + second)
        boundary_integral += 0.5 * params.dt / 3. * np.sum(first_top + second_top) * 1e6 * RHO_0 * C_P
        absolute_integral += 0.5 * params.dt / 3. * np.sum((np.abs(first) + np.abs(second))
                                                        * WIDTHS) * 1e6 * RHO_0 * C_P
    recorder = _StageRecorder(params)
    actual = _tracer_step_with_transport(initial, params, (jnp.asarray(column_flux), jnp.zeros((8, 8))), budget=recorder)
    np.testing.assert_allclose(actual.T, reference, rtol=1e-14, atol=4e-15)
    assert abs(boundary_integral) > 1e12
    np.testing.assert_allclose(recorder.advection_boundary[0], boundary_integral, rtol=1e-13, atol=1.)
    np.testing.assert_allclose(recorder.process_scale[0], absolute_integral, rtol=1e-13, atol=1.)
    np.testing.assert_allclose(recorder.difference(initial, actual)[0][0], boundary_integral, rtol=1e-11, atol=1.)
