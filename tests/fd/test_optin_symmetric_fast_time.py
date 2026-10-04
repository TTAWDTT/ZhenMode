"""Independent fast-wave/drag time gates; not a whole-mode second-order claim."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from tests.support.fd.reference_geometry import WIDTHS, _fixture
from tests.support.fd.subcycled_rk import _trajectory
from zhenmode.model.config import G_EARTH, RHO_0
from zhenmode.model.diagnostics.budgets import _StageRecorder, make_budget_step
from zhenmode.model.solver.dynamics.barotropic import (
    _barotropic_subcycle_transport,
    _free_surface_step_fd,
    _symmetric_free_surface_step,
)
from zhenmode.model.solver.timestepping.step import _linear_bottom_drag_step, _step_impl


def _candidate(**options):
    return _fixture(match_barotropic_transport=True, process_time_scheme="symmetric_fast_v3", **options)


def _kilometre(params):
    return params._replace(dx_2d=jnp.full_like(params.dx_2d, 1000.), dy=1000.,
                           inv_dx=jnp.full_like(params.inv_dx, 0.001), inv_dy=0.001,
                           inv_dx2=jnp.full_like(params.inv_dx2, 1e-6), inv_dy2=1e-6,
                           cos_lat=jnp.ones_like(params.cos_lat))


def gravity_order(scheme):
    _, (_, initialize, _, params, _) = _candidate(use_scan=True)
    base = _kilometre(params)._replace(process_time_scheme=scheme, adv_nsub=1)
    wave = 2. * np.pi * np.arange(8) / 8.
    amplitude = 1e-7
    wavenumber = np.sin(2. * np.pi / 8.) / 1000.
    frequency = np.sqrt(G_EARTH * 50.) * wavenumber
    reference_eta = amplitude * np.cos(frequency * 64.) * np.cos(wave)
    reference_u = amplitude * G_EARTH * wavenumber / frequency * np.sin(frequency * 64.) * np.sin(wave)
    rows = []
    for duration in (4., 2., 1.):
        current = base._replace(dt=duration, dt_bt=duration / 2., n_subcyc=2)
        initial = initialize()._replace(eta=jnp.broadcast_to(jnp.asarray(amplitude * np.cos(wave))[:, None], (8, 8)))
        actual = _trajectory(initial, current, 64.)
        eta_error = float(np.sqrt(np.mean((np.asarray(actual.eta)[:, 3] - reference_eta) ** 2)))
        velocity_error = float(np.sqrt(np.mean((np.asarray(actual.u)[:, 3, 0] - reference_u) ** 2)))
        rows.append({"dt_s": duration, "eta_rms_m": eta_error, "velocity_rms_m_per_s": velocity_error,
                     "normalized_error": float(np.sqrt(eta_error ** 2 + 50. / G_EARTH * velocity_error ** 2) / amplitude)})
    return rows


def wind_drag_order(scheme):
    forcing = (np.full((8, 8), 10.), np.zeros((8, 8)), np.zeros((8, 8)))
    _, (_, initialize, _, base, _) = _candidate(use_scan=True, forcing=forcing)
    base = base._replace(process_time_scheme=scheme, r_bot=0.01)
    initial = initialize()._replace(u=jnp.full((8, 8, 4), 0.2))
    reference = 0.2 * np.exp(-0.01 * 40.)
    rows = []
    for duration in (5., 2.5, 1.25):
        params = base._replace(dt=duration, dt_bt=duration / 2., n_subcyc=2)
        actual = _trajectory(initial, params, 40.)
        value = float(actual.u[0, 3, -1])
        rows.append({"dt_s": duration, "bottom_m_per_s": value, "reference_m_per_s": reference,
                     "error_m_per_s": abs(value - reference)})
        if scheme == "symmetric_fast_v3":
            np.testing.assert_allclose(np.asarray(actual.u)[..., 0], 0.2 + 10. * 40. / (RHO_0 * WIDTHS[0]), atol=2e-14, rtol=0.)
    return rows


def test_complete_step_wave_is_second_order_and_v2_remains_first_order():
    candidate = gravity_order("symmetric_fast_v3")
    control = gravity_order("subcycled_rk2_v2")
    ratios = [coarse["normalized_error"] / fine["normalized_error"]
              for coarse, fine in zip(candidate[:-1], candidate[1:], strict=True)]
    control_ratios = [coarse["normalized_error"] / fine["normalized_error"]
                      for coarse, fine in zip(control[:-1], control[1:], strict=True)]
    assert all(ratio >= 3.3 for ratio in ratios), ratios
    assert all(1.8 < ratio < 2.2 for ratio in control_ratios), control_ratios


def test_complete_step_surface_wind_does_not_artificially_force_the_dragged_bottom():
    candidate = wind_drag_order("symmetric_fast_v3")
    control = wind_drag_order("subcycled_rk2_v2")
    assert all(row["error_m_per_s"] < 1e-13 for row in candidate), candidate
    assert all(row["error_m_per_s"] > 1e-5 for row in control), control


def _numpy_faces(velocity_x, velocity_y, params):
    wet = np.asarray(params.wet_mask_z)
    common_x = np.sum(wet * np.roll(wet, -1, axis=0) * WIDTHS, axis=-1)
    common_y = np.sum(wet * np.roll(wet, -1, axis=1) * WIDTHS, axis=-1)
    face_x = 0.5 * (velocity_x + np.roll(velocity_x, -1, axis=0)) * common_x
    cosine = np.asarray(params.cos_lat)
    face_y = 0.5 * (velocity_y + np.roll(velocity_y, -1, axis=1)) * common_y * (0.5 * (cosine + np.roll(cosine, -1)))[None, :]
    face_y[:, -1] = 0.
    return face_x, face_y


def _numpy_divergence(faces, params):
    face_x, face_y = faces
    incoming_y = np.roll(face_y, 1, axis=1)
    incoming_y[:, 0] = 0.
    return ((face_x - np.roll(face_x, 1, axis=0)) / np.asarray(params.dx_2d)
            + (face_y - incoming_y) / (float(params.dy) * np.asarray(params.cos_lat)[None, :]))


def _numpy_operators(params):
    shape = params.wet_mask.shape
    size = np.prod(shape)
    basis = np.eye(2 * size)
    divergence = np.stack([_numpy_divergence(_numpy_faces(vector[:size].reshape(shape),
                                                        vector[size:].reshape(shape), params), params).ravel()
                           for vector in basis], axis=1)
    area = (np.asarray(params.dx_2d) * params.dy).ravel()
    depth = np.sum(np.asarray(params.wet_mask_z) * WIDTHS, axis=-1).ravel()
    weights = np.tile(area * np.maximum(depth, 1.), 2)
    wall = np.broadcast_to(np.asarray(params.interior_mask_z)[..., 0], shape)
    active = np.concatenate((np.asarray(params.wet_mask).ravel(), (np.asarray(params.wet_mask) * wall).ravel()))
    gradient = -(divergence.T * area[None, :]) / weights[:, None] * active[:, None]
    frequency = np.diag((np.asarray(params.f) * wall * np.asarray(params.wet_mask)).ravel())
    rotation = np.block([[np.zeros((size, size)), frequency], [-frequency, np.zeros((size, size))]])
    operator = np.block([[np.zeros((size, size)), -divergence], [-G_EARTH * gradient, rotation]])
    return divergence, gradient, rotation, operator, weights, area


@pytest.mark.parametrize("stairs", [False, True])
@pytest.mark.parametrize("scan", [False, True])
def test_actual_two_half_transport_matches_independent_masked_matrix(stairs, scan):
    _, (_, initialize, _, base, _) = _candidate(stairs=stairs, use_scan=scan)
    params = _kilometre(base)._replace(f=jnp.full_like(base.f, 0.001))
    divergence, gradient, rotation, _, _, _ = _numpy_operators(params)
    generator = np.random.default_rng(3191)
    shape = params.wet_mask.shape
    size = np.prod(shape)
    eta = generator.normal(size=shape) * 1e-5 * np.asarray(params.wet_mask)
    velocity_x = generator.normal(size=shape) * 1e-5 * np.asarray(params.wet_mask)
    velocity_y = generator.normal(size=shape) * 1e-5 * np.asarray(params.wet_mask) * np.asarray(params.interior_mask_z)[..., 0]
    velocity = np.concatenate((velocity_x.ravel(), velocity_y.ravel()))
    midpoint = eta.ravel() - 0.5 * params.dt_bt * (divergence @ velocity)
    identity = np.eye(2 * size)
    next_velocity = np.linalg.solve(identity - 0.5 * params.dt_bt * rotation,
                                    (identity + 0.5 * params.dt_bt * rotation) @ velocity
                                    - params.dt_bt * G_EARTH * (gradient @ midpoint))
    second_eta = midpoint - 0.5 * params.dt_bt * (divergence @ next_velocity)
    first_faces = _numpy_faces(velocity_x, velocity_y, params)
    second_faces = _numpy_faces(next_velocity[:size].reshape(shape), next_velocity[size:].reshape(shape), params)
    actual, faces = _symmetric_free_surface_step(jnp.asarray(eta), jnp.asarray(velocity_x),
                                                jnp.asarray(velocity_y), params, params.dt_bt)
    np.testing.assert_allclose(actual[0], second_eta.reshape(shape), atol=1e-19, rtol=1e-13)
    np.testing.assert_allclose(actual[1], next_velocity[:size].reshape(shape), atol=1e-19, rtol=1e-13)
    np.testing.assert_allclose(actual[2], next_velocity[size:].reshape(shape), atol=1e-19, rtol=1e-13)
    for actual_face, first, second in zip(faces, first_faces, second_faces, strict=True):
        np.testing.assert_allclose(actual_face, 0.5 * (first + second), atol=1e-18, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(actual[0]) - eta + params.dt_bt * _numpy_divergence(faces, params), 0., atol=1e-19)
    np.testing.assert_array_equal(np.asarray(actual[2])[:, [0, -1]], 0.)
    assert max(np.max(np.abs(mean - end)) for mean, end in zip(faces, second_faces, strict=True)) > 1e-8


@pytest.mark.parametrize("stairs", [False, True])
def test_fast_wet_geometry_rotating_wave_converges_to_independent_matrix_exponential(stairs):
    _, (_, initialize, _, base, _) = _candidate(stairs=stairs, use_scan=True)
    params = _kilometre(base)._replace(f=jnp.full_like(base.f, 0.001))
    _, _, _, operator, weights, area = _numpy_operators(params)
    initial_eta = np.random.default_rng(824).normal(size=(8, 8)) * 1e-7 * np.asarray(params.wet_mask)
    initial = (jnp.asarray(initial_eta), jnp.zeros((8, 8)), jnp.zeros((8, 8)))
    reference = expm(operator * 64.) @ np.concatenate((initial_eta.ravel(), np.zeros(128)))
    energy_weights = np.concatenate((G_EARTH * area, weights))
    errors = []
    for duration in (4., 2., 1.):
        count = int(64. / duration)
        actual = jax.jit(lambda values: jax.lax.fori_loop(0, count, lambda index, current:
            _free_surface_step_fd(*current, params, dt_half=duration), values))(initial)
        vector = np.concatenate([np.asarray(value).ravel() for value in actual])
        errors.append(float(np.sqrt(np.sum(energy_weights * (vector - reference) ** 2))))
    ratios = [coarse / fine for coarse, fine in zip(errors[:-1], errors[1:], strict=True)]
    assert all(ratio >= 3.3 for ratio in ratios), (errors, ratios)


def test_actual_drag_loss_and_face_change_are_separate_from_numerical_filtering():
    _, (_, initialize, _, params, _) = _candidate(use_scan=True)
    params = params._replace(r_bot=0.01)
    initial = initialize()._replace(u=jnp.full((8, 8, 4), 0.2))
    result, ledger = make_budget_step(params)(initial)
    depth = np.sum(np.asarray(params.bottom_mask * params.wet_mask_z) * WIDTHS, axis=-1)
    area = np.asarray(params.dx_2d) * params.dy
    expected_loss = 0.5 * RHO_0 * np.sum(area * depth) * 0.2 ** 2 * (1. - np.exp(-2. * 0.01 * params.dt))
    np.testing.assert_allclose(ledger["bottom_drag_reference_energy_loss_J"], expected_loss, rtol=1e-13)
    np.testing.assert_allclose(ledger["bottom_drag_face_change_max_m2_per_s"], 15. * 0.2 * (1. - np.exp(-0.05)), rtol=1e-13)
    assert float(ledger["bottom_drag_audited_halves"]) == 2.
    assert float(ledger["transport_consistency_max"][3]) == 0.
    np.testing.assert_allclose(np.asarray(result.u)[..., -1], 0.2 * np.exp(-0.1), atol=1e-15, rtol=0.)
    recorder = _StageRecorder(params)
    direct = _linear_bottom_drag_step(initial, params, 5., budget=recorder)
    np.testing.assert_array_equal(np.asarray(direct.T), np.asarray(initial.T))
    assert float(recorder.drag_halves) == 1.


@pytest.mark.parametrize("scan", [False, True])
def test_actual_source_count_transport_match_and_audited_state(scan):
    forcing = (np.full((8, 8), 0.05), np.zeros((8, 8)), np.full((8, 8), 100.))
    _, (_, initialize, _, base, _) = _candidate(use_scan=scan, forcing=forcing)
    params = _kilometre(base)
    initial = initialize(T_init=jnp.zeros((8, 8, 4)))
    direct = jax.jit(lambda state: _step_impl(state, params))(initial)
    audited, ledger = make_budget_step(params)(initial)
    for before, after in zip(direct, audited, strict=True):
        np.testing.assert_array_equal(before, after)
    momentum = RHO_0 * np.sum(np.asarray(audited.u - initial.u) * WIDTHS, axis=-1)
    np.testing.assert_allclose(momentum, 0.05 * params.dt, atol=1e-13, rtol=1e-11)
    assert float(ledger["transport_consistency_max"][0]) < 1e-14
    assert float(ledger["transport_consistency_max"][2]) < 1e-13
    np.testing.assert_allclose(ledger["source_inputs"][0, 0], 100. * 64. * 1e6 * params.dt, rtol=1e-13, atol=0.1)


@pytest.mark.parametrize("scan", [False, True])
def test_multiple_actual_substeps_include_common_wet_shear_in_both_half_faces(scan):
    _, (_, initialize, _, base, _) = _candidate(stairs=True, use_scan=scan)
    params = _kilometre(base)
    generator = np.random.default_rng(379)
    wet = np.asarray(params.wet_mask_z)
    initial = initialize(T_init=jnp.full((8, 8, 4), params.T_ref), S_init=jnp.full((8, 8, 4), params.S_ref))
    initial = initial._replace(u=jnp.asarray(generator.normal(size=wet.shape) * 1e-4 * wet),
                               v=jnp.asarray(generator.normal(size=wet.shape) * 1e-4 * wet * np.asarray(params.interior_mask_z)),
                               eta=jnp.asarray(generator.normal(size=(8, 8)) * 1e-5 * np.asarray(params.wet_mask)))
    depth = np.maximum(np.sum(wet * WIDTHS, axis=-1), 1.)
    mean_x = np.sum(np.asarray(initial.u) * wet * WIDTHS, axis=-1) / depth
    mean_y = np.sum(np.asarray(initial.v) * wet * WIDTHS, axis=-1) / depth
    shear_x = np.asarray(initial.u) - mean_x[..., None] * wet
    shear_y = np.asarray(initial.v) - mean_y[..., None] * wet
    _, gradient, _, _, _, _ = _numpy_operators(params)
    eta = np.asarray(initial.eta).copy()
    totals = [np.zeros((8, 8)), np.zeros((8, 8))]

    def layer_faces(velocity_x, velocity_y):
        face_x = np.sum(0.5 * (velocity_x + np.roll(velocity_x, -1, axis=0)) * wet * np.roll(wet, -1, axis=0) * WIDTHS, axis=-1)
        face_y = np.sum(0.5 * (velocity_y + np.roll(velocity_y, -1, axis=1)) * wet * np.roll(wet, -1, axis=1) * WIDTHS, axis=-1)
        face_y[:, -1] = 0.
        return face_x, face_y

    for _ in range(params.n_subcyc):
        first = layer_faces(shear_x + mean_x[..., None] * wet, shear_y + mean_y[..., None] * wet)
        middle = eta - 0.5 * params.dt_bt * _numpy_divergence(first, params)
        acceleration = -G_EARTH * gradient @ middle.ravel()
        mean_x = mean_x + params.dt_bt * acceleration[:64].reshape((8, 8))
        mean_y = mean_y + params.dt_bt * acceleration[64:].reshape((8, 8))
        second = layer_faces(shear_x + mean_x[..., None] * wet, shear_y + mean_y[..., None] * wet)
        eta = middle - 0.5 * params.dt_bt * _numpy_divergence(second, params)
        totals = [total + 0.5 * (first_face + second_face)
                  for total, first_face, second_face in zip(totals, first, second, strict=True)]
    result, faces, changes = _barotropic_subcycle_transport(initial, params)
    np.testing.assert_allclose(result.eta, eta, atol=3e-18, rtol=1e-13)
    for actual, reference in zip(faces, totals, strict=True):
        np.testing.assert_allclose(actual, reference / params.n_subcyc, atol=1e-17, rtol=1e-13)
    np.testing.assert_allclose(changes, 0., atol=1e-18, rtol=0.)
    np.testing.assert_allclose(np.asarray(result.u), shear_x + mean_x[..., None] * wet, atol=1e-18, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(result.v), shear_y + mean_y[..., None] * wet, atol=1e-18, rtol=1e-13)


def test_fast_gravity_preserves_independent_modified_energy_for_149_periods_below_cfl():
    _, (_, initialize, _, base, _) = _candidate(use_scan=True)
    params = _kilometre(base)
    amplitude = 1e-6
    wave = 2. * np.pi * np.arange(8) / 8.
    duration = 30.
    frequency = np.sqrt(G_EARTH * 50.) * np.sin(2. * np.pi / 8.) / 1000.
    correction = 1. - (duration * frequency / 2.) ** 2
    initial = (jnp.broadcast_to(jnp.asarray(amplitude * np.cos(wave))[:, None], (8, 8)),
               jnp.zeros((8, 8)), jnp.zeros((8, 8)))
    actual = jax.jit(lambda values: jax.lax.fori_loop(0, 2000, lambda index, current:
        _free_surface_step_fd(*current, params, dt_half=duration), values))(initial)
    modified_before = G_EARTH * np.sum(np.asarray(initial[0]) ** 2) / correction
    modified_after = G_EARTH * np.sum(np.asarray(actual[0]) ** 2) / correction + 50. * np.sum(np.asarray(actual[1]) ** 2)
    np.testing.assert_allclose(modified_after / modified_before, 1., rtol=0., atol=1e-10)
    physical_after = G_EARTH * np.sum(np.asarray(actual[0]) ** 2) + 50. * np.sum(np.asarray(actual[1]) ** 2)
    assert 0.94 < physical_after / (G_EARTH * np.sum(np.asarray(initial[0]) ** 2)) < 1.07


def test_actual_three_step_jvp_vjp_and_centered_finite_difference():
    _, (_, initialize, _, params, _) = _candidate(stairs=True, use_scan=True)
    params = params._replace(r_bot=0.01)
    initial = initialize()
    generator = np.random.default_rng(873)
    direction = jnp.asarray(generator.normal(size=initial.T.shape)) * params.wet_mask_z
    probe = jnp.asarray(generator.normal(size=initial.T.shape)) * params.wet_mask_z

    @jax.jit
    def objective(scale):
        current = initial._replace(T=initial.T + scale * direction)
        for _ in range(3):
            current = _step_impl(current, params)
        return jnp.sum(current.T * probe) + 0.01 * jnp.sum(current.eta)

    center = 0.02
    reverse = jax.grad(objective)(center)
    _, forward = jax.jvp(objective, (center,), (1.,))
    increment = 1e-4
    difference = (objective(center + increment) - objective(center - increment)) / (2. * increment)
    assert np.isfinite(float(reverse))
    np.testing.assert_allclose(reverse, forward, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(reverse, difference, rtol=1e-7, atol=1e-7)
