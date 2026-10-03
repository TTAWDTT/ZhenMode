"""Independent fast/tracer face gates; static geometry is not a moving budget."""
import jax.numpy as jnp
import numpy as np
import pytest

from ocean_solver.audit.schema import (
    MAXIMUM_BUDGET_FIELDS,
    SOURCE_NAMES,
    accumulate_budget,
    empty_budget,
)
from ocean_solver.audit.stages import make_budget_step
from ocean_solver.config.definitions import C_P, RHO_0, PhysicsConfig
from ocean_solver.dynamics.barotropic import _barotropic_subcycle_transport, _free_surface_step_fd
from ocean_solver.dynamics.processes import _linear_half_step
from ocean_solver.dynamics.transport import (
    _advection_scalar,
    _barotropic_velocity,
    _face_transport_divergence,
    _layer_face_transports,
    _match_layer_face_transports,
    _vertical_transport_iface,
)
from ocean_solver.model.factory import make_solver_global
from ocean_solver.timestepping.integration import _explicit_full_step, _tracer_step_with_transport
from tests.support.material.reference_geometry import WIDTHS, _fixture


def _numpy_faces(velocity_x, velocity_y, grid):
    wet = grid.wet_mask_3d
    flux_x = 0.5 * (velocity_x + np.roll(velocity_x, -1, axis=0)) * wet * np.roll(wet, -1, axis=0) * WIDTHS
    flux_y = 0.5 * (velocity_y + np.roll(velocity_y, -1, axis=1)) * wet * np.roll(wet, -1, axis=1) * WIDTHS
    flux_y *= 0.5 * (grid.cos_lat + np.roll(grid.cos_lat, -1))[None, :, None]
    flux_y[:, -1] = 0.
    return flux_x, flux_y


def _numpy_divergence(flux_x, flux_y, grid):
    incoming_y = np.roll(flux_y, 1, axis=1)
    incoming_y[:, 0] = 0.
    if flux_x.ndim == 3:
        inverse_dx = 1. / grid.dx_2d[..., None]
        cosine = grid.cos_lat[None, :, None]
    else:
        inverse_dx = 1. / grid.dx_2d
        cosine = grid.cos_lat[None, :]
    return (flux_x - np.roll(flux_x, 1, axis=0)) * inverse_dx + (flux_y - incoming_y) / (grid.dy * cosine)


def _perturbed_state(initialize, params):
    generator = np.random.default_rng(829)
    state = initialize()
    return state._replace(u=jnp.asarray(generator.normal(size=state.u.shape) * 0.2) * params.wet_mask_z,
                          v=jnp.asarray(generator.normal(size=state.v.shape) * 0.1) * params.wet_mask_z * params.interior_mask_z,
                          eta=jnp.asarray(generator.normal(size=state.eta.shape) * 0.05) * params.wet_mask)


@pytest.mark.parametrize("scan", [False, True])
def test_fast_mean_uses_old_faces_not_endpoint_velocity(scan):
    grid, (_, initialize, _, params, _) = _fixture(stairs=True, use_scan=scan,
                                                   match_barotropic_transport=True)
    state = _perturbed_state(initialize, params)
    updated, measured, filters = _barotropic_subcycle_transport(state, params)
    initial_x, initial_y = _barotropic_velocity(state.u, state.v, params)
    mean_x, mean_y, eta = initial_x, initial_y, state.eta
    expected = [np.zeros_like(grid.depth), np.zeros_like(grid.depth)]
    for _ in range(params.n_subcyc):
        velocity_x = np.asarray(state.u + (mean_x - initial_x)[..., None] * params.wet_mask_z)
        velocity_y = np.asarray(state.v + (mean_y - initial_y)[..., None] * params.wet_mask_z)
        layers = _numpy_faces(velocity_x, velocity_y, grid)
        faces = tuple(np.sum(flux, axis=-1) for flux in layers)
        for index, flux in enumerate(faces):
            expected[index] += flux / params.n_subcyc
        eta, mean_x, mean_y = _free_surface_step_fd(
            eta, mean_x, mean_y, params, dt_half=params.dt_bt,
            column_face_transport=tuple(jnp.asarray(flux) for flux in faces))
    for measured_flux, expected_flux in zip(measured, expected, strict=True):
        np.testing.assert_allclose(measured_flux, expected_flux, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(updated.eta, eta, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(np.asarray(updated.eta - state.eta) + params.dt * _numpy_divergence(*expected, grid),
                               filters, atol=1e-15, rtol=0.)
    endpoint = tuple(np.sum(flux, axis=-1) for flux in _numpy_faces(np.asarray(updated.u), np.asarray(updated.v), grid))
    assert max(np.max(np.abs(old - new)) for old, new in zip(expected, endpoint, strict=True)) > 1e-7


@pytest.mark.parametrize("stairs", [False, True])
def test_layer_match_preserves_shear_closes_faces_and_diagnoses_vertical_transport(stairs):
    grid, (_, initialize, _, params, _) = _fixture(stairs=stairs, match_barotropic_transport=True)
    state = _perturbed_state(initialize, params)
    target = tuple(np.sum(flux, axis=-1) for flux in _numpy_faces(np.asarray(state.u * 1.4), np.asarray(state.v * 0.8), grid))
    original = _numpy_faces(np.asarray(state.u), np.asarray(state.v), grid)
    corrected = _match_layer_face_transports(state.u, state.v, tuple(jnp.asarray(flux) for flux in target), params)
    for axis, before, after, column in zip((0, 1), original, corrected, target, strict=True):
        np.testing.assert_allclose(np.sum(after, axis=-1), column, rtol=1e-12, atol=1e-13)
        wet = grid.wet_mask_3d * np.roll(grid.wet_mask_3d, -1, axis=axis)
        if axis == 1:
            wet[:, -1] = 0.
        np.testing.assert_array_equal(np.asarray(after)[wet == 0.], 0.)
        correction = (np.asarray(after) - before) / WIDTHS
        shared = (wet[..., :-1] * wet[..., 1:]).astype(bool)
        np.testing.assert_allclose((correction[..., 1:] - correction[..., :-1])[shared], 0., atol=1e-14)
    vertical = _vertical_transport_iface(state.u, state.v, params, face_transport=corrected)
    expected = _numpy_divergence(*tuple(np.asarray(flux) for flux in corrected), grid)
    np.testing.assert_allclose(np.diff(vertical, axis=-1), -expected, rtol=1e-11, atol=1e-18)
    np.testing.assert_allclose(vertical[..., 0], _numpy_divergence(*target, grid), rtol=1e-11, atol=1e-18)
    np.testing.assert_array_equal(vertical[..., -1], 0.)


@pytest.mark.parametrize("fct,donor", [(False, False), (False, True), (True, False)])
def test_matched_advection_preserves_constant_tracer_without_deleting_surface_flux(fct, donor):
    grid, (_, initialize, _, params, _) = _fixture(stairs=True, fct_adv=fct, monotone_adv=donor,
                                                   match_barotropic_transport=True)
    state = _perturbed_state(initialize, params)
    _, mean, _ = _barotropic_subcycle_transport(state, params)
    faces = _match_layer_face_transports(state.u, state.v, mean, params)
    vertical = _vertical_transport_iface(state.u, state.v, params, face_transport=faces)
    tracer = jnp.where(params.wet_mask_z > 0., 7., 999.)
    tendency, surface = _advection_scalar(tracer, state.u, state.v, vertical, params,
                                          return_boundary=True, face_transport=faces)
    np.testing.assert_allclose(tendency, 0., atol=3e-18)
    np.testing.assert_allclose(surface, 7. * _numpy_divergence(*tuple(np.asarray(flux) for flux in mean), grid),
                               rtol=1e-11, atol=1e-18)
    assert np.max(np.abs(surface)) > 1e-7


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("scan", [False, True])
def test_full_step_uses_matched_transport_not_just_a_diagnostic(dtype, scan):
    grid, (step, initialize, _, params, _) = _fixture(
        stairs=True, dtype=dtype, use_scan=scan, match_barotropic_transport=True, project_adv_vel=True)
    before = _perturbed_state(initialize, params)
    before = before._replace(**{name: field.astype(getattr(params, "wet_mask_z").dtype)
                                 for name, field in before._asdict().items()})
    temperature = 16. + 2. * jnp.arange(8)[:, None, None] - 2. * jnp.arange(4)[None, None, :]
    before = before._replace(T=jnp.broadcast_to(temperature.astype(before.T.dtype), before.T.shape))
    after, ledger = make_budget_step(params)(before)
    normal = step(before)
    epsilon = np.finfo(np.dtype(dtype)).eps
    for measured, expected in zip(after, normal, strict=True):
        np.testing.assert_allclose(measured, expected, rtol=32. * epsilon, atol=1e-10)
    metrics = np.asarray(ledger["transport_consistency_max"])
    assert metrics[0] < 32. * epsilon
    assert metrics[2] < 128. * epsilon
    assert float(ledger["transport_audited_steps"]) == 1.
    stage_start = _linear_half_step(before, params, params.dt / 2.)
    provisional = _explicit_full_step(stage_start, params, params.dt)
    provisional = _linear_half_step(provisional, params, params.dt / 2.)
    _, mean, _ = _barotropic_subcycle_transport(provisional, params)
    replay = _tracer_step_with_transport(stage_start, params, mean)
    replay = _linear_half_step(replay, params, params.dt / 2.)
    np.testing.assert_allclose(after.T, replay.T, rtol=32. * epsilon, atol=1e-10)
    old_params = params._replace(match_barotropic_transport=False)
    _, old_ledger = make_budget_step(old_params)(before)
    assert float(old_ledger["transport_consistency_max"][2]) > 1e-4
    assert float(ledger["projection_relative_residual_max"]) == 0.
    volume = grid.dx_2d[..., None] * grid.dy * WIDTHS * grid.wet_mask_3d
    inventory_scale = np.array([RHO_0 * C_P * np.sum(np.abs(np.asarray(before.T)) * volume),
                                RHO_0 / 1000. * np.sum(np.abs(np.asarray(before.S)) * volume), 1.])
    assert np.all(np.abs(ledger["nonlinear_accounting_residual"]) <= 16. * epsilon * inventory_scale)
    assert np.any(np.abs(np.asarray(after.T - provisional.T)) > epsilon)


@pytest.mark.parametrize("ice", [False, True])
def test_replay_counts_external_heat_once_and_never_as_transport(ice):
    heat = -100. if ice else 100.
    forcing = (np.full((8, 8), 0.05), np.zeros((8, 8)), np.full((8, 8), heat))
    _, (_, initialize, _, params, _) = _fixture(forcing=forcing, dynamic_ice=ice,
                                               match_barotropic_transport=True)
    before = initialize()
    if ice:
        before = before._replace(T=jnp.full_like(before.T, params.ice_freeze_temp_c))
    after, ledger = make_budget_step(params)(before)
    expected = heat * params.dt * np.sum(np.asarray(params.dx_2d) * params.dy)
    sources = np.asarray(ledger["source_inputs"])
    assert np.sum(sources[:, 0]) == pytest.approx(expected, rel=1e-12)
    assert float(ledger["observed_change"][0]) == pytest.approx(expected, rel=1e-10)
    source = "ice_atmosphere_heat" if ice else "prescribed_heat"
    assert sources[SOURCE_NAMES.index(source), 0] == pytest.approx(expected, rel=1e-12)
    momentum = RHO_0 * np.sum(np.asarray(after.u - before.u) * WIDTHS, axis=-1)
    np.testing.assert_allclose(momentum, 0.05 * params.dt, rtol=1e-11, atol=1e-13)


def test_filters_are_not_hidden_in_the_transport_or_global_mean_correction():
    _, (_, initialize, _, params, _) = _fixture(stairs=True, polar_cap_rows=1, polar_cap_taper=1,
                                               match_barotropic_transport=True)
    before = _perturbed_state(initialize, params)
    tracer = 12. + jnp.sin(jnp.arange(8))[:, None, None] - 0.2 * jnp.arange(4)[None, None, :]
    before = before._replace(T=jnp.broadcast_to(tracer, before.T.shape))
    _, ledger = make_budget_step(params)(before)
    continuity, filters, _, final_flux_change = np.asarray(ledger["transport_consistency_max"])
    assert continuity < 1e-14
    assert filters > 1e-4
    assert final_flux_change > 1e-4
    assert not np.allclose(ledger["budget_residual"], 0., atol=1.)
    assert "transport" not in SOURCE_NAMES


@pytest.mark.parametrize("substeps", [1, 3])
def test_accepted_tracer_rk_and_advection_substeps_match_independent_flux_update(substeps):
    grid, (_, initialize, _, params, _) = _fixture(stairs=True, match_barotropic_transport=True)
    params = params._replace(adv_nsub=substeps)
    state = _perturbed_state(initialize, params)
    generator = np.random.default_rng(891)
    state = state._replace(T=jnp.asarray(generator.uniform(5., 15., state.T.shape)))
    target = tuple(jnp.sum(flux, axis=-1) * 1.3 for flux in _layer_face_transports(state.u, state.v, params))
    faces = _match_layer_face_transports(state.u, state.v, target, params)
    flux_x, flux_y = tuple(np.asarray(flux) for flux in faces)
    volume_divergence = _numpy_divergence(flux_x, flux_y, grid)
    vertical = np.concatenate([np.cumsum(volume_divergence[..., ::-1], axis=-1)[..., ::-1],
                               np.zeros_like(volume_divergence[..., :1])], axis=-1)
    wet_interfaces = grid.wet_mask_3d[..., :-1] * grid.wet_mask_3d[..., 1:]

    def tendency(tracer):
        tracer_flux_x = flux_x * 0.5 * (tracer + np.roll(tracer, -1, axis=0))
        tracer_flux_y = flux_y * 0.5 * (tracer + np.roll(tracer, -1, axis=1))
        internal = (vertical[..., 1:-1] * np.where(vertical[..., 1:-1] > 0., tracer[..., :-1], tracer[..., 1:])
                    * wet_interfaces)
        upper = np.concatenate([vertical[..., :1] * tracer[..., :1], internal], axis=-1)
        lower = np.concatenate([internal, np.zeros_like(internal[..., :1])], axis=-1)
        return -(_numpy_divergence(tracer_flux_x, tracer_flux_y, grid) + lower - upper) / WIDTHS * grid.wet_mask_3d

    def mean_tendency(tracer):
        working = tracer.copy()
        rates = []
        for _ in range(substeps):
            rate = tendency(working)
            rates.append(rate)
            working += params.dt / substeps * rate
        return np.mean(rates, axis=0)

    original = np.asarray(state.T)
    first = mean_tendency(original)
    second = mean_tendency(original + params.dt * first)
    expected = original + 0.5 * params.dt * (first + second)
    actual = _tracer_step_with_transport(state, params, target)
    np.testing.assert_allclose(actual.T, expected, rtol=1e-13, atol=1e-13)
    assert np.max(np.abs(expected - original)) > 1e-6


def test_transport_audit_accumulates_maxima_not_a_cancelling_signed_sum():
    first, second = empty_budget(), empty_budget()
    first["transport_consistency_max"] = jnp.array([1., 4., 2., 3.])
    second["transport_consistency_max"] = jnp.array([2., 1., 3., 4.])
    first["transport_audited_steps"] = second["transport_audited_steps"] = jnp.asarray(1.)
    totals = accumulate_budget(first, second)
    np.testing.assert_array_equal(totals["transport_consistency_max"], [2., 4., 3., 4.])
    assert totals["transport_audited_steps"] == 2.
    assert "transport_consistency_max" in MAXIMUM_BUDGET_FIELDS


@pytest.mark.parametrize("options", [{"column_geometry": "legacy"}, {"mode_split": False}])
def test_transport_candidate_rejects_incompatible_geometry_or_single_step(options):
    grid, _ = _fixture()
    settings = dict(column_geometry="nodal_dual_v1", mode_split=True, conservative_kv=True,
                    localize_conv=True, match_barotropic_transport=True)
    settings.update(options)
    with pytest.raises(ValueError, match="match_barotropic_transport requires"):
        make_solver_global(grid, PhysicsConfig(), 10., **settings)


def test_face_divergence_telescopes_in_the_declared_reference_volume():
    grid, (_, initialize, _, params, _) = _fixture(stairs=True)
    state = _perturbed_state(initialize, params)
    faces = _layer_face_transports(state.u, state.v, params)
    measured = _face_transport_divergence(*faces, params)
    expected = _numpy_divergence(*_numpy_faces(np.asarray(state.u), np.asarray(state.v), grid), grid)
    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-18)
    area = grid.dx_2d * grid.dy
    volume_change = np.sum(expected * area[..., None])
    scale = np.sum(np.abs(expected * area[..., None]))
    assert abs(volume_change) < 1e-14 * scale
    uniform = jnp.full_like(state.T, 7.)
    vertical = _vertical_transport_iface(state.u, state.v, params, face_transport=faces)
    tendency = _advection_scalar(uniform, state.u, state.v, vertical, params, face_transport=faces)
    heat_rate = RHO_0 * C_P * np.sum(np.asarray(tendency) * area[..., None] * WIDTHS)
    assert abs(heat_rate) < 1e-12 * RHO_0 * C_P * scale
