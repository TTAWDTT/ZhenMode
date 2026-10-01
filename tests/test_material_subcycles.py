"""Independent actual-volume schedules, source clocks, AD and restart gates."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_material_top import _material_fixture, _numpy_divergence, _numpy_transport_rhs, _state

from config import ALPHA_T, BETA_S, C_P, RHO_0, PhysicsConfig
from jax_solver_global import JaxStateG
from material_top import (
    _contents,
    _linear_material_subcycle,
    _material_tracer_step,
    _nonlinear_subcycle_plan,
    _subcycle_plan,
    make_material_top_restart_contract,
    make_material_top_step,
)
from restart_contract import load_restart, save_restart

POLICY = "actual_geometry_v2"


def _numpy_plan(state, params, faces, maximum):
    wet = np.asarray(params.wet_mask_z) > 0.
    east, north = faces
    divergence = _numpy_divergence(east, north, params)
    accumulated = np.flip(np.cumsum(np.flip(divergence, axis=-1), axis=-1), axis=-1)
    vertical = np.concatenate((np.zeros_like(accumulated[..., :1]), accumulated[..., 1:],
                               np.zeros_like(accumulated[..., :1])), axis=-1)
    south = np.roll(north, 1, axis=1)
    south[:, 0] = 0.
    outflow = ((np.maximum(east, 0.) + np.maximum(-np.roll(east, 1, axis=0), 0.))
               / np.asarray(params.dx_2d)[..., None]
               + (np.maximum(north, 0.) + np.maximum(-south, 0.))
               / (float(params.dy) * np.asarray(params.cos_lat)[None, :, None])
               + np.maximum(vertical[..., 1:], 0.) + np.maximum(-vertical[..., :-1], 0.))
    conductance = params.kappa_conv * wet[..., :-1] * wet[..., 1:] / np.asarray(params.dz_iface)
    zero = np.zeros_like(conductance[..., :1])
    convection = np.concatenate((zero, conductance), axis=-1) + np.concatenate((conductance, zero), axis=-1)
    feedback = np.zeros_like(outflow)
    feedback[..., 0] = np.maximum((params.lambda_bulk + np.asarray(params.coastal_bulk_lambda_2d)) / (RHO_0 * C_P)
                                  + np.asarray(params.dz_node).ravel()[0] * np.asarray(params.coastal_restore_coef_2d),
                                  np.asarray(params.dz_node).ravel()[0] * params.restore_coef_S)
    end_eta = np.asarray(state.eta) - params.dt * np.sum(divergence, axis=-1)
    thickness = np.broadcast_to(np.asarray(params.dz_node), wet.shape).copy()
    thickness[..., 0] += np.minimum(np.asarray(state.eta), end_eta)
    required = max(params.adv_nsub, params.conv_nsub,
                   int(np.ceil(np.max(np.where(wet, params.dt * (outflow + convection + feedback)
                                               / np.where(thickness > 0., thickness, 1.), 0.)) / .5)))
    return required, bool(required <= maximum and np.all(thickness[wet] > 0.))


def _numpy_convection(concentration, temperature, salinity, params):
    wet = np.asarray(params.wet_mask_z) > 0.
    density = -ALPHA_T * (temperature - params.T_ref) + BETA_S * (salinity - params.S_ref)
    gate = (density[..., :-1] > density[..., 1:]) & wet[..., :-1] & wet[..., 1:]
    flux = -params.kappa_conv * np.diff(concentration, axis=-1) / np.asarray(params.dz_iface) * gate
    zero = np.zeros_like(flux[..., :1])
    return np.concatenate((zero, flux), axis=-1) - np.concatenate((flux, zero), axis=-1)


def test_numpy_schedule_and_every_actual_heun_stage_share_volume_and_sources():
    _, (_, initialize, _, params, _) = _material_fixture(stairs=True)
    params = params._replace(kappa_conv=.03, Q_heat_2d=jnp.full_like(params.Q_heat_2d, 100.))
    generator = np.random.default_rng(292203)
    state = _state(initialize)
    wet = np.asarray(params.wet_mask_z) > 0.
    state = state._replace(eta=jnp.asarray(generator.uniform(-2.1, -1.8, state.eta.shape)) * params.wet_mask,
                           T=jnp.asarray(generator.uniform(2., 18., state.T.shape)),
                           S=jnp.asarray(generator.uniform(33., 36., state.S.shape)))
    east = generator.normal(size=state.T.shape) * .0004 * np.asarray(params.dz_node) * np.asarray(params.dx_2d)[..., None]
    north = generator.normal(size=state.T.shape) * .0002 * np.asarray(params.dz_node) * params.dy * np.asarray(params.cos_lat)[None, :, None]
    east *= wet * np.roll(wet, -1, axis=0)
    north *= wet * np.roll(wet, -1, axis=1)
    north[:, -1] = 0.
    faces = east, north
    plan = _nonlinear_subcycle_plan(state, params, tuple(jnp.asarray(flux) for flux in faces), 128)
    expected_count, expected_supported = _numpy_plan(state, params, faces, 128)
    assert int(plan.count) == expected_count and bool(plan.supported) == expected_supported
    result = _material_tracer_step(state, params, tuple(jnp.asarray(flux) for flux in faces), subcycle_plan=plan)
    eta = np.asarray(state.eta).copy()
    height = np.broadcast_to(np.asarray(params.dz_node), wet.shape).copy()
    height[..., 0] += eta
    eta_rate = -np.sum(_numpy_divergence(*faces, params), axis=-1)
    content = np.asarray(_contents(state, params)).copy()
    duration = params.dt / expected_count
    expected_source = np.zeros(2)
    area = np.asarray(params.dx_2d) * params.dy
    for _ in range(expected_count):
        concentrations = content / np.where(wet, height, 1.)[..., None]
        tendencies = []
        for index in range(2):
            tendency = (_numpy_transport_rhs(concentrations[..., index], faces, params)
                        + _numpy_convection(concentrations[..., index], concentrations[..., 0], concentrations[..., 1], params))
            if index == 0:
                tendency[..., 0] += 100. / (RHO_0 * C_P) * wet[..., 0]
            tendencies.append(tendency)
        first = np.stack(tendencies, axis=-1)
        predicted_eta = eta + duration * eta_rate
        predicted_height = height.copy()
        predicted_height[..., 0] += duration * eta_rate
        predicted = (content + duration * first) / np.where(wet, predicted_height, 1.)[..., None]
        tendencies = []
        for index in range(2):
            tendency = (_numpy_transport_rhs(predicted[..., index], faces, params)
                        + _numpy_convection(predicted[..., index], predicted[..., 0], predicted[..., 1], params))
            if index == 0:
                tendency[..., 0] += 100. / (RHO_0 * C_P) * wet[..., 0]
            tendencies.append(tendency)
        content += .5 * duration * (first + np.stack(tendencies, axis=-1))
        eta, height = predicted_eta, predicted_height
        expected_source[0] += duration * 100. * np.sum(area * wet[..., 0])
    np.testing.assert_allclose(_contents(result[0], params), content, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result[0].eta, eta, rtol=0., atol=1e-14)
    np.testing.assert_allclose(np.sum(np.asarray(result[3]), axis=0), expected_source, rtol=1e-12, atol=1e-12)
    assert float(result[-2][2]) <= .5


@pytest.mark.parametrize("source", ["convection", "bulk"])
def test_thin_actual_complete_step_subcycles_instead_of_changing_coefficients(source):
    _, (_, initialize, _, params, _) = _material_fixture(T_atm=np.full((8, 8), 5.))
    state = _state(initialize, eta=-2.499, temperature=10.)
    if source == "convection":
        params = params._replace(kappa_conv=.01)
        state = state._replace(T=state.T.at[..., 0].set(1.))
    else:
        params = params._replace(lambda_bulk=8000., T_atm_3d=jnp.full_like(params.T_atm_3d, 5.))
    advance = make_material_top_step(params, subcycle_scheme=POLICY)
    result = advance(state)
    assert bool(result.valid), result.checks
    count = int(result.checks["nonlinear_active_subcycles"])
    assert count > max(params.adv_nsub, params.conv_nsub)
    assert float(result.checks["nonlinear_combined_fraction_max"]) <= .5
    assert params.kappa_conv == (.01 if source == "convection" else 0.)
    assert params.dt == 10.
    if source == "bulk":
        height = float(np.asarray(2.5 + state.eta[0, 0]))
        fraction = params.dt * params.lambda_bulk / (count * RHO_0 * C_P * height)
        expected = 5. + 5. * (1. - fraction + .5 * fraction ** 2) ** count
        np.testing.assert_allclose(result.state.T[..., 0], expected, rtol=0., atol=1e-12)
        assert np.max(np.asarray(result.state.T[..., 0])) <= 10.
        assert np.min(np.asarray(result.state.T[..., 0])) >= 5.
    else:
        assert np.min(np.asarray(result.state.T)) >= 1. - 1e-12
        assert np.max(np.asarray(result.state.T)) <= 10. + 1e-12


def test_linear_thin_mixing_matches_independent_mass_weighted_matrix_heun():
    _, (_, initialize, _, params, _) = _material_fixture()
    params = params._replace(kappa_v=.1)
    state = _state(initialize, eta=-2.49)
    profile = np.array([1., 10., 8., 5.])
    state = state._replace(T=jnp.broadcast_to(jnp.asarray(profile), state.T.shape))
    updated, rhs, absolute, fraction, plan = _linear_material_subcycle(state, params, params.dt / 2., 128)
    assert bool(plan.supported) and int(plan.count) > 1
    height = np.asarray(params.dz_node).reshape(-1).copy()
    height[0] += float(state.eta[0, 0])
    distances = np.asarray(params.dz_iface).reshape(-1)
    matrix = np.zeros((4, 4))
    for index in range(3):
        conductance = params.kappa_v / distances[index]
        matrix[index, index] -= conductance / height[index]
        matrix[index, index + 1] += conductance / height[index]
        matrix[index + 1, index] += conductance / height[index + 1]
        matrix[index + 1, index + 1] -= conductance / height[index + 1]
    duration = params.dt / (2. * int(plan.count))
    heun = np.eye(4) + duration * matrix + .5 * duration ** 2 * (matrix @ matrix)
    expected = np.linalg.matrix_power(heun, int(plan.count)) @ profile
    np.testing.assert_allclose(updated.T, np.broadcast_to(expected, state.T.shape), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_contents(updated, params) - _contents(state, params), rhs, rtol=1e-10, atol=1e-12)
    assert float(fraction) <= .5
    assert np.min(np.asarray(updated.T)) >= 1. and np.max(np.asarray(updated.T)) <= 10.
    assert np.all(np.asarray(absolute) >= 0.)


@pytest.mark.parametrize("defect", ["capacity", "nonpositive"])
def test_uncut_schedule_rejects_without_partial_tracer_update_or_source_clock(defect):
    _, (_, initialize, _, params, _) = _material_fixture()
    params = params._replace(kappa_conv=.01, Q_heat_2d=jnp.full_like(params.Q_heat_2d, 100.))
    state = _state(initialize, eta=-2.499 if defect == "capacity" else -2.5)
    result = make_material_top_step(params, subcycle_scheme=POLICY, max_subcycles=2)(state)
    assert not bool(result.valid)
    assert not bool(result.checks["nonlinear_schedule_supported"])
    assert int(result.checks["nonlinear_active_subcycles"]) == 0
    if defect == "capacity":
        assert float(result.checks["nonlinear_required_subcycles"]) > 2.
    np.testing.assert_array_equal(result.budget["source_inputs"], 0.)
    for name in state._fields:
        assert np.asarray(getattr(result.state, name)).tobytes() == np.asarray(getattr(state, name)).tobytes()


@pytest.mark.parametrize("scheme,maximum", [("unknown", 128), (POLICY, 0), (POLICY, 1.5), (POLICY, True)])
def test_policy_validation_cannot_silently_change_or_clip_configuration(scheme, maximum):
    _, (_, _, _, params, _) = _material_fixture()
    with pytest.raises(ValueError, match="subcycle"):
        make_material_top_step(params, subcycle_scheme=scheme, max_subcycles=maximum)


@pytest.mark.parametrize("name,value", [("kappa_v", -1.), ("kappa_conv", np.nan),
                                      ("lambda_bulk", -1.), ("dt", 0.)])
def test_invalid_physical_coefficients_are_rejected_at_construction(name, value):
    _, (_, _, _, params, _) = _material_fixture()
    with pytest.raises(ValueError, match="finite|nonnegative|positive"):
        make_material_top_step(params._replace(**{name: value}), subcycle_scheme=POLICY)


def test_integer_schedule_crossings_are_exposed_not_called_smooth():
    _, (_, _, _, params, _) = _material_fixture()
    rate = jnp.ones_like(params.wet_mask_z) * .002
    lower = jnp.broadcast_to(params.dz_node, params.wet_mask_z.shape)
    counts = []
    for thickness in [.040001, .039999]:
        plan = _subcycle_plan(rate, lower.at[..., 0].set(thickness), 10., 1, params, 128)
        counts.append(int(plan.count))
    assert counts == [1, 2]


def test_endpoint_uses_authoritative_volume_without_compensating_content():
    _, (_, initialize, _, params, _) = _material_fixture()
    state = _state(initialize, eta=-2.4825)
    zero = jnp.zeros_like(state.T)
    faces = zero, zero
    endpoint = state.eta
    for _ in range(4):
        endpoint = jnp.nextafter(endpoint, jnp.full_like(endpoint, jnp.inf))
    plan = _nonlinear_subcycle_plan(state, params, faces, 128, endpoint_eta=endpoint)
    correct = _material_tracer_step(state, params, faces, subcycle_plan=plan, endpoint_eta=endpoint)
    unaligned = _material_tracer_step(state, params, faces, subcycle_plan=plan)
    before = _contents(state, params)
    expected = before + correct[1]
    floor = 64. * np.finfo(float).eps * (1. + np.abs(np.asarray(before)) + np.abs(np.asarray(expected)))
    assert np.max(np.abs(np.asarray(_contents(correct[0], params) - expected)) / floor) <= 1.
    wrong_geometry = _contents(unaligned[0]._replace(eta=endpoint), params)
    assert np.max(np.abs(np.asarray(wrong_geometry - before - unaligned[1])) / floor) > 1.
    np.testing.assert_array_equal(correct[1], 0.)
    np.testing.assert_array_equal(correct[3], 0.)
    np.testing.assert_array_equal(correct[0].eta, endpoint)


def _flowing_case():
    physics = replace(PhysicsConfig(), nu_h=0., nu_v=0., nu_bi=0., kappa_h=100., kappa_v=1e-5,
                      kappa_bi=0., kappa_conv=.01, kappa_gm=0., kappa_redi=0., r_bot=0.)
    grid, (_, initialize, _, params, _) = _material_fixture(stairs=True, physics=physics,
                                                           lambda_bulk=20., T_atm=np.full((8, 8), 5.))
    generator = np.random.default_rng(299112)
    state = _state(initialize, temperature=10., salinity=34.7)
    state = state._replace(T=state.T - .2 * jnp.arange(state.T.shape[-1])[None, None, :],
                           eta=(jnp.full_like(state.eta, -2.4825)
                                + jnp.asarray(generator.normal(size=state.eta.shape) * 1e-4)) * params.wet_mask,
                           u=jnp.asarray(generator.normal(size=state.u.shape) * .05) * params.wet_mask_z,
                           v=jnp.asarray(generator.normal(size=state.v.shape) * .05) * params.wet_mask_z * params.interior_mask_z)
    return grid, params, state


def test_three_actual_flowing_steps_have_fixed_schedule_tangent_and_adjoint():
    _, params, initial = _flowing_case()
    advance = make_material_top_step(params, subcycle_scheme=POLICY)
    zero = jnp.zeros_like(params.Q_heat_2d)

    def response(control):
        state = initial._replace(T=initial.T + control[0], eta=initial.eta + control[1] * params.wet_mask)
        for _ in range(3):
            state = advance(state, (zero, zero, zero + control[2])).state
        return jnp.asarray([jnp.mean(state.T[..., 0]), jnp.mean(state.u), jnp.mean(state.S[..., 0])])

    control = jnp.asarray([.1, .0002, 100.])
    direction = jnp.asarray([.3, 1e-5, 10.])
    cotangent = jnp.asarray([.7, -.2, -.4])
    for perturbation in [0., -1e-3, 1e-3]:
        selected = control + perturbation * direction
        state = initial._replace(T=initial.T + selected[0], eta=initial.eta + selected[1] * params.wet_mask)
        for _ in range(3):
            result = advance(state, (zero, zero, zero + selected[2]))
            assert bool(result.valid), result.checks
            assert int(result.checks["nonlinear_active_subcycles"]) == 3
            state = result.state
    _, tangent = jax.jvp(response, (control,), (direction,))
    _, pullback = jax.vjp(response, control)
    adjoint = pullback(cotangent)[0]
    assert float(jnp.dot(tangent, cotangent)) == pytest.approx(float(jnp.dot(direction, adjoint)), abs=1e-11)
    difference = (response(control + .001 * direction) - response(control - .001 * direction)) / .002
    np.testing.assert_allclose(tangent, difference, rtol=1e-5, atol=2e-8)


def test_two_flowing_forced_restarts_preserve_bytes_and_reject_policy_or_cap(tmp_path):
    grid, params, initial = _flowing_case()
    policy = dict(subcycle_scheme=POLICY, max_subcycles=128)
    advance = make_material_top_step(params, **policy)
    forcing = {"Q_sequence": np.arange(6.) * 20. + 100.}
    contract = make_material_top_restart_contract(grid, params, forcing=forcing, controls={"calendar": "fixed"},
                                                 execution={"backend": jax.default_backend()}, **policy)

    def run(restart):
        state, totals = initial, {"source": np.zeros((6, 2)), "residual": np.zeros(2)}
        history = []
        zero = jnp.zeros_like(params.Q_heat_2d)
        for index, heat in enumerate(forcing["Q_sequence"]):
            result = advance(state, (zero + .0001 * index, zero, zero + heat))
            assert bool(result.valid), result.checks
            state = result.state
            totals["source"] += np.asarray(result.budget["source_inputs"])
            totals["residual"] += np.asarray(result.budget["source_budget_residual"])
            history.append(int(result.checks["nonlinear_active_subcycles"]))
            if restart and index in {1, 3}:
                path = tmp_path / f"flow_{index}.npz"
                save_restart(path, state, contract, step=index + 1, counters={"accepted_steps": index + 1},
                             cumulative=totals, history={"nonlinear_count": history})
                loaded = load_restart(path, contract)
                state = JaxStateG(**{name: jnp.asarray(value) for name, value in loaded.state.items()})
                totals, history = loaded.cumulative, list(loaded.history["nonlinear_count"])
        return state, totals, np.asarray(history)

    continuous, resumed = run(False), run(True)
    for before, after in zip(jax.tree.leaves(continuous), jax.tree.leaves(resumed), strict=True):
        assert np.asarray(before).tobytes() == np.asarray(after).tobytes()
    for scheme, maximum in [("reference_static_v1", 128), (POLICY, 64)]:
        mismatch = make_material_top_restart_contract(grid, params, forcing=forcing, controls={"calendar": "fixed"},
                                                       execution={"backend": jax.default_backend()},
                                                       subcycle_scheme=scheme, max_subcycles=maximum)
        with pytest.raises(ValueError, match="contract"):
            load_restart(tmp_path / "flow_1.npz", mismatch)
