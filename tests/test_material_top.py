"""Independent moving-node stock gates; these are not industrial qualification."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_legacy_reference_geometry import WIDTHS, _fixture

from config import C_P, RHO_0, PhysicsConfig
from jax_solver_global import JaxStateG
from material_top import (
    _contents,
    _linear_material_step,
    _material_tracer_step,
    make_material_top_restart_contract,
    make_material_top_step,
    material_inventory,
    material_thickness,
)
from restart_contract import load_restart, make_restart_contract, save_restart


def _material_fixture(**options):
    settings = dict(process_time_scheme="symmetric_fast_v3", match_barotropic_transport=True,
                    monotone_adv=True, use_scan=True)
    settings.update(options)
    return _fixture(**settings)


def _state(initialize, eta=0., temperature=15., salinity=35.):
    state = initialize()
    return state._replace(T=jnp.full_like(state.T, temperature), S=jnp.full_like(state.S, salinity),
                          eta=jnp.full_like(state.eta, eta))


def _numpy_divergence(east, north, params):
    incoming_north = np.roll(north, 1, axis=1)
    incoming_north[:, 0] = 0.
    return ((east - np.roll(east, 1, axis=0)) / np.asarray(params.dx_2d)[..., None]
            + (north - incoming_north) / (float(params.dy) * np.asarray(params.cos_lat)[None, :, None]))


def _numpy_transport_rhs(concentration, faces, params):
    wet = np.asarray(params.wet_mask_z)
    east, north = faces
    vertical = np.flip(np.cumsum(np.flip(_numpy_divergence(east, north, params), axis=-1), axis=-1), axis=-1)
    internal = vertical[..., 1:] * np.where(vertical[..., 1:] >= 0., concentration[..., :-1], concentration[..., 1:])
    internal *= wet[..., :-1] * wet[..., 1:]
    zero = np.zeros_like(concentration[..., :1])
    out_z = np.concatenate((internal, zero), axis=-1) - np.concatenate((zero, internal), axis=-1)
    tracer_east = east * np.where(east >= 0., concentration, np.roll(concentration, -1, axis=0))
    tracer_north = north * np.where(north >= 0., concentration, np.roll(concentration, -1, axis=1))
    return -_numpy_divergence(tracer_east, tracer_north, params) - out_z


@pytest.mark.parametrize("eta", [0., .5, 2.5, -1.25])
def test_actual_complete_step_applies_heat_once_to_the_moving_capacity(eta):
    _, (_, initialize, _, params, _) = _material_fixture()
    params = params._replace(Q_heat_2d=jnp.full_like(params.Q_heat_2d, 100.))
    state = _state(initialize, eta=eta)
    result = make_material_top_step(params)(state)
    assert bool(result.valid), result.checks
    area = np.sum(np.asarray(params.dx_2d) * float(params.dy))
    flux = np.asarray(result.budget["observed_change"])[0] / (area * params.dt)
    assert flux == pytest.approx(100., abs=1e-7)
    assert result.budget["source_inputs"][0, 0] / (area * params.dt) == pytest.approx(100., abs=1e-12)
    np.testing.assert_array_equal(result.attempted_state.eta, state.eta)
    assert np.max(np.abs(np.asarray(result.budget["source_budget_residual"]))) / (area * params.dt) < 1e-7
    expected = 15. + params.dt * 100. / (RHO_0 * C_P * (2.5 + eta))
    np.testing.assert_allclose(result.state.T[..., 0], expected, rtol=0., atol=1e-13)
    np.testing.assert_array_equal(result.state.S, state.S)
    wrong_capacity_flux = 100. * (2.5 + eta) / 2.5
    if eta != 0.:
        assert abs(wrong_capacity_flux - flux) > 1.


@pytest.mark.parametrize("stairs", [False, True])
@pytest.mark.parametrize("constant", [False, True])
def test_shared_transport_matches_independent_moving_heun_and_preserves_bounds(stairs, constant):
    _, (_, initialize, _, params, _) = _material_fixture(stairs=stairs)
    generator = np.random.default_rng(29103)
    state = _state(initialize)
    wet = np.asarray(params.wet_mask_z).astype(bool)
    eta = generator.uniform(-.3, .3, size=state.eta.shape) * wet[..., 0]
    temperature = np.full(state.T.shape, 7.) if constant else generator.uniform(2., 15., state.T.shape)
    salinity = np.full(state.S.shape, 34.7) if constant else generator.uniform(30., 36., state.S.shape)
    temperature[~wet], salinity[~wet] = 999., -999.
    state = state._replace(T=jnp.asarray(temperature), S=jnp.asarray(salinity), eta=jnp.asarray(eta))
    east = generator.normal(size=state.T.shape) * .0003 * WIDTHS * np.asarray(params.dx_2d)[..., None]
    north = generator.normal(size=state.T.shape) * .00015 * WIDTHS * float(params.dy) * np.asarray(params.cos_lat)[None, :, None]
    east *= wet * np.roll(wet, -1, axis=0)
    north *= wet * np.roll(wet, -1, axis=1)
    north[:, -1] = 0.
    faces = east, north
    actual = _material_tracer_step(state, params, tuple(jnp.asarray(flux) for flux in faces))
    expected_eta = eta.copy()
    expected_thickness = np.broadcast_to(WIDTHS, state.T.shape).copy()
    expected_thickness[..., 0] += eta
    concentrations = [temperature.copy(), salinity.copy()]
    eta_rate = -np.sum(_numpy_divergence(*faces, params), axis=-1)
    duration = params.dt / max(params.adv_nsub, params.conv_nsub)
    old_thickness_predictor = []
    for _ in range(max(params.adv_nsub, params.conv_nsub)):
        next_eta = expected_eta + duration * eta_rate
        next_thickness = expected_thickness.copy()
        next_thickness[..., 0] += duration * eta_rate
        for index, concentration in enumerate(concentrations):
            content = expected_thickness * np.where(wet, concentration, 0.)
            first = _numpy_transport_rhs(concentration, faces, params)
            predicted = (content + duration * first) / np.where(wet, next_thickness, 1.)
            second = _numpy_transport_rhs(predicted, faces, params)
            concentrations[index] = np.where(wet, (content + .5 * duration * (first + second))
                                             / np.where(wet, next_thickness, 1.), concentration)
            if index == 0:
                wrong_predicted = (content + duration * first) / expected_thickness
                wrong_second = _numpy_transport_rhs(wrong_predicted, faces, params)
                old_thickness_predictor.append(np.where(wet, .5 * duration * (wrong_second - second), 0.))
        expected_eta, expected_thickness = next_eta, next_thickness
    np.testing.assert_allclose(actual[0].eta, expected_eta, rtol=0., atol=1e-14)
    for observed, expected, initial in zip((actual[0].T, actual[0].S), concentrations,
                                          (temperature, salinity), strict=True):
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(np.asarray(observed)[~wet], initial[~wet])
        assert np.min(np.asarray(observed)[wet]) >= np.min(initial[wet]) - 1e-12
        assert np.max(np.asarray(observed)[wet]) <= np.max(initial[wet]) + 1e-12
    assert np.max(np.abs(np.asarray(actual[0].eta - state.eta))) > .001
    assert max(np.max(np.abs(change)) for change in old_thickness_predictor) > 1e-7
    assert np.max(np.asarray(actual[-2])) < .5
    difference = np.asarray(_contents(actual[0], params) - _contents(state, params))
    np.testing.assert_allclose(difference, actual[1], rtol=1e-9, atol=1e-12)
    area = np.asarray(params.dx_2d) * float(params.dy)
    global_transport = np.sum(np.asarray(actual[4]) * area[..., None, None], axis=(0, 1, 2))
    absolute_transport = np.sum(np.abs(np.asarray(actual[4])) * area[..., None, None], axis=(0, 1, 2))
    assert np.all(np.abs(global_transport) <= 64. * np.finfo(float).eps * absolute_transport)


def test_mixing_exchanges_moving_inventory_not_fixed_concentration_stock():
    physics = replace(PhysicsConfig(), nu_h=0., nu_v=0., nu_bi=0., kappa_h=200.,
                      kappa_v=.001, kappa_bi=0., kappa_conv=0., kappa_gm=0., kappa_redi=0., r_bot=0.)
    _, (_, initialize, _, params, _) = _material_fixture(stairs=True, physics=physics)
    generator = np.random.default_rng(29382)
    state = _state(initialize)
    wet = np.asarray(params.wet_mask_z).astype(bool)
    state = state._replace(T=jnp.asarray(np.where(wet, generator.uniform(2., 18., state.T.shape), 1e6)),
                           S=jnp.asarray(np.where(wet, generator.uniform(32., 36., state.S.shape), -1e6)),
                           eta=jnp.asarray(generator.uniform(-1., 1., state.eta.shape) * wet[..., 0]))
    updated, rhs, absolute, fraction = _linear_material_step(state, params, params.dt / 2.)
    assert float(fraction) < .5
    difference = np.asarray(_contents(updated, params) - _contents(state, params))
    np.testing.assert_allclose(difference, rhs, atol=1e-12, rtol=1e-9)
    area = np.asarray(params.dx_2d) * float(params.dy)
    net = np.sum(np.asarray(rhs) * area[..., None, None], axis=(0, 1, 2))
    scale = np.sum(np.asarray(absolute) * area[..., None, None], axis=(0, 1, 2))
    assert np.all(np.abs(net) < 64. * np.finfo(float).eps * scale)
    np.testing.assert_array_equal(np.asarray(updated.T)[~wet], np.asarray(state.T)[~wet])
    for before, after in ((state.T, updated.T), (state.S, updated.S)):
        assert np.min(np.asarray(after)[wet]) >= np.min(np.asarray(before)[wet])
        assert np.max(np.asarray(after)[wet]) <= np.max(np.asarray(before)[wet])


@pytest.mark.parametrize("setting,value", [("polar_cap_rows", 1), ("dynamic_ice", True),
                                          ("kappa_gm", 1.), ("kappa_redi", 1.),
                                          ("conservative_kv", False), ("monotone_adv", False),
                                          ("ice_salt_flux", 1e-5),
                                          ("localize_conv", False), ("eta_relax_rate", 1e-5)])
def test_unsupported_configuration_is_explicitly_rejected(setting, value):
    _, (_, _, _, params, _) = _material_fixture()
    with pytest.raises(ValueError, match="requires"):
        make_material_top_step(params._replace(**{setting: value}))


def test_float32_and_wrong_state_shape_are_not_implicitly_promoted():
    _, (_, initialize, _, params, _) = _material_fixture(dtype="float32")
    with pytest.raises(ValueError, match="float64"):
        make_material_top_step(params)
    _, (_, initialize, _, params, _) = _material_fixture()
    advance = make_material_top_step(params)
    with pytest.raises(ValueError, match="float64 T"):
        advance(initialize()._replace(T=initialize().T.astype(jnp.float32)))
    with pytest.raises(ValueError, match="shapes"):
        advance(initialize()._replace(S=initialize().S[..., :1]))


@pytest.mark.parametrize("defect", ["depleted", "ice", "nan", "speed", "diffusion_cfl"])
def test_first_invalid_attempt_preserves_authoritative_state_bytes(defect):
    _, (_, initialize, _, params, _) = _material_fixture()
    state = _state(initialize)
    if defect == "depleted":
        state = state._replace(eta=jnp.full_like(state.eta, -2.5))
    elif defect == "ice":
        state = state._replace(ice=jnp.full_like(state.ice, .1))
    elif defect == "nan":
        state = state._replace(T=state.T.at[2, 3, 0].set(jnp.nan))
    elif defect == "speed":
        state = state._replace(u=jnp.full_like(state.u, 11.))
    elif defect == "diffusion_cfl":
        params = params._replace(kappa_v=20.)
    result = make_material_top_step(params)(state)
    assert not bool(result.valid)
    for name in state._fields:
        assert np.asarray(getattr(result.state, name)).tobytes() == np.asarray(getattr(state, name)).tobytes()
    if defect == "diffusion_cfl":
        assert float(result.checks["linear_diffusion_fraction_max"]) > .5


def test_surface_bulk_sources_use_actual_predictor_sst_and_are_not_booked_twice():
    _, (_, initialize, _, params, _) = _material_fixture(lambda_bulk=80., T_atm=np.full((8, 8), 5.))
    state = _state(initialize, eta=.5, temperature=10.)
    result = make_material_top_step(params)(state)
    assert bool(result.valid), result.checks
    capacity = RHO_0 * C_P * 3.
    first_flux = 80. * (5. - 10.)
    predicted_temperature = 10. + params.dt * first_flux / capacity
    expected_flux = .5 * (first_flux + 80. * (5. - predicted_temperature))
    area = np.sum(np.asarray(params.dx_2d) * float(params.dy))
    assert float(result.budget["source_inputs"][1, 0]) / (area * params.dt) == pytest.approx(expected_flux, abs=1e-10)
    assert float(result.budget["observed_change"][0]) / (area * params.dt) == pytest.approx(expected_flux, abs=1e-7)
    assert abs(float(result.budget["observed_change"][0]) / (area * params.dt) - 2. * expected_flux) > 100.


@pytest.mark.parametrize("limited", [False, True])
def test_actual_coupled_step_keeps_constant_tracers_while_the_surface_moves(limited):
    _, (_, initialize, _, params, _) = _material_fixture(stairs=True, fct_adv=limited,
                                                        monotone_adv=not limited)
    generator = np.random.default_rng(29183)
    state = _state(initialize, temperature=7., salinity=34.7)
    state = state._replace(u=jnp.asarray(generator.normal(size=state.u.shape) * .05) * params.wet_mask_z,
                           v=jnp.asarray(generator.normal(size=state.v.shape) * .05) * params.wet_mask_z * params.interior_mask_z,
                           eta=jnp.asarray(generator.uniform(-.1, .1, state.eta.shape)) * params.wet_mask)
    result = make_material_top_step(params)(state)
    assert bool(result.valid), result.checks
    wet = np.asarray(params.wet_mask_z) > 0.
    np.testing.assert_allclose(np.asarray(result.state.T)[wet], 7., rtol=0., atol=8e-12)
    np.testing.assert_allclose(np.asarray(result.state.S)[wet], 34.7, rtol=0., atol=3.57e-11)
    assert np.max(np.abs(np.asarray(result.state.eta - state.eta))) > 1e-7


def test_biharmonic_exchange_conserves_inventory_without_a_monotonicity_claim():
    _, (_, initialize, _, params, _) = _material_fixture(stairs=True, fct_adv=True, monotone_adv=False)
    params = params._replace(kappa_bi=2e14)
    generator = np.random.default_rng(29182)
    state = _state(initialize)
    state = state._replace(T=state.T + jnp.asarray(generator.normal(size=state.T.shape)),
                           eta=jnp.asarray(generator.uniform(-1., 1., state.eta.shape)) * params.wet_mask)
    make_material_top_step(params)
    updated, rhs, absolute, fraction = _linear_material_step(state, params, params.dt / 2.)
    assert float(fraction) < .5
    np.testing.assert_allclose(_contents(updated, params) - _contents(state, params), rhs, rtol=1e-6, atol=1e-12)
    area = np.asarray(params.dx_2d) * float(params.dy)
    net = np.sum(np.asarray(rhs) * area[..., None, None], axis=(0, 1, 2))
    scale = np.sum(np.asarray(absolute) * area[..., None, None], axis=(0, 1, 2))
    assert np.all(np.abs(net) <= 64. * np.finfo(float).eps * scale)


def test_actual_complete_step_tangent_adjoint_and_resolvable_difference():
    _, (_, initialize, _, params, _) = _material_fixture()
    advance = make_material_top_step(params)
    initial = _state(initialize, eta=.5)

    def response(control):
        state = initial._replace(T=initial.T + control[0], eta=initial.eta + control[1])
        zero = jnp.zeros_like(params.Q_heat_2d)
        result = advance(state, (zero, zero, zero + control[2]))
        return jnp.asarray([jnp.mean(result.state.T[..., 0]), jnp.mean(result.state.S[..., 0])])

    control = jnp.asarray([.1, .2, 100.])
    direction = jnp.asarray([.3, -.2, 10.])
    cotangent = jnp.asarray([.7, -.4])
    _, tangent = jax.jvp(response, (control,), (direction,))
    _, pullback = jax.vjp(response, control)
    adjoint = pullback(cotangent)[0]
    assert float(jnp.dot(tangent, cotangent)) == pytest.approx(float(jnp.dot(direction, adjoint)), abs=1e-12)
    difference = (response(control + 1e-3 * direction) - response(control - 1e-3 * direction)) / .002
    np.testing.assert_allclose(tangent, difference, rtol=1e-7, atol=1e-10)


def test_two_strict_restarts_preserve_candidate_bytes_and_reject_original_contract(tmp_path):
    grid, (_, initialize, _, params, _) = _material_fixture()
    params = params._replace(Q_heat_2d=jnp.full_like(params.Q_heat_2d, 100.))
    advance = make_material_top_step(params)
    contract = make_material_top_restart_contract(grid, params, forcing={"Q": params.Q_heat_2d},
                                                 controls={"calendar": "360_day"}, execution={"backend": "cpu"})

    def run(restarts):
        state = _state(initialize, eta=.5)
        cumulative = {"sources": jnp.zeros((6, 2)), "residual": jnp.zeros(2)}
        history = []
        for index in range(5):
            result = advance(state)
            assert bool(result.valid)
            state = result.state
            cumulative["sources"] += result.budget["source_inputs"]
            cumulative["residual"] += result.budget["source_budget_residual"]
            history.append(float(result.checks["minimum_wet_thickness_m"]))
            if restarts and index in {1, 3}:
                path = tmp_path / f"material_{index}.npz"
                save_restart(path, state, contract, step=index + 1, counters={"accepted_steps": index + 1},
                             cumulative=cumulative, history={"thickness_m": history})
                loaded = load_restart(path, contract)
                state = JaxStateG(**{name: jnp.asarray(value) for name, value in loaded.state.items()})
                cumulative = {name: jnp.asarray(value) for name, value in loaded.cumulative.items()}
                history = list(loaded.history["thickness_m"])
        return state, cumulative, np.asarray(history)

    continuous, resumed = run(False), run(True)
    for before, after in zip(jax.tree.leaves(continuous), jax.tree.leaves(resumed), strict=True):
        assert np.asarray(before).tobytes() == np.asarray(after).tobytes()
    original = make_restart_contract(grid, params, dtype="float64", forcing={}, controls={}, code_paths={}, execution={})
    with pytest.raises(ValueError, match="contract"):
        load_restart(tmp_path / "material_1.npz", original)
    before = np.asarray(material_inventory(continuous[0], params))
    assert np.isfinite(before).all()
    np.testing.assert_array_equal(material_thickness(continuous[0].eta, params)[..., 0], 3.)
