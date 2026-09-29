"""Independent viscosity matrices, modal energy, time, AD and restart gates."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm
from test_legacy_reference_geometry import WIDTHS, _fixture
from test_material_subcycles import _flowing_case

from config import OMEGA, R_EARTH, RHO_0, PhysicsConfig
from jax_solver_global import JaxStateG, _linear_half_step, make_solver_global
from material_top import (
    _joint_momentum_diffusion,
    _momentum_diffusion_norm_bound,
    _momentum_diffusion_plan,
    make_material_top_restart_contract,
    make_material_top_step,
)
from restart_contract import load_restart, save_restart

POLICY = dict(subcycle_scheme="actual_geometry_v2", momentum_diffusion_scheme="joint_heun_v1")


def _controlled_factory(*, stairs=False, metric=False, duration=600., physics=None):
    grid, _ = _fixture(stairs=stairs)
    rotation = np.broadcast_to(2. * OMEGA * np.sin(np.deg2rad(grid.lat)), (8, 8)).copy() if metric else np.zeros((8, 8))
    grid = replace(grid, dx_2d=np.full((8, 8), 1000.), dy=1000., cos_lat=np.ones(8), f=rotation)
    if physics is None:
        physics = replace(PhysicsConfig(), nu_h=800., nu_v=.01625, nu_bi=7.5e7,
                          kappa_h=0., kappa_v=0., kappa_bi=0., kappa_conv=0., kappa_gm=0., kappa_redi=0., r_bot=0.)
    _, initialize, _, params, _ = make_solver_global(
        grid, physics, duration, dt_bt=duration / 24., mode_split=True, dtype="float64", return_params=True,
        nu_nsub="cfl", column_geometry="nodal_dual_v1", conservative_kv=True, localize_conv=True,
        match_barotropic_transport=True, process_time_scheme="symmetric_fast_v3", monotone_adv=True,
        polar_cap_rows=0, polar_cap_taper=0, use_scan=True)
    state = initialize()._replace(T=jnp.full((8, 8, 4), 15.), S=jnp.full((8, 8, 4), 35.))
    return grid, params, state


def _numpy_operators(params):
    shape = params.wet_mask_z.shape
    size = int(np.prod(shape))
    wet = np.asarray(params.wet_mask_z)
    horizontal, vertical = np.zeros((size, size)), np.zeros((size, size))
    widths = np.asarray(params.dz_node).ravel()
    distances = np.asarray(params.dz_iface).ravel()
    for longitude, latitude, depth in np.ndindex(shape):
        row = np.ravel_multi_index((longitude, latitude, depth), shape)
        for next_longitude in ((longitude - 1) % shape[0], (longitude + 1) % shape[0]):
            column = np.ravel_multi_index((next_longitude, latitude, depth), shape)
            rate = wet[longitude, latitude, depth] * wet[next_longitude, latitude, depth] / float(params.dx_2d[longitude, latitude]) ** 2
            horizontal[row, column] += rate
            horizontal[row, row] -= rate
        for next_latitude in (max(latitude - 1, 0), min(latitude + 1, shape[1] - 1)):
            column = np.ravel_multi_index((longitude, next_latitude, depth), shape)
            rate = wet[longitude, latitude, depth] * wet[longitude, next_latitude, depth] / float(params.dy) ** 2
            horizontal[row, column] += rate
            horizontal[row, row] -= rate
        metric = float(params.f[0, latitude]) / (2. * OMEGA * float(params.cos_lat[latitude]) * R_EARTH)
        north = np.ravel_multi_index((longitude, min(latitude + 1, shape[1] - 1), depth), shape)
        south = np.ravel_multi_index((longitude, max(latitude - 1, 0), depth), shape)
        horizontal[row, north] -= metric / (2. * float(params.dy))
        horizontal[row, south] += metric / (2. * float(params.dy))
        for next_depth in (depth - 1, depth + 1):
            if 0 <= next_depth < shape[2] and wet[longitude, latitude, depth] and wet[longitude, latitude, next_depth]:
                column = np.ravel_multi_index((longitude, latitude, next_depth), shape)
                rate = 1. / (widths[depth] * distances[min(depth, next_depth)])
                vertical[row, column] += rate
                vertical[row, row] -= rate
    operator = (params.nu_h * horizontal + params.nu_v * vertical - params.nu_bi * horizontal @ horizontal) * wet.ravel()[:, None]
    return horizontal, vertical, operator


@pytest.mark.parametrize("stairs,metric", [(False, False), (True, False), (True, True)])
def test_every_joint_heun_stage_matches_independent_dense_operator(stairs, metric):
    _, params, state = _controlled_factory(stairs=stairs, metric=metric)
    _, _, operator = _numpy_operators(params)
    bound = float(_momentum_diffusion_norm_bound(params))
    assert np.max(np.sum(np.abs(operator), axis=1)) <= bound * (1. + 1e-14)
    plan, fraction = _momentum_diffusion_plan(params, params.dt / 2., 128)
    count = max(1, int(np.ceil(params.dt / 2. * bound)))
    assert int(plan.required) == count and int(plan.count) == count and bool(plan.supported)
    assert float(fraction) <= .5
    generator = np.random.default_rng(292918)
    velocity = generator.normal(size=state.u.shape)
    velocity[np.asarray(params.wet_mask_z) == 0.] = 123.
    state = state._replace(u=jnp.asarray(velocity), v=jnp.asarray(-.3 * velocity))
    actual = jax.jit(lambda current: _joint_momentum_diffusion(current, params, params.dt / 2., 128))(state)
    expected = velocity.ravel().copy()
    duration = params.dt / (2. * count)
    for _ in range(count):
        first = operator @ expected
        predicted = expected + duration * first
        expected += .5 * duration * (first + operator @ predicted)
    np.testing.assert_allclose(actual[0], expected.reshape(state.u.shape), rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(actual[1], -.3 * expected.reshape(state.u.shape), rtol=1e-11, atol=1e-13)
    dry = np.asarray(params.wet_mask_z) == 0.
    np.testing.assert_array_equal(np.asarray(actual[0])[dry], velocity[dry])


def test_joint_stages_remove_combined_growth_without_changing_the_spatial_generator():
    _, params, state = _controlled_factory()
    assert params.nu_nsub == 2 and params.n_subcyc == 24
    _, vertical, _ = _numpy_operators(params)
    column = vertical[:4, :4]
    square_root = np.sqrt(WIDTHS)
    eigenvalues, eigenvectors = np.linalg.eigh(column * square_root[:, None] / square_root[None, :])
    vertical_mode = eigenvectors[:, 0] / square_root
    vertical_mode /= np.max(np.abs(vertical_mode))
    mode = (-1.) ** np.arange(8)[:, None, None] * np.cos(7. * np.pi * (np.arange(8) + .5) / 8.)[None, :, None] * vertical_mode[None, None, :]
    horizontal_eigenvalue = (-4. - 4. * np.sin(7. * np.pi / 16.) ** 2) / 1e6
    generator = params.nu_h * horizontal_eigenvalue + params.nu_v * eigenvalues[0] - params.nu_bi * horizontal_eigenvalue ** 2
    plan, _ = _momentum_diffusion_plan(params, params.dt / 2., 128)
    scaled = generator * params.dt / (2. * int(plan.count))
    amplification = (1. + scaled + .5 * scaled ** 2) ** int(plan.count)
    assert generator < 0. and 0. < amplification < 1.
    weights = np.asarray(params.dx_2d)[..., None] * params.dy * WIDTHS
    for amplitude in (1e-8, 1e-10, 1e-12):
        initial = state._replace(u=jnp.asarray(mode * amplitude))
        old_half = _linear_half_step(initial, params, params.dt / 2.)
        half_velocity = _joint_momentum_diffusion(initial, params, params.dt / 2., 128)[0]
        result = make_material_top_step(params, **POLICY)(initial)
        assert bool(result.valid), result.checks
        np.testing.assert_allclose(half_velocity / amplitude, mode * amplification, rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(result.state.u / amplitude, mode * amplification ** 2, rtol=1e-11, atol=1e-13)
        before = .5 * RHO_0 * np.sum(weights * np.asarray(initial.u) ** 2)
        old_energy = .5 * RHO_0 * np.sum(weights * np.asarray(old_half.u) ** 2)
        after = .5 * RHO_0 * np.sum(weights * (np.asarray(result.state.u) ** 2 + np.asarray(result.state.v) ** 2))
        assert old_energy / before == pytest.approx(3.522760274093153, rel=1e-12)
        assert after <= before and after / before == pytest.approx(amplification ** 4, rel=1e-11)
        np.testing.assert_allclose(result.state.eta, initial.eta, rtol=0., atol=1e-14)


@pytest.mark.parametrize("mixed", [False, True])
def test_complete_column_velocity_converges_second_order_to_independent_exponential(mixed):
    physics = replace(PhysicsConfig(), nu_h=0., nu_v=.01, nu_bi=0., kappa_h=0., kappa_v=0.,
                      kappa_bi=0., kappa_conv=0., kappa_gm=0., kappa_redi=0., r_bot=.001 if mixed else 0.)
    _, base, initial = _controlled_factory(duration=40., physics=physics)
    initial = initial._replace(u=jnp.broadcast_to(jnp.array([-.04, .28, .36, .12]), initial.u.shape))
    _, vertical, _ = _numpy_operators(base)
    operator = .01 * vertical[:4, :4]
    operator[-1, -1] -= physics.r_bot
    augmented = np.zeros((5, 5))
    augmented[:4, :4] = operator
    if mixed:
        base = base._replace(tau_x_2d=jnp.full_like(base.tau_x_2d, .1))
        augmented[0, -1] = .1 / (RHO_0 * WIDTHS[0])
    vector = np.r_[np.asarray(initial.u)[0, 3], 1.]
    reference = (expm(400. * augmented) @ vector)[:4]
    half_reference = (expm(200. * augmented) @ expm(200. * augmented) @ vector)[:4]
    reference_gap = np.max(np.abs(reference - half_reference))
    errors = []
    for duration in (40., 20., 10.):
        params = base._replace(dt=duration, dt_bt=duration / 8., n_subcyc=8)
        advance = make_material_top_step(params, **POLICY)

        def body(unused_index, carry):
            current, valid = carry
            result = advance(current)
            return result.state, valid & result.valid

        current, valid = jax.jit(lambda current: jax.lax.fori_loop(
            0, int(400. / duration), body, (current, jnp.asarray(True))))(initial)
        assert bool(valid)
        errors.append(np.sqrt(np.sum(WIDTHS * (np.asarray(current.u)[0, 3] - reference) ** 2) / WIDTHS.sum()))
    assert reference_gap < .01 * errors[-1]
    assert errors[0] / errors[1] >= 3.3 and errors[1] / errors[2] >= 3.3, errors


def test_constancy_and_unsupported_schedule_roll_back_without_clipping():
    _, params, initial = _controlled_factory()
    constant = initial._replace(u=jnp.full_like(initial.u, .2))
    velocity = _joint_momentum_diffusion(constant, params, params.dt / 2., 128)
    np.testing.assert_array_equal(velocity[0], constant.u)
    limited = make_material_top_step(params, **POLICY, max_subcycles=1)(initial)
    assert not bool(limited.valid)
    assert not bool(limited.checks["momentum_schedule_supported"])
    assert int(limited.checks["momentum_required_subcycles"]) > 1
    assert int(limited.checks["momentum_active_subcycles"]) == 0
    for actual, expected in zip(limited.state, initial, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_joint_velocity_jvp_difference_and_vjp_duality_keep_the_fixed_operator():
    _, params, initial = _controlled_factory(stairs=True, metric=True)
    generator = np.random.default_rng(292919)
    control = jnp.asarray(generator.normal(size=initial.u.shape))
    direction = jnp.asarray(generator.normal(size=initial.u.shape))
    cotangent = jnp.asarray(generator.normal(size=initial.u.shape))

    @jax.jit
    def response(velocity):
        return _joint_momentum_diffusion(initial._replace(u=velocity), params, params.dt / 2., 128)[0]

    _, tangent = jax.jvp(response, (control,), (direction,))
    _, pullback = jax.vjp(response, control)
    adjoint = pullback(cotangent)[0]
    assert float(jnp.vdot(tangent, cotangent)) == pytest.approx(float(jnp.vdot(direction, adjoint)), rel=1e-11, abs=1e-11)
    difference = (response(control + .001 * direction) - response(control - .001 * direction)) / .002
    np.testing.assert_allclose(tangent, difference, rtol=1e-9, atol=1e-10)


def test_default_candidate_keeps_the_component_scheme_and_rejects_bad_joint_policies():
    _, params, initial = _controlled_factory()
    default = make_material_top_step(params, subcycle_scheme="actual_geometry_v2")(initial)
    explicit = make_material_top_step(params, subcycle_scheme="actual_geometry_v2", momentum_diffusion_scheme="legacy_component_v1")(initial)
    for actual, expected in zip(jax.tree.leaves(default), jax.tree.leaves(explicit), strict=True):
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    for scheme, subcycle in [("unknown", "actual_geometry_v2"), ("joint_heun_v1", "reference_static_v1")]:
        with pytest.raises(ValueError, match="momentum diffusion"):
            make_material_top_step(params, momentum_diffusion_scheme=scheme, subcycle_scheme=subcycle)


def test_forced_two_restarts_preserve_bytes_and_reject_wrong_momentum_scheme(tmp_path):
    grid, params, initial = _flowing_case()
    params = params._replace(nu_v=.001)
    advance = make_material_top_step(params, **POLICY)
    contract_options = dict(forcing={"heat": [100., 110., 120., 130., 140., 150.]},
                            controls={"calendar": "fixed"}, execution={"backend": jax.default_backend()})
    contract = make_material_top_restart_contract(grid, params, **contract_options, **POLICY)
    assert contract["controls"]["material_momentum_diffusion_scheme"] == "joint_heun_v1"

    def run(restart):
        current, totals, history = initial, {}, []
        zero = jnp.zeros_like(params.Q_heat_2d)
        for index in range(6):
            result = advance(current, (zero + .0001 * index, zero, zero + 100. + 10. * index))
            assert bool(result.valid), result.checks
            current = result.state
            for name, value in result.budget.items():
                totals[name] = totals.get(name, np.zeros_like(value)) + np.asarray(value)
            history.append(int(result.checks["momentum_active_subcycles"]))
            if restart and index in {1, 3}:
                path = tmp_path / f"joint_{index}.npz"
                save_restart(path, current, contract, step=index + 1, cumulative=totals,
                             counters={"accepted_steps": index + 1}, history={"momentum_count": history})
                loaded = load_restart(path, contract)
                current = JaxStateG(**{name: jnp.asarray(value) for name, value in loaded.state.items()})
                totals, history = loaded.cumulative, list(loaded.history["momentum_count"])
        return current, totals, np.asarray(history)

    continuous, resumed = run(False), run(True)
    for actual, expected in zip(jax.tree.leaves(continuous), jax.tree.leaves(resumed), strict=True):
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    wrong = make_material_top_restart_contract(grid, params, **contract_options, subcycle_scheme="actual_geometry_v2")
    with pytest.raises(ValueError, match="contract"):
        load_restart(tmp_path / "joint_1.npz", wrong)
    with pytest.raises(ValueError, match="frozen"):
        make_material_top_restart_contract(grid, params, **{**contract_options, "controls": {"material_momentum_diffusion_scheme": "joint_heun_v1"}}, **POLICY)
