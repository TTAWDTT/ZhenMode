"""Nonzero moving-stock prototypes; not historical/industrial qualification."""
import numpy as np
import pytest

from research.experiments.material_top_band import dynamic as d


def geometry():
    nx, ny = 8, 4
    cosine = np.cos(np.deg2rad(np.linspace(-30., 30., ny)))
    dx = np.broadcast_to(6371000 * 2 * np.pi / nx * cosine[None, :], (nx, ny)).copy()
    return d.Geometry(dx, 6371000 * np.deg2rad(20.), cosine,
                      np.array([2.5, 7.5, 12.5, 42.5, 235., 200.]))


def moving_state(grid, constant=False):
    x = 2 * np.pi * np.arange(8)[:, None] / 8
    eta = -.4 + .02 * np.cos(x) * np.ones((8, 4))
    h = grid.thickness(eta)
    values = np.zeros(h.shape + (4,))
    values[..., 0] = 15. if constant else 15. - np.arange(6)[None, None, :] + .05 * np.cos(x)[..., None]
    values[..., 1] = 35. if constant else 35. + .01 * np.sin(x)[..., None]
    values[..., 2] = .025 if constant else (.02 + .01 * np.sin(x)[..., None]) * np.array([1., 1., 1., .6, .3, .1])
    values[..., 3] = 0.
    values[..., 2:] *= d.RHO
    return d.State(h, h[..., None] * values, grid.bottom)


def test_ale_surface_and_fixed_deep_contract():
    grid = geometry()
    state = moving_state(grid)
    faces = d.faces(state, grid)
    divergence, eta_rate, interface = d.ale_rates(state, grid, faces)
    assert np.max(abs(eta_rate)) > 0
    np.testing.assert_array_equal(interface[..., 0], 0.)
    np.testing.assert_array_equal(interface[..., -1], 0.)
    derivative = -divergence + interface[..., :-1] - interface[..., 1:]
    np.testing.assert_allclose(derivative[..., :3], eta_rate[..., None] * grid.fractions,
                               rtol=0, atol=1e-18)
    np.testing.assert_allclose(derivative[..., 3:], 0., rtol=0, atol=1e-18)
    np.testing.assert_allclose(interface[..., 3], divergence[..., 3:].sum(-1), rtol=0, atol=1e-18)


def test_constant_specific_fields_survive_joint_transport():
    grid = geometry()
    state = moving_state(grid, constant=True)
    expected = state.n / state.h[..., None]
    moved, report = d.transport(state, grid, 10., fct=True)
    np.testing.assert_allclose(moved.n / moved.h[..., None], expected, rtol=0, atol=1e-12)
    assert report['cross_band_max_m_s'] > 0
    assert np.max(abs(moved.eta - state.eta)) > 0
    assert report['water_roundoff_ratio'] <= 1


def test_pressure_kick_uses_actual_midpoint_flux_and_fixed_mass():
    grid = geometry()
    state = moving_state(grid)
    out, report = d.pressure_kick(state, grid, 2.)
    np.testing.assert_array_equal(out.h, state.h)
    assert report['pressure_work_J'] != 0
    assert report['pressure_work_roundoff_ratio'] <= 1
    assert abs(report['kinetic_change_J'] - report['pressure_work_J']) <= report['pressure_work_bound_J']
    assert np.max(abs(out.n[..., 2:] - state.n[..., 2:])) > 0


def test_nonidentity_p0_momentum_remap_conserves_stock_and_does_not_gain_kinetic():
    grid = geometry()
    state = moving_state(grid)
    state.h[..., :3] = np.array([1., 8., 13.5])
    state.n[..., :3, 2] = state.h[..., :3] * d.RHO * np.array([.1, -.1, .02])
    state.n[..., :3, :2] = state.h[..., :3, None] * np.array([15., 35.])
    before = state.copy()
    out, report = d.remap(state, grid)
    np.testing.assert_allclose(out.n[..., :3, :].sum(-2), before.n[..., :3, :].sum(-2), rtol=1e-14, atol=1e-11)
    assert report['kinetic_change_J'] <= report['kinetic_bound_J']
    np.testing.assert_array_equal(out.n[..., 3:, :], before.n[..., 3:, :])


def test_nonzero_dynamic_subset_complete_step_and_deep_response():
    grid = geometry()
    state = moving_state(grid)
    params = d.Parameters(dt=30., nu_h=2e6, nu_v=1e-4, kappa_h=100., kappa_v=1e-6,
                          kappa_conv=.01, r_bot=.001, lambda_bulk=80., air_temperature=17.,
                          tau_x=.001, tau_y=.0002, coriolis=1e-4, fct=True)
    before = state.copy()
    out, accepted, report = d.advance(state, grid, params)
    assert accepted, report
    assert len(report['fast_steps']) == 12
    assert report['cross_band_max_m_s'] > 0
    assert sum(abs(x['pressure_work_J']) for x in report['fast_steps']) > 0
    assert np.max(abs(out.eta - state.eta)) > 0
    assert np.max(abs(out.n[..., 3:, :] - state.n[..., 3:, :])) > 0
    assert report['independent_water_roundoff_ratio'] <= 1
    assert max(report['independent_stock_roundoff_ratio']) <= 1
    assert not report['qualification_passed'] and not report['original_configuration_covered']
    assert len(report['committed_fast_steps']) == 12 and out.step == 1
    assert state.h.tobytes() == before.h.tobytes() and state.n.tobytes() == before.n.tobytes()


@pytest.mark.parametrize('unsupported', ['biharmonic', 'momentum_filter'])
def test_active_missing_process_rejects_before_dynamics(unsupported):
    grid = geometry()
    state = moving_state(grid)
    params = d.Parameters(kappa_bi=2e14) if unsupported == 'biharmonic' else d.Parameters(momentum_filter=True)
    out, accepted, report = d.advance(state, grid, params)
    assert not accepted and report['committed_fast_steps'] == []
    assert unsupported in report['rejection_reason']
    assert out.h.tobytes() == state.h.tobytes() and out.n.tobytes() == state.n.tobytes()


def test_cfl_failure_rolls_back_full_dynamic_state():
    grid = geometry()
    state = moving_state(grid)
    out, accepted, report = d.advance(state, grid, d.Parameters(dt=1e12))
    assert not accepted, report
    assert out.step == state.step and out.bottom == state.bottom
    assert out.h.tobytes() == state.h.tobytes() and out.n.tobytes() == state.n.tobytes()


def test_affine_physical_density_has_zero_common_depth_pressure_jump():
    grid = geometry()
    state = moving_state(grid)
    # Canonical bands are required by faces, so unequal eta already provides
    # distinct physical breakpoints. Compare a same-depth affine anomaly after
    # subtracting the analytically integrated surface caps.
    z = d.interfaces(state)
    center = .5 * (z[..., :-1] + z[..., 1:])
    temperature = 10. + .01 * center
    state.n[..., 0] = state.h * temperature
    state.n[..., 1] = state.h * 35.
    profile = d.density_reconstruction(state)
    for depth in (-1., -10., -21., -60., -300.):
        p0 = d.pressure_at(profile, (0, 0), depth)
        p1 = d.pressure_at(profile, (1, 0), depth)
        eta0, eta1 = state.eta[0, 0], state.eta[1, 0]
        expected = d.RHO * d.G * ((eta1 - eta0) - 1e-6 * (eta1 ** 2 - eta0 ** 2))
        np.testing.assert_allclose(p1 - p0, expected, rtol=0, atol=1e-9)


def test_pressure_work_independent_sum_uses_midpoint_not_initial_velocity():
    grid, state = geometry(), moving_state(geometry())
    out, receipt = d.pressure_kick(state, grid, 7.)
    face_list = d.faces(state, grid)
    before_u = state.n[..., 2:] / (d.RHO * state.h[..., None])
    after_u = out.n[..., 2:] / (d.RHO * out.h[..., None])
    actual_work = 0.
    initial_work = 0.
    for face in face_list:
        velocity_sum = (before_u[face.left][face.axis] + before_u[face.right][face.axis]
                        + after_u[face.left][face.axis] + after_u[face.right][face.axis])
        actual_work -= 7. * face.pressure_jump * face.conductance * velocity_sum / 4
        initial_work -= 7. * face.pressure_jump * face.conductance * (
            before_u[face.left][face.axis] + before_u[face.right][face.axis]) / 2
    assert abs(actual_work - initial_work) > receipt['pressure_work_bound_J']
    assert abs(receipt['kinetic_change_J'] - actual_work) <= receipt['pressure_work_bound_J']


def test_current_mass_mixing_changes_deep_stocks_and_conserves_all_four():
    grid, state = geometry(), moving_state(geometry())
    params = d.Parameters(nu_h=2e6, nu_v=1e-4, kappa_h=100., kappa_v=1e-5, kappa_conv=.01)
    out, _ = d.mixing(state, grid, params, 15.)
    before = np.sum(grid.area[..., None, None] * state.n, axis=(0, 1, 2))
    after = np.sum(grid.area[..., None, None] * out.n, axis=(0, 1, 2))
    np.testing.assert_allclose(after, before, rtol=1e-14, atol=1e-4)
    assert np.max(abs(out.n[..., 3:, :] - state.n[..., 3:, :])) > 0
    assert d.kinetic(out, grid) <= d.kinetic(state, grid)


def test_independent_impulse_and_heat_from_accepted_operations(monkeypatch):
    grid, state = geometry(), moving_state(geometry())
    params = d.Parameters(dt=30., tau_x=.001, tau_y=.0002, coriolis=1e-4,
                          r_bot=.001, lambda_bulk=80.)
    expected = np.zeros(4)
    original_source, original_pressure = d.sources, d.pressure_kick

    def source_oracle(input_state, g, p, duration):
        # Independent analytic source budget, before looking at the receipt.
        angle = p.coriolis * duration
        mu, mv = input_state.n[..., 2], input_state.n[..., 3]
        rotated_mu = np.cos(angle) * mu + np.sin(angle) * mv
        rotated_mv = -np.sin(angle) * mu + np.cos(angle) * mv
        impulse_u = (np.cos(angle) - 1) * mu + np.sin(angle) * mv
        impulse_v = -np.sin(angle) * mu + (np.cos(angle) - 1) * mv
        impulse_u[..., -1] += np.expm1(-p.r_bot * duration) * rotated_mu[..., -1]
        impulse_v[..., -1] += np.expm1(-p.r_bot * duration) * rotated_mv[..., -1]
        expected[2] += np.sum(g.area[..., None] * impulse_u) + g.area.sum() * duration * p.tau_x
        expected[3] += np.sum(g.area[..., None] * impulse_v) + g.area.sum() * duration * p.tau_y
        relaxation = -np.expm1(-p.lambda_bulk * duration / (d.RHO * d.CP * input_state.h[..., 0]))
        heat = input_state.h[..., 0] * (p.air_temperature - input_state.n[..., 0, 0] / input_state.h[..., 0]) * relaxation
        expected[0] += np.sum(g.area * heat)
        return original_source(input_state, g, p, duration)

    def pressure_oracle(input_state, g, duration):
        for face in d.faces(input_state, g):
            expected[face.axis + 2] -= duration * face.conductance * face.pressure_jump
        return original_pressure(input_state, g, duration)

    monkeypatch.setattr(d, 'sources', source_oracle)
    monkeypatch.setattr(d, 'pressure_kick', pressure_oracle)
    # Explicitly remove the discarded predictor's receipts from the oracle.
    original_fast = d._fast
    def predictor_oracle(input_state, g, p, duration):
        snapshot = expected.copy()
        output = original_fast(input_state, g, p, duration)
        if duration == p.dt:
            expected[:] = snapshot
        return output
    monkeypatch.setattr(d, '_fast', predictor_oracle)
    original_slow = d._slow
    slow_calls = 0
    def slow_oracle(input_state, g, p, duration):
        nonlocal slow_calls
        slow_calls += 1
        snapshot = expected.copy()
        output = original_slow(input_state, g, p, duration)
        if slow_calls == 2:
            expected[:] = snapshot
        return output
    monkeypatch.setattr(d, '_slow', slow_oracle)
    out, accepted, report = d.advance(state, grid, params)
    assert accepted, report
    observed = np.sum(grid.area[..., None, None] * (out.n - state.n), axis=(0, 1, 2))
    scale = np.sum(grid.area[..., None, None] * (abs(out.n) + abs(state.n)), axis=(0, 1, 2))
    assert np.all(abs(observed - expected) <= d.roundoff_bound(scale))


def test_forged_source_receipt_and_late_failure_roll_back(monkeypatch):
    grid, state = geometry(), moving_state(geometry())
    original = d.sources
    calls = 0
    def forged(*args):
        nonlocal calls
        calls += 1
        out, report = original(*args)
        if calls == 3:
            report['external_stock'][0] = 1e17
        return out, report
    monkeypatch.setattr(d, 'sources', forged)
    out, accepted, report = d.advance(state, grid, d.Parameters(tau_x=.001))
    assert not accepted and len(report['executed_fast_steps']) == 12
    assert report['committed_fast_steps'] == []
    assert 'source' in report['rejection_reason']
    assert out.h.tobytes() == state.h.tobytes() and out.n.tobytes() == state.n.tobytes()


@pytest.mark.parametrize('corruption', ['nan_parameter', 'string_parameter', 'masked_state',
                                      'negative_area', 'overflow_momentum'])
def test_invalid_inputs_never_accept_nan_diagnostics(corruption):
    import json
    grid, state = geometry(), moving_state(geometry())
    params = d.Parameters()
    if corruption == 'nan_parameter':
        params = d.Parameters(nu_h=np.nan)
    elif corruption == 'string_parameter':
        params = d.Parameters(nu_h='bad')
    elif corruption == 'masked_state':
        state.n = np.ma.array(state.n, mask=np.ones_like(state.n, dtype=bool))
    elif corruption == 'negative_area':
        grid.area *= -1
    else:
        state.n[..., 2] = 1e200 * (-1.) ** np.arange(8)[:, None, None]
        state.n[..., 3] = 0.
    out, accepted, report = d.advance(state, grid, params)
    assert not accepted and report['committed_fast_steps'] == []
    json.dumps(report, allow_nan=False)
    assert out.h.tobytes() == state.h.tobytes() and out.n.tobytes() == state.n.tobytes()


def test_fast_only_fixed_endpoint_order_is_measured():
    grid = d.Geometry(np.full((8, 4), 800.), 800., np.ones(4),
                      np.array([2.5, 7.5, 12.5, 42.5, 235., 200.]))
    initial = moving_state(grid)
    outputs = []
    for dt in (1., .5, .25, .125):
        current = initial.copy()
        for _ in range(round(2. / dt)):
            current, accepted, report = d.advance(current, grid, d.Parameters(dt=dt))
            assert accepted, report
        outputs.append(np.concatenate([current.h.ravel(),
                                       (current.n / np.array([1., 1., d.RHO, d.RHO])).ravel()]))
    differences = [np.linalg.norm(outputs[k] - outputs[k + 1]) for k in range(3)]
    observed = np.log2(np.array(differences[:-1]) / differences[1:])
    # This measured witness qualifies only this fixed endpoint, fast-only case.
    assert np.min(observed) >= 1.9, (differences, observed)


def test_mutually_matching_forged_stock_and_source_receipt_is_rejected(monkeypatch):
    grid, state = geometry(), moving_state(geometry())
    original = d.sources
    def forged(*args):
        out, report = original(*args)
        out.n[..., 0, 0] += .1
        report['external_stock'][0] += .1 * grid.area.sum()
        return out, report
    monkeypatch.setattr(d, 'sources', forged)
    out, accepted, report = d.advance(state, grid, d.Parameters())
    assert not accepted, report
    assert 'source' in report['rejection_reason']
    assert report['committed_fast_steps'] == []
    assert out.h.tobytes() == state.h.tobytes() and out.n.tobytes() == state.n.tobytes()

def test_two_gauss_pressure_segment_average_for_different_affine_densities():
    grid, state = geometry(), moving_state(geometry())
    z = d.interfaces(state)
    center = .5 * (z[..., :-1] + z[..., 1:])
    phase = 2 * np.pi * np.arange(8)[:, None] / 8
    intercept = np.broadcast_to(.05 * np.cos(phase), (8, 4))
    slope = np.broadcast_to(.01 + .002 * np.sin(phase), (8, 4))
    state.n[..., 0] = state.h * (10 + intercept[..., None] + slope[..., None] * center)
    state.n[..., 1] = state.h * 35.
    witness_curvature = 0.
    for face in d.faces(state, grid):
        left, right = face.left, face.right
        lo = max(z[left[:2]][left[-1] + 1], z[right[:2]][right[-1] + 1])
        hi = min(z[left[:2]][left[-1]], z[right[:2]][right[-1]])
        midpoint = .5 * (lo + hi)
        depth_square = (lo * lo + lo * hi + hi * hi) / 3
        def average_pressure(column, square):
            eta = state.eta[column]
            anomaly_intercept = -d.RHO * 2e-4 * intercept[column]
            anomaly_slope = -d.RHO * 2e-4 * slope[column]
            return d.G * ((d.RHO + anomaly_intercept) * (eta - midpoint)
                          + .5 * anomaly_slope * (eta * eta - square))
        expected = average_pressure(right[:2], depth_square) - average_pressure(left[:2], depth_square)
        midpoint_jump = average_pressure(right[:2], midpoint * midpoint) - average_pressure(left[:2], midpoint * midpoint)
        witness_curvature = max(witness_curvature, abs(expected - midpoint_jump))
        np.testing.assert_allclose(face.pressure_jump, expected, rtol=0, atol=2e-9)
    assert witness_curvature > .01


def test_p0_momentum_remap_matches_exact_overlap_not_capacity_substitution():
    grid, state = geometry(), moving_state(geometry())
    state.h[..., :3] = np.array([1., 8., 13.5])
    state.n[..., :3, 0] = state.h[..., :3] * np.array([10., 20., 30.])
    state.n[..., :3, 1] = state.h[..., :3] * 35.
    state.n[..., :3, 2] = d.RHO * state.h[..., :3] * np.array([.1, -.1, .02])
    out, _ = d.remap(state, grid)
    np.testing.assert_allclose(out.n[..., :3, 2] / d.RHO,
                               np.broadcast_to([-.05, -.63, .25], (8, 4, 3)), rtol=0, atol=2e-14)
    np.testing.assert_allclose(out.n[..., :3, 0],
                               np.broadcast_to([40., 160., 375.], (8, 4, 3)), rtol=0, atol=2e-11)


@pytest.mark.parametrize('bad_receipt', [np.nan, np.array([np.nan]), None, 'forged'])
def test_second_pressure_receipt_cannot_hide_nonfinite_ratio(monkeypatch, bad_receipt):
    grid, state = geometry(), moving_state(geometry())
    original = d.pressure_kick
    calls = 0
    def corrupt(*args):
        nonlocal calls
        calls += 1
        out, report = original(*args)
        if calls == 4:
            report['pressure_work_roundoff_ratio'] = bad_receipt
        return out, report
    monkeypatch.setattr(d, 'pressure_kick', corrupt)
    out, accepted, report = d.advance(state, grid, d.Parameters())
    assert not accepted, report
    assert report['committed_fast_steps'] == []
    assert out.h.tobytes() == state.h.tobytes() and out.n.tobytes() == state.n.tobytes()


def test_convective_capacity_is_planned_for_new_inversions():
    grid = d.Geometry(np.ones((8, 4)), 1., np.ones(4),
                      np.array([2.5, 7.5, 12.5, 42.5, 235., 200.]))
    rng = np.random.default_rng(5)
    for _ in range(3):
        h = grid.thickness(rng.uniform(-20., 20., (8, 4)))
        temperature = np.sort(rng.uniform(0., 40., (8, 4, 6)), axis=-1)[..., ::-1]
    state = d.State(h, np.zeros((8, 4, 6, 4)), grid.bottom)
    state.n[..., 0] = h * temperature
    state.n[..., 1] = h * 35.
    out, report = d.mixing(state, grid, d.Parameters(kappa_h=1., kappa_conv=100.), .175)
    assert report['diffusion_subcycles'] > 2
    np.testing.assert_allclose(d.totals(out, grid)[1], d.totals(state, grid)[1], rtol=1e-14)


@pytest.mark.parametrize('bad_value', [True, np.bool_(True), np.nan, np.inf])
def test_numeric_parameter_cannot_be_boolean_or_nonfinite(bad_value):
    grid, state = geometry(), moving_state(geometry())
    out, accepted, report = d.advance(state, grid, d.Parameters(nu_h=bad_value))
    assert not accepted and not report['executed_fast_steps']
    assert out.h.tobytes() == state.h.tobytes() and out.n.tobytes() == state.n.tobytes()


def test_malformed_none_stock_rolls_back_as_structured_refusal():
    grid, state = geometry(), moving_state(geometry())
    state.n = None
    out, accepted, report = d.advance(state, grid, d.Parameters())
    assert not accepted and out.n is None and report['committed_fast_steps'] == []


@pytest.mark.parametrize('receipt_value', [np.array(0.), np.array([0., 0.]), np.array([[0., 0., 0., 0.]])])
def test_bad_stock_receipt_shape_is_a_structured_rollback(monkeypatch, receipt_value):
    grid, state = geometry(), moving_state(geometry())
    original = d.sources
    def corrupt(*args):
        out, report = original(*args)
        report['external_stock'] = receipt_value
        return out, report
    monkeypatch.setattr(d, 'sources', corrupt)
    out, accepted, report = d.advance(state, grid, d.Parameters())
    assert not accepted and report['committed_fast_steps'] == []
    assert out.h.tobytes() == state.h.tobytes() and out.n.tobytes() == state.n.tobytes()


def test_nonparameter_object_is_structured_refusal():
    grid, state = geometry(), moving_state(geometry())
    out, accepted, report = d.advance(state, grid, None)
    assert not accepted and report['committed_fast_steps'] == []
    assert out.h.tobytes() == state.h.tobytes() and out.n.tobytes() == state.n.tobytes()


@pytest.mark.parametrize('bad_bottom', [np.array([-500.]), np.array(-500.), np.nan])
def test_bottom_authority_must_be_finite_scalar(bad_bottom):
    grid, state = geometry(), moving_state(geometry())
    state.bottom = bad_bottom
    out, accepted, report = d.advance(state, grid, d.Parameters())
    assert not accepted and report['committed_fast_steps'] == []
    assert out.h.tobytes() == state.h.tobytes() and out.n.tobytes() == state.n.tobytes()


def test_arbitrary_precision_parameter_does_not_escape_rollback():
    grid, state = geometry(), moving_state(geometry())
    out, accepted, report = d.advance(state, grid, d.Parameters(nu_h=10**100))
    assert not accepted and report['committed_fast_steps'] == []
    assert out.h.tobytes() == state.h.tobytes() and out.n.tobytes() == state.n.tobytes()


@pytest.mark.parametrize('field', ['external_stock', 'pressure_work_J'])
def test_arbitrary_precision_receipt_cannot_escape_full_rollback(monkeypatch, field):
    grid, state = geometry(), moving_state(geometry())
    original = d.pressure_kick
    def corrupt(*args):
        out, report = original(*args)
        report[field] = [10**1000, 0., 0., 0.] if field == 'external_stock' else 10**1000
        return out, report
    monkeypatch.setattr(d, 'pressure_kick', corrupt)
    out, accepted, report = d.advance(state, grid, d.Parameters())
    assert not accepted and report['committed_fast_steps'] == []
    assert out.h.tobytes() == state.h.tobytes() and out.n.tobytes() == state.n.tobytes()


@pytest.mark.parametrize('forged_call', [1, 3])
def test_finite_wrong_bulk_heat_receipt_rejects_full_macro_state(monkeypatch, forged_call):
    grid, state = geometry(), moving_state(geometry())
    original = d.sources
    original_state = state.copy()
    calls = 0
    def forge_heat_only(*args):
        nonlocal calls
        calls += 1
        out, receipt = original(*args)
        if calls == forged_call:
            receipt['bulk_heat_J'] *= 2.
        return out, receipt
    monkeypatch.setattr(d, 'sources', forge_heat_only)
    out, accepted, report = d.advance(state, grid, d.Parameters(dt=1., lambda_bulk=80., air_temperature=17.))
    assert not accepted, report
    assert 'heat' in report['rejection_reason']
    if forged_call == 3:
        assert len(report['executed_fast_steps']) == 12
    assert report['committed_fast_steps'] == []
    assert out.step == state.step and out.bottom == state.bottom
    assert out.h.tobytes() == original_state.h.tobytes()
    assert out.n.tobytes() == original_state.n.tobytes()
    assert state.h.tobytes() == original_state.h.tobytes()
    assert state.n.tobytes() == original_state.n.tobytes()


def test_nonzero_heat_receipt_matches_independent_surface_law():
    grid, state = geometry(), moving_state(geometry())
    parameters = d.Parameters(dt=1., lambda_bulk=80., air_temperature=17.)
    duration = parameters.dt / 2
    surface_temperature = state.n[..., 0, 0] / state.h[..., 0]
    relaxation = -np.expm1(-parameters.lambda_bulk * duration / (d.RHO * d.CP * state.h[..., 0]))
    expected_heat_J = np.sum(grid.area * state.h[..., 0]
                             * (parameters.air_temperature - surface_temperature) * relaxation) * d.RHO * d.CP
    out, receipt = d._slow(state, grid, parameters, duration)
    assert receipt['bulk_heat_J'] != 0
    assert abs(receipt['bulk_heat_J'] - expected_heat_J) <= d.roundoff_bound(abs(expected_heat_J))
    observed_heat_J = np.sum(grid.area[..., None] * (out.n[..., 0] - state.n[..., 0])) * d.RHO * d.CP
    stock_scale = np.sum(grid.area[..., None] * (abs(out.n[..., 0]) + abs(state.n[..., 0]))) * d.RHO * d.CP
    assert abs(observed_heat_J - expected_heat_J) <= d.roundoff_bound(stock_scale)
