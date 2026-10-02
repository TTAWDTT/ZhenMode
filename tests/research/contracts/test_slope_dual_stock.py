"""Independent weighted physical projection tests; zero accepted model steps."""
from dataclasses import replace

import numpy as np
import pytest

from research.experiments.material_top_band import affine_physical_oracle as physical
from research.experiments.material_top_band import slope_dual_stock_oracle as oracle
from research.experiments.material_top_band.affine_physical_cases import manufactured_case
from research.experiments.material_top_band.inventory_pressure import EOS
from research.experiments.material_top_band.slope_dual_stock import FrozenPhysicalProjection


def fixture(**parameters):
    parameters.setdefault('eta', (-.2, .3))
    profile, spec = manufactured_case(**parameters)
    return FrozenPhysicalProjection(profile, distance_m=2., length_m=1.5), spec


def identity(value, reference, scale):
    assert abs(value - reference) <= 512. * np.finfo(float).eps * max(1., scale)


def array_identity(value, reference, scale):
    assert np.all(abs(value - reference) <= 512. * np.finfo(float).eps * np.maximum(1., scale))


@pytest.mark.parametrize('eta', [(-.2, .3), (-.2, -.2)])
def test_each_basis_stock_moment_and_mass_against_independent_volume(eta):
    projection, spec = fixture(eta=eta)
    before = projection.profile.state.stocks.copy()
    mapped = projection.evaluate()
    ref = oracle.mapped(spec, projection.chart, projection.profile.state)
    for index in np.ndindex(mapped.values.shape):
        identity(mapped.values[index], ref['values'][index], mapped.scale[index] + ref['scale'][index])
    array_identity(mapped.mass, ref['mass'], abs(mapped.mass) + abs(ref['mass']))
    identity(mapped.PE, ref['PE'], mapped.PE_scale + ref['PE_scale'])
    reference_H = 1025. * mapped.values[:, :, 0].ravel()
    array_identity(mapped.mass @ np.ones(28), reference_H, abs(mapped.mass) @ np.ones(28) + abs(reference_H))
    for slot in range(2):
        reference_M = mapped.values[:, :, 3 + slot].ravel()
        array_identity(mapped.mass @ mapped.velocity[:, slot], reference_M, abs(mapped.mass) @ abs(mapped.velocity[:, slot]) + abs(reference_M))
    assert np.linalg.eigvalsh(mapped.mass).min() > 0.
    assert not mapped.raw_conservative_remap
    assert not mapped.production_qualified
    np.testing.assert_array_equal(projection.profile.state.stocks, before)


def test_original_inventory_and_two_PE_defects_are_retained():
    projection, spec = fixture()
    mapped = projection.evaluate()
    ref = oracle.raw_inventory(projection.profile.state)
    array_identity(mapped.raw_inventory, ref, abs(mapped.raw_inventory) + abs(ref))
    defect = mapped.raw_inventory - mapped.values[:, :, :5].sum(axis=(0, 1))
    array_identity(defect, np.array([0., 0., 50. / 779., 4.1, 0.]), abs(mapped.raw_inventory) + mapped.scale[:, :, :5].sum(axis=(0, 1)))
    identity(mapped.raw_PE - mapped.PE, 629.17048125, mapped.PE_scale + abs(mapped.raw_PE))
    identity(mapped.raw_free_PE - mapped.free_PE, 628.453125, abs(mapped.raw_free_PE) + abs(mapped.free_PE))
    assert abs((mapped.raw_PE - mapped.raw_free_PE) - (mapped.PE - mapped.free_PE) - .71735625) < 1e-10
    assert abs(mapped.raw_PE - mapped.PE) > 1.
    identity(mapped.PE, 94.08599325, mapped.PE_scale)


@pytest.mark.parametrize('eta', [(-.2, .3), (-.2, -.2)])
@pytest.mark.parametrize('sign', [1., -1.])
def test_full_Bdot_weighted_local_balances_and_raw_direction_defects(eta, sign):
    U, strain = .03 * sign, .008 * sign
    projection, spec = fixture(eta=eta, u0=U, strain=strain)
    mapped, direction = projection.evaluate(), projection.direction(U, strain)
    ref = oracle.direction(spec, projection.chart, projection.profile.state, U, strain)
    for index in np.ndindex(direction.values.shape):
        identity(direction.values[index], ref['values'][index], direction.scale[index] + ref['scale'][index])
    energy = physical.energy_direction(spec, U, strain)
    identity(direction.PE_dot, energy['PE_dot'], direction.PE_scale + energy['scale'])
    for slot in range(2):
        chained = direction.mass_dot @ mapped.velocity[:, slot] + mapped.mass @ direction.velocity_dot[:, slot]
        array_identity(chained, direction.values[:, :, 3 + slot].ravel(), abs(direction.mass_dot) @ abs(mapped.velocity[:, slot]) + abs(mapped.mass) @ abs(direction.velocity_dot[:, slot]) + direction.scale[:, :, 3 + slot].ravel())
    if eta[0] != eta[1]:
        defects = mapped.raw_inventory - mapped.values[:, :, :5].sum(axis=(0, 1))
        rate_defects = direction.raw_inventory_dot - direction.values[:, :, :5].sum(axis=(0, 1))
        expected = oracle.analytic_defects(spec, U, strain)
        array_identity(defects, expected, abs(mapped.raw_inventory) + mapped.scale[:, :, :5].sum(axis=(0, 1)))
        array_identity(rate_defects, -3. * strain * expected, abs(direction.raw_inventory_dot) + direction.scale[:, :, :5].sum(axis=(0, 1)))
        assert abs(direction.raw_PE_dot - direction.PE_dot) > 1.
    weighted = oracle.weighted_balances(spec, projection.chart, projection.profile.state, U, strain)
    for candidate, independent in zip(direction.rows, weighted, strict=True):
        for slot in range(6):
            scale = candidate['scale'][slot] + independent['scale'][slot]
            identity(candidate['value'][slot], independent['content_dot'][slot], scale)
            identity(candidate['value'][slot] + independent['flux'][slot], independent['source'][slot], scale)
    # Water GCL needs the weighted basis term as well as the moving top.
    assert max(abs(row['source'][0]) for row in weighted) > .001
    assert max(abs(row['shape'][0]) for row in weighted) > .001


def test_omit_shape_basis_term_gravity_source_and_active_virtual_flux_fail():
    projection, spec = fixture()
    direction = projection.direction(.03, .008)
    weighted = oracle.weighted_balances(spec, projection.chart, projection.profile.state, .03, .008)
    failures = dict(shape=False, basis=False, gravity=False, internal=False)
    for candidate, independent in zip(direction.rows, weighted, strict=True):
        tolerance = 512. * np.finfo(float).eps * max(1., np.sum(independent['scale'] + candidate['scale']))
        failures['shape'] |= np.max(abs(candidate['volume'] - independent['content_dot'])) > 1000. * tolerance
        failures['basis'] |= abs(candidate['value'][0] + independent['flux'][0]) > 1000. * tolerance
        failures['gravity'] |= abs(candidate['value'][5] + independent['flux'][5] - independent['basis_source'][5]) > 1000. * tolerance
        if not candidate['strip'].actual_top:
            without_cut = candidate['value'] + independent['flux'] - independent['top_flux'] - independent['source']
            failures['internal'] |= np.max(abs(without_cut)) > 1000. * tolerance
    assert all(failures.values())


@pytest.mark.parametrize('sign', [1., -1.])
def test_affine_raw_stock_parameter_curve_freezes_chart_and_PE_remainder(sign):
    U, strain = .03 * sign, .008 * sign
    projection, spec = fixture(u0=U, strain=strain)
    initial, direction = projection.evaluate(), projection.direction(U, strain)
    for epsilon in (2.**-8, 2.**-9, 2.**-10):
        state, changed = oracle.curve(spec, projection.profile.state, U, strain, epsilon)
        after = projection.evaluate(replace(projection.profile, state=state))
        envelope = oracle.forward_envelope(spec, projection.chart, projection.profile.state, U, strain, epsilon)
        observed = (after.values - initial.values) / epsilon
        assert np.all(abs(observed - direction.values) <= envelope['values_bound'])
        assert abs((after.PE - initial.PE) / epsilon - direction.PE_dot) <= envelope['PE_bound']
        # Regenerating normalized cuts changes weighted basis masses, even
        # though total physical volume remains independent of partition.
        moving = FrozenPhysicalProjection(replace(projection.profile, state=state), distance_m=2., length_m=1.5).evaluate()
        mismatch = abs((moving.values[:, :, 0] - initial.values[:, :, 0]) / epsilon - direction.values[:, :, 0])
        assert np.max(mismatch[:, 1:]) > 1e-5
        assert changed.eta != spec.eta


def shear_profile(projection, *, zero_depth=False):
    state = projection.profile.state
    if zero_depth:
        velocity = np.zeros((2, 14, 2))
        velocity[0, 0, 0] = 1.
        velocity[0, -1, 0] = -state.h[0, 0] / state.h[0, -1]
    else:
        n = np.arange(28).reshape(2, 14)
        velocity = np.stack([.03 + .002 * n + .01 * np.cos(n), -.02 + .001 * np.sin(n)], axis=-1)
    stocks = state.stocks.copy()
    stocks[:, :, 2:] = 1025. * state.h[:, :, None] * velocity
    return replace(projection.profile, state=replace(state, stocks=stocks))


@pytest.mark.parametrize('zero_depth', [False, True])
def test_actual_shear_projection_fixed_mass_work_and_pressure_independence(zero_depth):
    original, spec = fixture()
    projection = FrozenPhysicalProjection(shear_profile(original, zero_depth=zero_depth), distance_m=2., length_m=1.5)
    mapped = projection.evaluate()
    np.testing.assert_array_equal(projection.base.force_N, original.base.force_N)
    ref = oracle.static_momentum_defect(spec, projection.chart, projection.profile.state)
    array_identity(mapped.raw_inventory[3:5] - mapped.values[:, :, 3:5].sum(axis=(0, 1)), ref, abs(mapped.raw_inventory[3:5]) + mapped.scale[:, :, 3:5].sum(axis=(0, 1)))
    K = oracle.kinetic(spec, projection.chart, mapped.velocity)
    identity(.5 * sum(mapped.velocity[:, j] @ mapped.mass @ mapped.velocity[:, j] for j in range(2)), K['value'], K['scale'])
    kick = projection.fixed_mass_impulse(.01)
    before = mapped.velocity.copy()
    after = before.copy()
    after[:, 0] = kick['velocity_after']
    independent_K = oracle.kinetic(spec, projection.chart, after)
    midpoint = .5 * (before[:, 0] + after[:, 0])
    power = physical.pressure_power(spec, projection.chart, midpoint.reshape(2, 14))
    identity(kick['kinetic_change'], independent_K['value'] - K['value'], kick['scale'] + K['scale'] + independent_K['scale'])
    identity(kick['kinetic_change'], .01 * power['value'], kick['scale'] + .01 * power['scale'])
    array_identity(mapped.mass @ after[:, 0], kick['momentum_after'], abs(mapped.mass) @ abs(after[:, 0]) + abs(kick['momentum_after']))
    assert not kick['accepted_state_returned']
    with pytest.raises(ValueError, match='momentum'):
        projection.direction(.03, .008)
    if zero_depth:
        assert abs(np.sum(projection.profile.state.h * before[:, 0].reshape(2, 14))) < 1e-14
        assert abs(physical.pressure_power(spec, projection.chart, before[:, 0].reshape(2, 14))['value']) > .01


def test_static_basis_covariance_and_partition_unity_limit():
    projection, _ = fixture()
    mapped = projection.evaluate()
    transform = np.eye(28)
    transform[np.ix_([0, 14], [0, 14])] = [[1.2, -.2], [-.1, 1.1]]
    ones = np.ones(28)
    np.testing.assert_allclose(transform @ ones, ones, atol=1e-15, rtol=0)
    changed_u = np.linalg.solve(transform, mapped.velocity[:, 0])
    changed_W = transform.T @ mapped.mass @ transform
    changed_F = transform.T @ projection.base.force_N.ravel()
    original_u, original_F = mapped.velocity[:, 0], projection.base.force_N.ravel()
    energy_scale = abs(changed_u) @ abs(changed_W) @ abs(changed_u) + abs(original_u) @ abs(mapped.mass) @ abs(original_u)
    work_scale = np.sum(abs(changed_F * changed_u)) + np.sum(abs(original_F * original_u))
    identity(changed_u @ changed_W @ changed_u, original_u @ mapped.mass @ original_u, energy_scale)
    identity(changed_F @ changed_u, original_F @ original_u, work_scale)
    for slot in range(6):
        value = mapped.values[:, :, slot].ravel()
        identity(np.sum(transform.T @ value), np.sum(value), np.sum(abs(transform.T) * abs(value)[None, :]) + np.sum(abs(value)))
    bad = np.eye(28)
    bad[0, 0] = 2.
    assert abs(np.sum(bad.T @ mapped.values[:, :, 0].ravel()) - np.sum(mapped.values[:, :, 0])) > .001


def test_zero_force_and_invalid_chart_T_or_geometry_refuse():
    projection, _ = fixture(eta=(-.2, -.2), density_gradient_x=0., external_pressure=(80., 80.))
    assert np.max(abs(projection.base.force_N)) == 0.
    profile, _ = manufactured_case(nonlinear_TS=True)
    with pytest.raises(ValueError, match='Tref'):
        FrozenPhysicalProjection(profile, distance_m=2., length_m=1.5)
    projection, spec = fixture()
    state, _ = oracle.curve(spec, projection.profile.state, .03, .008, .84)
    assert np.all(state.h > 0.)
    with pytest.raises(ValueError, match='cross'):
        projection.evaluate(replace(projection.profile, state=state))
    z = projection.profile.state.interfaces.copy()
    z[0, 5] += 1e-5
    with pytest.raises(ValueError, match='interior'):
        projection.evaluate(replace(projection.profile, state=replace(projection.profile.state, interfaces=z)))
    with pytest.raises(ValueError):
        projection.direction(np.nan, .008)
    for duration in (0., -1., np.nan):
        with pytest.raises(ValueError):
            projection.fixed_mass_impulse(duration)


def test_altered_EOS_is_not_silently_admitted_to_historical_contract():
    profile, _ = manufactured_case()
    with pytest.raises(ValueError, match='EOS'):
        FrozenPhysicalProjection(replace(profile, eos=replace(EOS(), rho0=1030.)), distance_m=2., length_m=1.5)


def test_false_deep_projected_rates_and_identity_map_fail_independent_checks():
    projection, spec = fixture()
    direction = projection.direction(.03, .008)
    ref = oracle.direction(spec, projection.chart, projection.profile.state, .03, .008)
    fake = direction.values.copy()
    fake[:, 3:, 2:4] = 0.
    tolerance = 512. * np.finfo(float).eps * np.maximum(1., direction.scale + ref['scale'])
    assert np.max(abs(fake - ref['values']) / tolerance) > 1000.
    raw_hdot = np.zeros((2, 14))
    raw_hdot[:, 0] = [physical.surface_direction(spec, x, .03, .008) for x in (0., spec.distance)]
    bad_volume_dot = 1.5 * raw_hdot
    assert np.max(abs(bad_volume_dot - direction.values[:, :, 0])) > 1e-4


@pytest.mark.parametrize('fault', ['owner', 'missing'])
def test_independent_raw_owner_and_coverage_refuse_a_forged_chart(fault):
    projection, spec = fixture()
    chart = list(projection.chart)
    if fault == 'owner':
        chart[0] = replace(chart[0], left_layer=(chart[0].left_layer + 1) % 14)
    else:
        chart = chart[1:]
    with pytest.raises(ValueError, match='ownership|coverage'):
        oracle.mapped(spec, chart, projection.profile.state)


def test_mixed_sign_polynomial_refuses_FD_certificate_without_skipping_case():
    projection, spec = fixture(density_intercept=0.)
    assert np.isfinite(projection.evaluate().PE)
    with pytest.raises(ValueError, match='sign'):
        oracle.forward_envelope(spec, projection.chart, projection.profile.state, .03, .008, 2.**-8)


def test_scalar_witness_refuses_uncommitted_sources(monkeypatch):
    from research.experiments.material_top_band import slope_dual_stock_evidence as evidence
    monkeypatch.setattr(evidence.subprocess, 'check_output', lambda *args, **kwargs: '?? pending.py')
    with pytest.raises(ValueError, match='committed clean'):
        evidence.provenance()


@pytest.mark.parametrize('top_only', [False, True])
def test_FD_certificate_refuses_actual_shear_that_disagrees_with_material_curve(top_only):
    projection, spec = fixture()
    profile = shear_profile(projection)
    if top_only:
        stocks = projection.profile.state.stocks.copy()
        stocks[:, 0, 2] = 1025. * projection.profile.state.h[:, 0]
        profile = replace(projection.profile, state=replace(projection.profile.state, stocks=stocks))
    with pytest.raises(ValueError, match='momentum'):
        oracle.forward_envelope(spec, projection.chart, profile.state, .03, .008, 2.**-8)


def test_independent_projection_binds_actual_thermodynamic_stock_to_specification():
    projection, spec = fixture()
    with pytest.raises(ValueError, match='density'):
        oracle.mapped(replace(spec, density_gradient_x=spec.density_gradient_x + .01), projection.chart, projection.profile.state)


def test_equal_overlap_and_gap_cannot_hide_in_closed_raw_width_sums():
    projection, spec = fixture()
    state, chart = projection.profile.state, list(projection.chart)
    changed = False
    for side in range(2):
        for index, strip in enumerate(chart):
            layer = (strip.left_layer, strip.right_layer)[side]
            lower, upper = (strip.lower_left, strip.upper_left) if side == 0 else (strip.lower_right, strip.upper_right)
            margin = min(lower - state.interfaces[side, layer + 1], state.interfaces[side, layer] - upper)
            if margin > 1e-5:
                shift = .1 * margin
                fields = dict(lower_left=lower + shift, upper_left=upper + shift) if side == 0 else dict(lower_right=lower + shift, upper_right=upper + shift)
                chart[index] = replace(strip, **fields)
                changed = True
                break
        if changed:
            break
    assert changed
    assert physical.validate_strip_ownership(spec, state, chart)
    with pytest.raises(ValueError, match='partition|overlap or gap'):
        oracle.mapped(spec, chart, state)


def test_independent_binding_refuses_division_overflow_before_comparisons():
    projection, spec = fixture()
    h = projection.profile.state.h.copy()
    h[0, 0] = 1e-320
    with pytest.raises(ValueError, match='finite derived'):
        oracle.mapped(spec, projection.chart, replace(projection.profile.state, h=h))


def test_crossed_left_right_pairing_refuses_despite_closed_endpoint_partitions():
    projection, spec = fixture()
    chart = list(projection.chart)
    left, right = chart[:2]
    assert not left.actual_top and not right.actual_top
    chart[0] = replace(left, lower_right=right.lower_right, upper_right=right.upper_right, right_layer=right.right_layer)
    chart[1] = replace(right, lower_right=left.lower_right, upper_right=left.upper_right, right_layer=left.right_layer)
    assert physical.validate_strip_ownership(spec, projection.profile.state, chart)
    with pytest.raises(ValueError, match='partition'):
        oracle.mapped(spec, chart, projection.profile.state)
