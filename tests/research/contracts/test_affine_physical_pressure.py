"""Independent physical-volume gates, not accepted steps or real integration."""
from dataclasses import replace

import numpy as np
import pytest

from research.experiments.material_top_band import affine_physical_oracle as oracle
from research.experiments.material_top_band.affine_physical import (
    AffinePhysicalDual,
    UnsupportedAffineContract,
)
from research.experiments.material_top_band.affine_physical_cases import manufactured_case


def operator(**kwargs):
    profile, specification = manufactured_case(**kwargs)
    op = AffinePhysicalDual(profile, distance_m=2., length_m=1.5)
    oracle.validate_strip_ownership(specification, op.state, op.segments)
    return op, specification


def identity(candidate, reference, scale):
    assert abs(candidate - reference) <= 512. * np.finfo(float).eps * max(1., scale)


def test_identical_affine_rho_different_fourteen_partitions_zero_spurious_force():
    op, spec = operator(density_gradient_x=0., external_pressure=(80., 80.))
    assert op.state.h.shape == (2, 14)
    assert not np.array_equal(op.state.interfaces[0], op.state.interfaces[1])
    for side in range(2):
        for layer in range(14):
            ref = oracle.basis_force(spec, op.segments, side, layer)
            identity(op.force_N[side, layer], ref['value'], op.force_scale_N[side, layer] + ref['scale'])
            assert abs(ref['value']) < 1e-10


@pytest.mark.parametrize('eta', [(-.2, -.2), (-.2, .3)])
def test_four_contours_and_every_actual_basis_independent_geometry(eta):
    op, spec = operator(eta=eta)
    for row in op.contours():
        ref = oracle.segment_contour(spec, row['segment'])
        identity(row['force_N'], ref['value'], row['scale'] + ref['scale'])
        assert set(row['edges']) == {'left', 'right', 'top', 'bottom'}
        identity(sum(row['basis']), row['force_N'], row['scale'] + sum(abs(x) for x in row['basis']))
    for side in range(2):
        for layer in range(14):
            ref = oracle.basis_force(spec, op.segments, side, layer)
            identity(op.force_N[side, layer], ref['value'], op.force_scale_N[side, layer] + ref['scale'])
            for epsilon in (2.**-8, 2.**-9, 2.**-10):
                varied = oracle.geometric_load_variation(spec, op.segments, side, layer, epsilon)
                assert abs(varied['value'] - op.force_N[side, layer]) <= varied['bound'] + 512. * np.finfo(float).eps * max(1., op.force_scale_N[side, layer])
    assert np.max(abs(op.force_N)) > 1.


def test_omitted_active_internal_top_and_bottom_are_real_failures():
    op, spec = operator(eta=(-.2, .3))
    active = set()
    for row in op.contours():
        ref = oracle.segment_contour(spec, row['segment'])
        bound = 512. * np.finfo(float).eps * max(1., row['scale'] + ref['scale'])
        for edge in ('top', 'bottom'):
            if abs(row['edges'][edge]) > 1.:
                assert abs(row['force_N'] - row['edges'][edge] - ref['value']) > 1000. * bound
                active.add(edge)
    assert active == {'top', 'bottom'}


def test_state_bound_pressure_zero_reverse_shear_superposition_and_zero_depth_flux():
    op, spec = operator()
    force = op.force_N.copy()
    first = np.arange(28).reshape(2, 14) * .002 - .02
    second = np.cos(np.arange(28)).reshape(2, 14) * .01
    for velocity in (np.zeros((2, 14)), first, -first, second, first + second):
        ref = oracle.pressure_power(spec, op.segments, velocity)
        identity(op.pressure_power(velocity), ref['value'], np.sum(abs(force * velocity)) + ref['scale'])
        np.testing.assert_array_equal(op.force_N, force)
    assert op.pressure_power(first + second) == pytest.approx(op.pressure_power(first) + op.pressure_power(second), abs=1e-11)
    shear = np.zeros((2, 14))
    shear[0, 0] = 1.
    shear[0, -1] = -op.state.h[0, 0] / op.state.h[0, -1]
    assert abs(np.sum(op.state.h * shear)) < 1e-14
    assert abs(op.pressure_power(shear)) > .01


@pytest.mark.parametrize('eta', [(-.2, -.2), (-.2, .3)])
@pytest.mark.parametrize('sign', [1., -1.])
def test_actual_layer_stock_directions_and_virtual_ALE_fluxes(eta, sign):
    op, spec = operator(eta=eta, u0=.03 * sign, strain=.008 * sign)
    rate = op.affine_rates(.03 * sign, .008 * sign)
    for side in range(2):
        for layer in range(14):
            ref = oracle.layer_stock_direction(spec, op.state, side, layer, .03 * sign, .008 * sign)
            np.testing.assert_allclose(rate['stockdot'][side, layer], ref, atol=1e-11, rtol=0)
    assert np.all(rate['hdot'][:, 1:] == 0.)
    assert np.max(abs(rate['relative_downward'][:, 3])) > .005
    expected_R = .008 * sign * (op.state.interfaces - spec.bottom)
    expected_R[:, [0, -1]] = 0.
    np.testing.assert_allclose(rate['relative_downward'], expected_R, atol=1e-14, rtol=0)
    assert np.max(abs(rate['stockdot'][:, 3:, 1:3])) > .005
    flux = oracle.ale_fluxes(spec, op.segments, .03 * sign, .008 * sign)
    for slot, value in enumerate(op.physical_stock_direction(rate)):
        identity(value, -flux['outer'][slot], abs(value) + flux['scale'][slot])
    assert np.max(abs(flux['internal'])) > .001
    assert np.max(abs(flux['internal_pair_residual'])) < 1e-11
    assert flux['omitted_local_residual'] > .001
    for row in flux['local']:
        for slot in range(4):
            identity(row['content_dot'][slot], -row['flux'][slot], row['scale'][slot])
    assert flux['omitted_local_residual'] > 1000. * flux['omitted_local_bound']


@pytest.mark.parametrize('sign', [1., -1.])
def test_nonzero_flat_surface_inventory_PE_physical_energy_pair(sign):
    op, spec = operator(u0=.03 * sign, strain=.008 * sign)
    rate = op.require_endpoint_pairing(.03 * sign, .008 * sign)
    ref = oracle.energy_direction(spec, .03 * sign, .008 * sign)
    identity(rate['physical_PE_dot_W'], ref['PE_dot'], rate['PE_scale'] + ref['scale'])
    identity(rate['node_PE_dot_W'], ref['PE_dot'], rate['node_PE_scale'] + ref['scale'])
    identity(rate['pressure_power_W'] + ref['PE_dot'], ref['boundary_power'], rate['pressure_power_scale'] + ref['scale'])
    assert abs(rate['pressure_power_W']) > 1.
    assert rate['endpoint_pairing_passed']
    assert not rate['production_force_consumption_qualified']


def test_unequal_eta_analytical_B_Bdot_gap_and_endpoint_refusal():
    op, spec = operator(eta=(-.2, .3))
    rate = op.affine_rates(.03, .008)
    ref = oracle.endpoint_PE_discrepancy(spec, .03, .008)
    identity(op.node_PE() - op.physical_PE(), ref['energy'], op.PE_scale() + abs(ref['energy']))
    identity(rate['node_PE_dot_W'] - rate['physical_PE_dot_W'], ref['direction'], rate['node_PE_scale'] + rate['PE_scale'])
    assert abs(ref['energy']) > 1.
    assert abs(ref['direction']) > .01
    assert not rate['endpoint_pairing_passed']
    with pytest.raises(UnsupportedAffineContract, match='inventory.*physical'):
        op.require_endpoint_pairing(.03, .008)


def test_full_stock_one_sided_PE_chain_slope_and_fixed_stock_controls():
    op, _ = operator()
    rate = op.affine_rates(.03, .008)
    for epsilon in (2.**-8, 2.**-9, 2.**-10):
        ref = oracle.stock_PE_forward(op.state, rate, epsilon, op.node_area_m2)
        assert abs(ref['value'] - rate['node_PE_dot_W']) <= ref['bound']
    assert abs(rate['slope_PE_dot_W']) > 1e-6
    assert abs(rate['slope_PE_dot_W']) > 1000. * 512. * np.finfo(float).eps * max(1., rate['node_PE_scale'])
    physical = oracle.energy_direction(operator()[1], .03, .008)
    omitted_slope_residual = abs(rate['node_PE_dot_W'] - rate['slope_PE_dot_W'] - physical['PE_dot'])
    assert omitted_slope_residual > 1000. * 512. * np.finfo(float).eps * max(1., rate['node_PE_scale'] + physical['scale'])
    fixed = oracle.stock_PE_forward(op.state, dict(rate, stockdot=np.zeros_like(rate['stockdot'])), 2.**-10, op.node_area_m2)
    assert abs(fixed['value'] - rate['node_PE_dot_W']) > .01
    fake = rate['stockdot'].copy()
    fake[:, 3:] = 0.
    with pytest.raises(UnsupportedAffineContract, match='affine.*direction'):
        op.stock_PE_direction(rate['hdot'], fake, rate['etadot'])


def test_real_M_midpoint_and_distinct_KE_metrics():
    op, spec = operator()
    before = op.state.stocks.copy()
    witness = op.fixed_mass_impulse()
    ref = oracle.pressure_power(spec, op.segments, witness['midpoint_u'])
    identity(witness['kinetic_change_J'], ref['value'], witness['scale'] + ref['scale'])
    assert witness['nodal_lumped_mass_metric']
    assert not witness['accepted_state_returned']
    assert oracle.nodal_minus_volume_KE(spec, .03, .008) > 1e-4
    np.testing.assert_array_equal(op.state.stocks, before)


def test_external_pressure_gauge_force_and_boundary_power_invariant():
    op, spec = operator()
    shifted, shifted_spec = operator(external_pressure=(1080., 1092.))
    np.testing.assert_allclose(op.force_N, shifted.force_N, atol=1e-9, rtol=0)
    assert oracle.energy_direction(spec, .03, .008)['boundary_power'] == pytest.approx(oracle.energy_direction(shifted_spec, .03, .008)['boundary_power'], abs=1e-10)


def test_P0_partition_defect_and_nonlinear_TS_density_authority():
    op, _ = operator(density_gradient_x=0., external_pressure=(80., 80.))
    assert abs(oracle.p0_partition_pressure_defect(op.state, -.5)) > 1e-4
    other, spec = operator(density_gradient_x=0., external_pressure=(80., 80.), nonlinear_TS=True)
    for side in range(2):
        for layer in range(14):
            ref = oracle.basis_force(spec, other.segments, side, layer)
            identity(other.force_N[side, layer], ref['value'], other.force_scale_N[side, layer] + ref['scale'])


@pytest.mark.parametrize('invalid', ['near_uniform', 'nonaffine', 'step', 'dry', 'overflow'])
def test_unsupported_state_refuses(invalid):
    profile, _ = manufactured_case()
    state = profile.state
    if invalid == 'step':
        state = replace(state, bottom=state.bottom + np.array([0., -.1]))
    elif invalid == 'dry':
        state = replace(state, active_layers=np.array([14, 0]))
    else:
        stocks = state.stocks.copy()
        stocks[0, 7, 0 if invalid == 'near_uniform' else 1 if invalid == 'nonaffine' else 2] += state.h[0, 7] * 1e-10 if invalid == 'near_uniform' else .01 if invalid == 'nonaffine' else 1e308
        state = replace(state, stocks=stocks)
    with pytest.raises((UnsupportedAffineContract, ValueError)):
        AffinePhysicalDual(replace(profile, state=state), distance_m=2., length_m=1.5)


def test_wrong_area_crossing_and_nonfinite_direction_refuse():
    profile, _ = manufactured_case()
    with pytest.raises(UnsupportedAffineContract, match='area'):
        AffinePhysicalDual(profile, distance_m=2., length_m=1.5, node_area_m2=np.array([1.5, 1.6]))
    op, _ = operator()
    rate = op.affine_rates(.03, .008)
    with pytest.raises(UnsupportedAffineContract, match='cross'):
        op.validate_perturbation(rate, 100.)
    with pytest.raises(ValueError):
        op.affine_rates(np.nan, .008)


@pytest.mark.parametrize('wrong', ['flow', 'Mu', 'Mv'])
def test_direction_binds_actual_momentum_inventory_without_changing_pressure(wrong):
    op, _ = operator()
    if wrong == 'flow':
        with pytest.raises(UnsupportedAffineContract, match='actual.*momentum'):
            op.affine_rates(-.03, -.008)
        return
    profile, _ = manufactured_case()
    stocks = profile.state.stocks.copy()
    stocks[0, 7, 2 if wrong == 'Mu' else 3] += 1025. * profile.state.h[0, 7] * .01
    altered = AffinePhysicalDual(replace(profile, state=replace(profile.state, stocks=stocks)), distance_m=2., length_m=1.5)
    np.testing.assert_array_equal(op.force_N, altered.force_N)
    with pytest.raises(UnsupportedAffineContract, match='actual.*momentum'):
        altered.affine_rates(.03, .008)


def test_zero_flow_does_not_hide_unequal_eta_static_inventory_map_gap():
    op, _ = operator(eta=(-.2, .3), u0=0., strain=0.)
    assert abs(op.node_PE() - op.physical_PE()) > 1.
    with pytest.raises(UnsupportedAffineContract, match='inventory.*physical'):
        op.require_endpoint_pairing(0., 0.)


def test_virtual_cut_crossing_refuses_before_actual_inventory_cell_crossing():
    op, _ = operator(eta=(-.2, .3))
    rate = op.affine_rates(.03, .008)
    assert np.all(op.state.h + .84 * rate['hdot'] > 0.)
    with pytest.raises(UnsupportedAffineContract, match='cross'):
        op.validate_perturbation(rate, .84)


def test_common_slope_must_reconstruct_real_deep_column_density_means():
    profile, _ = manufactured_case()
    old = profile.state
    z = old.eta[:, None] + (old.interfaces - old.eta[:, None]) * ((1e6 + old.eta) / (old.eta + 3.))[:, None]
    z[:, -1] = -1e6
    h = -np.diff(z, axis=-1)
    centers = .5 * (z[:, :-1] + z[:, 1:])
    stocks = np.zeros_like(old.stocks)
    stocks[:, :, 0] = 15. * h
    for side, slope in enumerate((-1e-5, -1e-5 - 1e-12)):
        stocks[side, :, 1] = h[side] * (35. + (1. + slope * centers[side]) / (1025. * 7.6e-4))
    deep = replace(old, bottom=np.full(2, -1e6), interfaces=z, h=h,
                   band_bottom=z[:, 3], stocks=stocks)
    with pytest.raises(UnsupportedAffineContract, match='common.*density'):
        AffinePhysicalDual(replace(profile, state=deep), distance_m=2., length_m=1.5)


def test_common_slope_direction_binds_each_real_deep_stock_rate():
    op, _ = operator(bottom=-1e6, density_slope=-1e-5)
    stockdot = np.zeros_like(op.state.stocks)
    for side, ds in enumerate((1e-5, 1e-5 + 1e-12)):
        stockdot[side, :, 1] = op.state.h[side] * ds * op.centers[side] / (1025. * 7.6e-4)
    with pytest.raises(UnsupportedAffineContract, match='common.*direction'):
        op.stock_PE_direction(np.zeros_like(op.state.h), stockdot, np.zeros(2))


def test_true_near_uniform_density_positive_and_nonaffine_negative_control():
    profile, _ = manufactured_case(density_slope=0., density_gradient_x=0.)
    AffinePhysicalDual(profile, distance_m=2., length_m=1.5)
    stocks = profile.state.stocks.copy()
    stocks[0, 7, 0] += 1e-10 * profile.state.h[0, 7]
    with pytest.raises(UnsupportedAffineContract, match='affine.*density'):
        AffinePhysicalDual(replace(profile, state=replace(profile.state, stocks=stocks)), distance_m=2., length_m=1.5)


def test_raw_interface_ownership_independent_of_candidate_layer_labels():
    op, spec = operator()
    oracle.validate_strip_ownership(spec, op.state, op.segments)
    forged = list(op.segments)
    forged[0] = replace(forged[0], left_layer=(forged[0].left_layer + 1) % 14)
    with pytest.raises(ValueError, match='ownership'):
        oracle.validate_strip_ownership(spec, op.state, forged)


def test_transport_binds_constant_actual_T_at_each_layer():
    profile, _ = manufactured_case()
    stocks = profile.state.stocks.copy()
    stocks[0, 7, 0] += 1e-10 * profile.state.h[0, 7]
    stocks[0, 7, 1] += (2e-4 / 7.6e-4) * 1e-10 * profile.state.h[0, 7]
    op = AffinePhysicalDual(replace(profile, state=replace(profile.state, stocks=stocks)), distance_m=2., length_m=1.5)
    with pytest.raises(UnsupportedAffineContract, match='constant T'):
        op.affine_rates(.03, .008)


def test_geometry_and_difference_probes_reject_nonfinite_or_nonpositive_input():
    op, spec = operator()
    rate = op.affine_rates(.03, .008)
    for key in ('hdot', 'etadot'):
        bad = rate[key].copy()
        bad.flat[0] = np.nan
        with pytest.raises(ValueError, match='finite'):
            op.validate_perturbation(dict(rate, **{key: bad}), 2.**-10)
    for epsilon in (0., -2.**-10, np.nan):
        with pytest.raises(ValueError, match='positive'):
            oracle.stock_PE_forward(op.state, rate, epsilon, op.node_area_m2)
        with pytest.raises(ValueError, match='positive'):
            oracle.geometric_load_variation(spec, op.segments, 0, 0, epsilon)


def test_auxiliary_mass_metric_and_future_geometry_must_remain_resolved():
    profile, _ = manufactured_case()
    with pytest.raises(ValueError, match='area'):
        AffinePhysicalDual(profile, distance_m=1e-200, length_m=1e-200)
    op, _ = operator(u0=.03, strain=-1.)
    rate = op.affine_rates(.03, -1.)
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(ValueError, match='finite'):
            op.validate_perturbation(rate, 1e308)


def test_unstable_affine_density_is_outside_the_contract():
    profile, _ = manufactured_case(density_slope=.01)
    with pytest.raises(UnsupportedAffineContract, match='stable.*density'):
        AffinePhysicalDual(profile, distance_m=2., length_m=1.5)


def test_scalar_receipt_requires_all_gates_and_preserves_qualification_boundary():
    from research.experiments.material_top_band.affine_physical_evidence import build_evidence
    receipt = build_evidence()
    assert receipt['instantaneous_probe_passed']
    assert not receipt['qualification_passed']
    assert receipt['accepted_steps'] == 0
    assert receipt['cases']['flat_positive']['pressure_power_W'] > 0. or receipt['cases']['flat_positive']['pressure_power_W'] < 0.
    assert receipt['cases']['unequal_eta']['endpoint_pairing_passed'] is False
    for case in receipt['cases'].values():
        assert all(gate['passed'] for gate in case['gates'].values())
