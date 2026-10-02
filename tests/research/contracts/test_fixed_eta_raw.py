"""Actual raw-CV pressure steps and independently rejected moving cases."""

from dataclasses import replace

import numpy as np
import pytest

from research.experiments.material_top_band import fixed_eta_raw_oracle as oracle
from research.experiments.material_top_band import inventory_pressure as inventory
from research.experiments.material_top_band.affine_physical_cases import manufactured_case
from research.experiments.material_top_band.fixed_eta_raw import (
    FixedEtaRawPressureSlice,
    variable_mass_identity,
)


def fixture(**parameters):
    parameters.setdefault('density_gradient_x', 0.)
    parameters.setdefault('strain', 0.)
    profile, spec = manufactured_case(**parameters)
    return FixedEtaRawPressureSlice(profile, distance_m=2., length_m=1.5), spec


def close(value, reference, scale):
    assert all(np.isfinite(v).all() for v in map(np.asarray, (value, reference, scale)))
    assert np.all(abs(np.asarray(value) - reference) <= 512. * np.finfo(float).eps * np.maximum(1., scale))


@pytest.mark.parametrize('sign', [1., -1.])
def test_two_actual_raw_commits_faces_force_inverse_and_boundary_energy(sign):
    stepper, spec = fixture(u0=.03 * sign, external_pressure=(80., 92.) if sign > 0 else (92., 80.))
    original = stepper.profile.state.stocks.copy()
    for dt in (.01, .02):
        before = stepper.profile
        result = stepper.advance(dt)
        ref = oracle.audit(before, result, spec)
        assert result.state is stepper.profile.state
        assert result.accepted_raw_steps == stepper.accepted_raw_steps
        assert not result.production_qualified
        close(result.raw_force, ref['force'], result.raw_force_scale + ref['force_scale'])
        close(result.net_water, ref['net_water'], result.ledger_scale[:, :, 0] + ref['transport_scale'][:, :, 0])
        close(result.net_stocks, ref['net_stocks'], result.ledger_scale[:, :, 1:] + ref['transport_scale'][:, :, 1:])
        close(1.5 * (result.state.stocks - before.state.stocks), ref['stock_increment'],
              1.5 * (abs(result.state.stocks) + abs(before.state.stocks)) + dt * ref['force_scale'][:, :, None])
        for candidate, independent in zip(result.faces, ref['faces'], strict=True):
            assert (candidate['left_layer'], candidate['right_layer']) == (independent['left_layer'], independent['right_layer'])
            close(candidate['transport'], independent['transport'], candidate['transport_scale'] + independent['transport_scale'])
            close(candidate['KE_transport'], independent['KE_transport'], candidate['KE_scale'] + independent['KE_scale'])
            close(candidate['PE_transport'], independent['PE_transport'], candidate['PE_scale'] + independent['PE_scale'])
            close(candidate['pressure'], independent['pressure'], candidate['pressure_scale'])
        close(result.net_energy, 0., result.energy_flux_scale)
        close(result.net_energy, ref['net_energy'], result.energy_flux_scale + ref['energy_flux_scale'])
        close(result.raw_KE_change_J, ref['KE_change'], result.energy_scale + ref['energy_scale'])
        close(result.raw_KE_change_J, ref['boundary_pressure_work'], result.energy_scale + ref['energy_scale'])
        close(result.PE_change_J, 0., result.energy_scale)
        np.testing.assert_array_equal(result.state.h, before.state.h)
        np.testing.assert_array_equal(result.state.interfaces, before.state.interfaces)
        np.testing.assert_array_equal(result.state.stocks[:, :, :2], before.state.stocks[:, :, :2])
        np.testing.assert_array_equal(result.state.stocks[:, :, 3], before.state.stocks[:, :, 3])
        assert np.max(abs(result.state.stocks[:, 3:, 2] - before.state.stocks[:, 3:, 2])) > 1e-5
    assert stepper.accepted_raw_steps == 2
    assert stepper.time_s == .03
    assert not np.array_equal(stepper.profile.state.stocks[:, :, 2], original[:, :, 2])


def test_sign_crossing_has_zero_water_but_nonzero_time_integrated_Mu_flux():
    stepper, spec = fixture(u0=(6. / 1025.) * .02 / 2.)
    before = stepper.profile
    result = stepper.advance(.02)
    ref = oracle.audit(before, result, spec)
    assert max(abs(row['transport'][0]) for row in result.faces) < 1e-15
    assert max(row['transport'][3] for row in result.faces) > 1e-9
    for row, independent in zip(result.faces, ref['faces'], strict=True):
        close(row['transport'][3], independent['transport'][3], row['transport_scale'][3] + independent['transport_scale'][3])
        assert independent['transport'][3] > 0.


def test_P1_flux_and_shared_segment_omission_controls_are_resolved():
    stepper, spec = fixture()
    before = stepper.profile
    result = stepper.advance(.01)
    ref = oracle.audit(before, result, spec)
    wrong_P0 = []
    for row in ref['faces']:
        layer = row['left_layer']
        wrong = row['transport'][0] * before.salinity_mean[0, layer]
        wrong_P0.append(abs(wrong - row['transport'][2]))
    assert max(wrong_P0) > 1e-7
    assert max(abs(row['transport'][2]) for row in result.faces) > 1e-4


@pytest.mark.parametrize('fault', ['missing', 'owner', 'pressure', 'flux'])
def test_bad_face_after_preparation_rolls_back_all_raw_clock_and_receipt(monkeypatch, fault):
    stepper, _ = fixture()
    stepper.advance(.01)
    before, clock, count, receipt = stepper.profile, stepper.time_s, stepper.accepted_raw_steps, stepper.last_receipt
    original_faces = stepper._faces
    def broken(*args):
        rows = [dict(row) for row in original_faces(*args)]
        if fault == 'missing':
            rows.pop(0)
        elif fault == 'owner':
            rows[0]['left_layer'] = (rows[0]['left_layer'] + 1) % 14
        elif fault == 'pressure':
            rows[0]['pressure'] = np.full(3, np.nan)
        else:
            rows[0]['outer_transport'] = rows[0]['outer_transport'].copy()
            rows[0]['outer_transport'][0, 2] += .1
        return rows
    monkeypatch.setattr(stepper, '_faces', broken)
    with pytest.raises(ValueError):
        stepper.advance(.01)
    assert stepper.profile is before
    assert stepper.time_s == clock and stepper.accepted_raw_steps == count
    assert stepper.last_receipt is receipt


@pytest.mark.parametrize('dt', [0., -.01, np.nan, np.inf, True, .101, '0.01', 1j])
def test_invalid_duration_full_rollback(dt):
    stepper, _ = fixture()
    before = stepper.profile
    with pytest.raises(ValueError):
        stepper.advance(dt)
    assert stepper.profile is before and stepper.time_s == 0. and stepper.accepted_raw_steps == 0
    assert stepper.last_receipt is None


@pytest.mark.parametrize('parameters', [dict(eta=(-.2, .3)), dict(strain=.008), dict(density_gradient_x=.15), dict(nonlinear_TS=True)])
def test_unsupported_original_family_is_not_silently_accepted(parameters):
    with pytest.raises(ValueError):
        fixture(**parameters)


def test_sloped_same_velocity_inverse_and_flat_metric_storage_obstructions():
    ref = oracle.obstructions()
    close(ref['sloped_Mu_gap'], 4.1, ref['sloped_scale'])
    close(ref['sloped_IS_gap'], 50. / 779., ref['sloped_scale'])
    assert abs(ref['sloped_PE_gap'] - 629.17048125) < 1e-9
    assert abs(ref['moving_local_storage_change']) > 1e-5
    assert abs(ref['moving_KE_gap_change']) > 1e-5
    assert ref['moving_accepted_steps'] == 0


def test_true_moving_mass_identity_and_Wdot_Rdot_omission_controls():
    ref = oracle.moving_snapshots()
    for name in ('raw', 'physical'):
        row = ref[name]
        identity = variable_mass_identity(row['mass_before'], row['mass_after'], row['u_before'], row['u_after'])
        close(identity['kinetic_change'], identity['impulse_work'] - identity['mass_work'], identity['scale'])
        close(identity['kinetic_change'], row['KE_after'] - row['KE_before'], identity['scale'])
        assert abs(identity['mass_work']) > 1e-5
        assert abs(identity['kinetic_change'] - identity['impulse_work']) > 1e-5
    assert ref['omitted_Wdot_max_residual'] > 1e-5
    assert ref['omitted_Rdot_max_residual'] > 1e-5


def test_pressure_force_depends_on_thermodynamics_geometry_not_velocity():
    first, _ = fixture(u0=.03, velocity_y=-.02)
    second, _ = fixture(u0=-.07, velocity_y=.04)
    np.testing.assert_array_equal(first.advance(.01).raw_force, second.advance(.01).raw_force)


def test_failure_after_actual_stock_preparation_keeps_complete_previous_commit(monkeypatch):
    stepper, _ = fixture()
    result = stepper.advance(.01)
    before, clock = stepper.profile, stepper.time_s
    def reject(*args):
        raise ValueError('independent candidate ledger rejection')
    monkeypatch.setattr(stepper, '_audit', reject)
    with pytest.raises(ValueError, match='ledger rejection'):
        stepper.advance(.01)
    assert stepper.profile is before and stepper.last_receipt is result
    assert stepper.time_s == clock and stepper.accepted_raw_steps == 1


@pytest.mark.parametrize('fault', ['all_zero', 'paired_IS_zero', 'paired_IS_P0', 'common_Pa_offset', 'KE_nan', 'paired_KE_zero', 'paired_PE_zero', 'outer_PE'])
def test_common_face_error_cannot_hide_in_zero_divergence(monkeypatch, fault):
    stepper, _ = fixture()
    previous = stepper.advance(.01)
    before, clock, count = stepper.profile, stepper.time_s, stepper.accepted_raw_steps
    original = stepper._faces
    def broken(*args):
        rows = [dict(row) for row in original(*args)]
        row = rows[0]
        row['transport'] = row['transport'].copy()
        row['outer_transport'] = row['outer_transport'].copy()
        row['outer_energy'] = row['outer_energy'].copy()
        if fault == 'all_zero':
            row['transport'][:] = 0.
            row['outer_transport'][:] = 0.
        elif fault.startswith('paired_IS'):
            value = 0. if fault.endswith('zero') else row['transport'][0] * before.salinity_mean[0, row['left_layer']]
            row['transport'][2] = value
            row['outer_transport'][:, 2] = value
        elif fault == 'common_Pa_offset':
            row['pressure'] = row['pressure'] + 1.
        elif fault == 'KE_nan':
            row['KE_transport'] = np.nan
        elif fault in ('paired_KE_zero', 'paired_PE_zero'):
            slot, key = (0, 'KE_transport') if fault == 'paired_KE_zero' else (1, 'PE_transport')
            row[key] = 0.
            row['outer_energy'][:, slot] = 0.
        else:
            row['outer_energy'][0, 1] += .1
        return rows
    monkeypatch.setattr(stepper, '_faces', broken)
    with pytest.raises(ValueError, match='independent actual face'):
        stepper.advance(.01)
    assert stepper.profile is before and stepper.last_receipt is previous
    assert stepper.time_s == clock and stepper.accepted_raw_steps == count


@pytest.mark.parametrize('fault', ['spec_density', 'before_T', 'before_Mu', 'after_Mu'])
def test_independent_oracle_binds_actual_before_and_after_stocks(fault):
    stepper, spec = fixture()
    before = stepper.profile
    result = stepper.advance(.01)
    if fault == 'spec_density':
        spec = replace(spec, density_intercept=spec.density_intercept + .1)
    else:
        state = result.state if fault == 'after_Mu' else before.state
        stocks = state.stocks.copy()
        stocks[0, 0, 0 if fault == 'before_T' else 2] += .01
        changed = replace(state, stocks=stocks)
        if fault == 'after_Mu':
            result = replace(result, state=changed)
        else:
            before = inventory.reconstruct(changed, external_pressure_Pa=spec.external)
    with pytest.raises(ValueError):
        oracle.audit(before, result, spec)


def test_zero_pressure_keeps_actual_raw_state_with_nonzero_verified_transport():
    stepper, spec = fixture(external_pressure=(80., 80.))
    before = stepper.profile
    result = stepper.advance(.01)
    oracle.audit(before, result, spec)
    np.testing.assert_array_equal(before.state.stocks, result.state.stocks)
    assert max(abs(row['transport'][2]) for row in result.faces) > 1e-4
    close(result.raw_KE_change_J, 0., result.energy_scale)


def test_actual_vertical_shear_is_rejected():
    profile, _ = manufactured_case(density_gradient_x=0., strain=0.)
    stocks = profile.state.stocks.copy()
    stocks[0, 7, 2] += .1
    with pytest.raises(ValueError, match='uniform actual CV velocity'):
        FixedEtaRawPressureSlice(replace(profile, state=replace(profile.state, stocks=stocks)), distance_m=2., length_m=1.5)


def test_wrong_raw_force_full_vector_rejects_before_commit(monkeypatch):
    stepper, _ = fixture()
    before = stepper.profile
    original = stepper._consume
    def wrong(*args):
        net, scale, force, force_scale, energy, energy_scale = original(*args)
        force[0, 0] += .1
        force[1, 0] -= .1  # global impulse alone still agrees
        return net, scale, force, force_scale, energy, energy_scale
    monkeypatch.setattr(stepper, '_consume', wrong)
    with pytest.raises(ValueError):
        stepper.advance(.01)
    assert stepper.profile is before and stepper.accepted_raw_steps == 0
    assert stepper.time_s == 0. and stepper.last_receipt is None


def test_moving_state_rejection_preserves_current_state_clock_count_and_receipt():
    stepper, _ = fixture()
    receipt = stepper.advance(.01)
    moving, _ = manufactured_case(density_gradient_x=0., strain=.008)
    stepper.profile = moving
    with pytest.raises(ValueError):
        stepper.advance(.01)
    assert stepper.profile is moving and stepper.last_receipt is receipt
    assert stepper.time_s == .01 and stepper.accepted_raw_steps == 1


def test_limited_actual_S_reconstruction_is_not_replaced_by_density_P1():
    profile, _ = manufactured_case(density_gradient_x=0., strain=0., density_slope=-.1558, density_intercept=11.2176779)
    assert profile.slope_limited_count > 0 and profile.density_slope_limited_count == 0
    assert np.max(profile.salinity_mean) < 50.
    with pytest.raises(ValueError, match='limiter'):
        FixedEtaRawPressureSlice(profile, distance_m=2., length_m=1.5)
    with pytest.raises(ValueError, match='limiter'):
        oracle.actual_face_reference(profile, .01, 2., 1.5)


@pytest.mark.parametrize('scale', [np.nan, np.inf, -1.])
def test_receipt_operation_scale_is_validated_before_floor(scale):
    from research.experiments.material_top_band.fixed_eta_raw_evidence import compare
    class UnusedGate:
        def check(self, *args):
            raise AssertionError('invalid scale reached gate')
    with pytest.raises(ValueError, match='primitive operation scale'):
        compare(UnusedGate(), 0., 0., scale)
