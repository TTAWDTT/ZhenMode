"""Actual moving raw commits, independent finite ledgers and full rollback."""
import copy
import re
from dataclasses import fields, replace

import numpy as np
import pytest

from research.experiments.material_top_band.moving_raw_cases import manufactured_mean_case
from research.experiments.material_top_band.moving_raw_characteristic import MovingRawSlice
from research.experiments.material_top_band.moving_raw_oracle import audit_receipt

AUTHORITY = 'manufactured_half_prism_raw_means'


def snapshot(profile):
    return {field.name: getattr(profile.state, field.name).copy() for field in fields(profile.state)}


def unchanged(profile, saved):
    for name, array in saved.items():
        assert np.array_equal(getattr(profile.state, name), array), name


@pytest.mark.parametrize('sign', [1., -1.])
def test_two_actual_moving_steps(sign):
    external = (80., 200.) if sign > 0 else (200., 80.)
    profile, spec = manufactured_mean_case(U=sign*.03, alpha=sign*.08, external=external)
    integrator = MovingRawSlice(profile, authority=AUTHORITY)
    original = snapshot(profile)
    for dt in (.02, .03):
        before = integrator.profile
        receipt = integrator.step(dt)
        evidence = audit_receipt(spec, receipt)
        assert evidence['passed']
        assert receipt.before is before
        assert receipt.profile is integrator.profile
        assert np.any(receipt.profile.state.h != before.state.h)
        assert np.any(receipt.profile.state.stocks[:, 3:, 1] != before.state.stocks[:, 3:, 1])
        assert np.any(receipt.profile.state.stocks[:, 3:, 2] != before.state.stocks[:, 3:, 2])
        assert np.linalg.norm(receipt.M_after - receipt.M_before) > 0.
        assert np.linalg.norm(receipt.R_after - receipt.R_before) > 0.
        assert receipt.physical_KE_change_J != receipt.raw_KE_change_J
        assert abs(receipt.midpoint_pressure_work_J - receipt.true_pressure_work_J) > 100. * receipt.work_error_bound_J
        for chain in (receipt.raw_mass_chain,receipt.physical_mass_chain):
            assert abs(chain['mass_work']) > 100.*512.*np.finfo(float).eps*max(1.,chain['scale'])
        assert receipt.accepted_moving_geometry
        assert not receipt.production_qualified
        assert not receipt.original_global_CV_identified
    assert integrator.accepted_raw_steps == 2 and integrator.time_s == .05
    unchanged(profile, original)


@pytest.mark.parametrize('alpha,external,moving', [(.08, (80.,80.), True), (0.,(80.,200.),False)])
def test_zero_pressure_and_fixed_eta_controls(alpha, external, moving):
    profile, spec = manufactured_mean_case(alpha=alpha, external=external)
    receipt = MovingRawSlice(profile, authority=AUTHORITY).step(.02)
    assert audit_receipt(spec, receipt)['passed']
    assert receipt.accepted_moving_geometry == moving
    if external[0] == external[1]:
        assert receipt.true_pressure_work_J == 0.


@pytest.mark.parametrize('fault', ['zero_stock', 'Pa_offset', 'no_R', 'no_internal_Pa_energy', 'no_cap_Pa_energy', 'no_KE'])
def test_absolute_face_faults_reject_with_full_rollback(monkeypatch, fault):
    profile, _ = manufactured_mean_case()
    integrator = MovingRawSlice(profile, authority=AUTHORITY)
    original_faces = integrator._faces
    saved = (integrator.profile, integrator.time_s, integrator.accepted_raw_steps, integrator.last_receipt, integrator.cumulative_gcl.copy())
    arrays = snapshot(integrator.profile)
    saved_bound = integrator.cumulative_gcl_bound.copy()

    def corrupted(binding, dt):
        result = copy.deepcopy(original_faces(binding, dt))
        for row in result['horizontal']:
            if fault == 'zero_stock':
                row['transport'][:, 1:5] = 0.
            if fault == 'Pa_offset':
                row['pressure'] += 1.
            if fault == 'no_KE':
                row['transport'][:, 5] = 0.
        for row in result['vertical']:
            if fault == 'no_R':
                row['transport'][:] = 0.
            if fault == 'no_internal_Pa_energy' and 0 < row['interface'] < 14:
                row['pressure_energy'] = 0.
            if fault == 'no_cap_Pa_energy' and row['interface'] == 0:
                row['pressure_energy'] = 0.
        return result

    monkeypatch.setattr(integrator, '_faces', corrupted)
    with pytest.raises(ValueError) as failure:
        integrator.step(.02)
    ratio = re.search(r'ratio ([0-9.e+\-]+)', str(failure.value))
    assert ratio is not None and float(ratio.group(1)) > 100.
    assert integrator.profile is saved[0] and integrator.last_receipt is saved[3]
    assert (integrator.time_s, integrator.accepted_raw_steps) == saved[1:3]
    assert np.array_equal(integrator.cumulative_gcl, saved[4])
    unchanged(integrator.profile, arrays)
    assert np.array_equal(integrator.cumulative_gcl_bound,saved_bound)


def test_postprepare_audit_failure_preserves_every_field(monkeypatch):
    profile, _ = manufactured_mean_case()
    integrator = MovingRawSlice(profile, authority=AUTHORITY)
    integrator.step(.02)
    saved = (integrator.profile, integrator.time_s, integrator.accepted_raw_steps, integrator.last_receipt, integrator.cumulative_gcl.copy())
    arrays = snapshot(integrator.profile)
    saved_bound = integrator.cumulative_gcl_bound.copy()
    monkeypatch.setattr(integrator, '_audit', lambda *_: (_ for _ in ()).throw(ValueError('injected final audit')))
    with pytest.raises(ValueError, match='injected'):
        integrator.step(.03)
    assert integrator.profile is saved[0] and integrator.last_receipt is saved[3]
    assert (integrator.time_s, integrator.accepted_raw_steps) == saved[1:3]
    assert np.array_equal(integrator.cumulative_gcl, saved[4])
    unchanged(integrator.profile, arrays)
    assert np.array_equal(integrator.cumulative_gcl_bound,saved_bound)


def test_raw_owner_corruption_refuses_before_consumption(monkeypatch):
    profile,_ = manufactured_mean_case()
    integrator = MovingRawSlice(profile,authority=AUTHORITY)
    original_faces = integrator._faces
    saved = snapshot(integrator.profile)

    def corrupted(binding,dt):
        faces = original_faces(binding,dt)
        faces['horizontal'][0]['owners'] = (0,0)
        return faces

    monkeypatch.setattr(integrator,'_faces',corrupted)
    with pytest.raises(ValueError,match='ownership'):
        integrator.step(.02)
    unchanged(integrator.profile,saved)
    assert integrator.accepted_raw_steps == 0 and integrator.last_receipt is None
    assert integrator.time_s == 0. and np.all(integrator.cumulative_gcl == 0.) and np.all(integrator.cumulative_gcl_bound == 0.)


def test_raw_mean_KE_substitution_refuses_after_good_faces(monkeypatch):
    from research.experiments.material_top_band import moving_raw_characteristic as candidate
    profile,_ = manufactured_mean_case()
    integrator = MovingRawSlice(profile,authority=AUTHORITY)
    original_energy = candidate.local_energy
    saved = snapshot(integrator.profile)

    def without_covariance(geometry):
        energy = original_energy(geometry)
        energy[...,0] = (.5*geometry.D.diagonal()*np.sum(geometry.means**2,axis=-1)).reshape(2,14)
        return energy

    monkeypatch.setattr(candidate,'local_energy',without_covariance)
    with pytest.raises(ValueError) as failure:
        integrator.step(.02)
    ratio = re.search(r'ratio ([0-9.e+\-]+)',str(failure.value))
    assert ratio is not None and float(ratio.group(1)) > 100.
    unchanged(integrator.profile,saved)
    assert integrator.accepted_raw_steps == 0 and integrator.last_receipt is None
    assert integrator.time_s == 0. and np.all(integrator.cumulative_gcl == 0.) and np.all(integrator.cumulative_gcl_bound == 0.)


@pytest.mark.parametrize('dt', [True, 0., -.01, .051, np.nan, np.inf, 1j])
def test_invalid_duration_refuses(dt):
    profile, _ = manufactured_mean_case()
    integrator = MovingRawSlice(profile, authority=AUTHORITY)
    with pytest.raises(ValueError):
        integrator.step(dt)
    assert integrator.accepted_raw_steps == 0


def test_limiter_active_family_refuses():
    profile, _ = manufactured_mean_case(density_intercept=11.2176779, density_slope=-.1558)
    with pytest.raises(ValueError, match='limiter'):
        MovingRawSlice(profile, authority=AUTHORITY)


@pytest.mark.parametrize('fault', ['shear', 'a1', 'nonflat', 'nonaffine'])
def test_unsupported_family_refuses(fault):
    from research.experiments.material_top_band import inventory_pressure as inventory
    profile, _ = manufactured_mean_case()
    state, stocks = profile.state, profile.state.stocks.copy()
    if fault == 'shear':
        stocks[0, 6, 2] += .01
    elif fault == 'a1':
        stocks[1, :, 1] += .1 * state.h[1]
    elif fault == 'nonaffine':
        stocks[0, 6, 1] += .01
    else:
        z, h = state.interfaces.copy(), state.h.copy()
        z[1, 0] += .001
        h[1, 0] += .001
        state = replace(state, eta=z[:, 0].copy(), interfaces=z, h=h)
    profile = inventory.reconstruct(replace(state, stocks=stocks), external_pressure_Pa=profile.external_pressure_Pa)
    with pytest.raises(ValueError):
        MovingRawSlice(profile, authority=AUTHORITY)


@pytest.mark.parametrize('bad', [np.nan, -1., 1j, np.ma.array(1.,mask=True)])
def test_bad_receipt_scale_rejects_before_floor(bad):
    from research.experiments.material_top_band.moving_raw_oracle import assert_bound
    with pytest.raises(ValueError):
        assert_bound(0.,0.,bad)


def test_nonfinite_bound_or_difference_cannot_pass():
    from research.experiments.material_top_band.moving_raw_oracle import assert_bound
    with pytest.raises(ValueError):
        assert_bound(0.,0.,1.7e308,extra=1.7976931348623157e308)
    with pytest.raises(ValueError):
        assert_bound(1.7e308,-1.7e308,1.)
    with pytest.raises(ValueError):
        assert_bound(np.iinfo(np.int64).min,0,1.)
