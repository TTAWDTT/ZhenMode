"""Independent physical-time area bounds for expanding canonical top faces."""
import math
from dataclasses import replace

import numpy as np
import pytest

from research.experiments.material_top_band.moving_raw_cases import manufactured_mean_case
from research.experiments.material_top_band.moving_raw_characteristic import MovingRawSlice
from research.experiments.material_top_band.moving_raw_oracle import audit_receipt


@pytest.mark.parametrize('alpha,dt', [(-.08,.02),(-.08,.05),(-.398,.05),(.08,.02),(0.,.05)])
def test_area_upper_bounds_independent_canonical_path(alpha,dt):
    profile,spec = manufactured_mean_case(alpha=alpha)
    integrator = MovingRawSlice(profile,authority='manufactured_half_prism_raw_means')
    binding = integrator._binding(integrator.profile)
    faces = integrator._faces(binding,dt)
    envelope = integrator._input_envelope(binding,dt)
    delta_eta = envelope['deta']
    H = spec.eta-spec.bottom
    nodes,weights = np.polynomial.legendre.leggauss(32)
    times = .5*dt*(nodes+1.)
    for row in faces['horizontal']:
        top = row['upper'] == spec.eta
        endpoint_heights = np.array([H/(1.+alpha*t)+spec.bottom-row['lower'] if top else row['upper']-row['lower'] for t in (0.,dt)])
        instantaneous = (H/(1.+alpha*times)+spec.bottom-row['lower']) if top else np.full(32,row['upper']-row['lower'])
        exact_area = spec.length*(H*math.log1p(alpha*dt)/alpha+(spec.bottom-row['lower'])*dt) if top and alpha != 0. else spec.length*(row['upper']-row['lower'])*dt
        quadrature_area = .5*spec.length*dt*math.fsum(weights*instantaneous)
        scale = spec.length*dt*(H+abs(spec.bottom)+abs(row['lower']))
        rounding = 512.*np.finfo(float).eps*max(1.,scale)
        # Water's content is one, so independently recover the area actually
        # used by its input-product majorant before checking diagnostic fields.
        boundary_error = spec.length*dt*delta_eta*(envelope['umax']+envelope['du'])
        used_area = (row['transport_input'][0,0]-boundary_error)/envelope['du']
        assert exact_area <= used_area+rounding
        maximum = row['canonical_max_width_m']
        area_upper = row['input_area_time_bound_m2_s']
        assert all(math.isfinite(value) and value > 0. for value in (maximum,area_upper,used_area))
        independent_sup_area = spec.length*dt*(float(np.max(endpoint_heights))+delta_eta)
        assert independent_sup_area <= used_area+rounding
        assert abs(used_area-area_upper) <= rounding
        assert abs(maximum-float(np.max(endpoint_heights))) <= rounding/(spec.length*dt)
        if not top or alpha >= 0.:
            assert abs(maximum-(row['upper']-row['lower'])) <= rounding/(spec.length*dt)
        assert np.max(endpoint_heights) <= maximum+rounding/(spec.length*dt)
        assert np.max(instantaneous) <= maximum+rounding/(spec.length*dt)
        assert abs(quadrature_area-exact_area) <= rounding
        assert exact_area <= area_upper+rounding
        assert area_upper == pytest.approx(spec.length*dt*(maximum+delta_eta),abs=0.,rel=3e-15)
        if alpha < 0. and top:
            old_incomplete_bound = spec.length*dt*(row['upper']-row['lower']+delta_eta)
            assert (exact_area-old_incomplete_bound)/rounding > 100.


def test_negative_alpha_near_legal_duration_is_actual_raw_commit():
    profile,spec = manufactured_mean_case(U=-.03,alpha=-.398,external=(200.,80.))
    integrator = MovingRawSlice(profile,authority='manufactured_half_prism_raw_means')
    receipt = integrator.step(.05)
    assert audit_receipt(spec,receipt)['passed']
    assert receipt.accepted_raw_steps == 1 and receipt.accepted_moving_geometry
    assert np.max(receipt.profile.state.h[:,0]-profile.state.h[:,0]) > .05
    assert np.any(receipt.profile.state.stocks[:,3:,1] != profile.state.stocks[:,3:,1])
    assert np.any(receipt.profile.state.stocks[:,3:,2] != profile.state.stocks[:,3:,2])
    assert not receipt.production_qualified and not receipt.original_global_CV_identified


def test_positive_surface_domain_PE_content_covers_expansion():
    from research.experiments.material_top_band import inventory_pressure as inventory
    profile,_ = manufactured_mean_case(alpha=-.398)
    state = profile.state
    offset = 3.2
    shifted = replace(state,eta=state.eta+offset,bottom=state.bottom+offset,interfaces=state.interfaces+offset,band_bottom=state.band_bottom+offset)
    profile = inventory.reconstruct(shifted,external_pressure_Pa=profile.external_pressure_Pa)
    integrator = MovingRawSlice(profile,authority='manufactured_half_prism_raw_means')
    binding = integrator._binding(integrator.profile)
    spec,dt = binding['spec'],.05
    envelope = integrator._input_envelope(binding,dt)
    H,q = spec.eta-spec.bottom,abs(spec.alpha*dt)
    rho_max = abs(spec.density_intercept+spec.density_slope*spec.bottom)+abs(spec.density_slope)*(1.+q)*H/(1.-q)
    implied_z_max = envelope['content'][6]/(profile.eos.gravity*(profile.eos.rho0+rho_max))
    end_eta = spec.bottom+H/(1.+spec.alpha*dt)
    assert end_eta > spec.eta
    assert end_eta <= implied_z_max+512.*np.finfo(float).eps*(abs(spec.eta)+H)
