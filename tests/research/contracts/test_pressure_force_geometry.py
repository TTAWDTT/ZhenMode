"""New sigma momentum-dual identities; old Cmin failures remain regressions."""
from dataclasses import replace

import numpy as np
import pytest

from research.experiments.material_top_band import inventory_pressure as p
from research.experiments.material_top_band.force_contour_oracle import (
    _minmod_direction,
    contour,
    stock_pe_direction,
)
from research.experiments.material_top_band.force_geometry import (
    BarotropicOperator,
    UnsupportedGeometry,
    static_contours,
)
from research.experiments.material_top_band.force_geometry_cases import columns


def operator(**kwargs):
    return BarotropicOperator(*columns(**kwargs))


def test_periodic_H123_closes_physical_force_and_preserves_old_Cmin_failure():
    op = operator()
    np.testing.assert_allclose(op.force_total(), [0., 0.], atol=1e-10, rtol=0)
    old = p.certify_force_consumption(op.profile, op.requests, op.walls, footprint=op.footprint)
    assert not old['accepted']
    assert old['incompatibility_N'][0] == pytest.approx(-1025. * 9.81 * 2.)
    assert all(item['accepted'] for item in op.contour_certificate())
    np.testing.assert_allclose(old['physical_boundary_force_N'], op.force_total(), atol=1e-10, rtol=0)


@pytest.mark.parametrize('slots', [1, 3, 6, 14])
def test_full_column_constant_specific_transport_and_ALE(slots):
    op = operator(slots=slots, temperature=16., salinity=34., velocity=(.03, -.02))
    rate = op.rates()
    expected = rate.hdot[..., None] * np.array([16., 34., 1025. * .03, -1025. * .02])
    np.testing.assert_allclose(rate.stock_transport, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(rate.hdot, -rate.divergence + rate.relative_flux[..., :-1] - rate.relative_flux[..., 1:], atol=1e-14)
    assert np.max(abs(rate.etadot)) > .01
    assert np.all(rate.relative_flux[:, [0, slots]] == 0.)
    assert np.all(rate.hdot[:, 3:] == 0.)
    np.testing.assert_allclose(rate.hdot.sum(axis=-1), rate.etadot, atol=1e-14)
    assert rate.transport_budget['water_residual_m3_s'] == pytest.approx(0., abs=1e-13)
    np.testing.assert_allclose(rate.transport_budget['stock_residual_per_s'], 0., atol=1e-11)


def test_nonzero_shear_changes_crossband_and_deep_stocks_without_deep_water_change():
    op = operator(shear=True)
    rate = op.rates()
    assert np.max(abs(rate.relative_flux[:, 3])) > 1e-3
    assert np.max(abs(rate.stock_transport[:, 3:, 2:])) > 1e-2
    assert np.all(rate.hdot[:, 3:] == 0.)
    assert rate.kinetic_transport['residual_ratio'] < 1.
    assert rate.kinetic_transport['upwind_loss_W'] > 0.


@pytest.mark.parametrize('temperature,salinity', [(15., 35.), (17., 34.)])
def test_true_stock_PE_direction_and_external_power(temperature, salinity):
    op = operator(temperature=temperature, salinity=salinity, shear=True, external_pressure=(100., 130., 90.))
    rate = op.rates()
    independent = stock_pe_direction(op.profile, op.footprint.area_m2, rate.hdot, rate.stock_transport, rate.etadot)
    assert abs(independent - rate.energy['gravity_PE_rate_W']) <= rate.energy['bound_W']
    assert rate.energy['pairing_residual_ratio'] < 1.
    assert abs(rate.energy['external_pressure_power_W']) > .1
    assert abs(rate.energy['pressure_power_W']) > .1
    assert op.midpoint_identity()['work_roundoff_ratio'] < 1.


def test_uniform_external_pressure_gauge_preserves_operator_and_rates():
    base, shifted = operator(shear=True), operator(shear=True, external_pressure=1234.)
    np.testing.assert_array_equal(base.potential(), shifted.potential())
    np.testing.assert_array_equal(base.rates().pressure_momentum, shifted.rates().pressure_momentum)
    for a, b in zip(base.contour_certificate(), shifted.contour_certificate()):
        np.testing.assert_allclose(a['contour_force_N'], b['contour_force_N'], atol=1e-9, rtol=0)


def test_closed_dual_has_nonzero_top_bottom_contours_and_physical_total():
    op = operator(eta=(0., 1.), periodic=False)
    expected = -1025. * 9.81 * 2. * 1.5
    assert op.force_total()[0] == pytest.approx(expected, abs=1e-9)
    boundary = p.boundary_force_budget(op.profile, op.requests, op.walls)
    np.testing.assert_allclose(boundary['physical_boundary_force_N'], op.force_total(), atol=1e-9, rtol=0)
    result = op.contour_certificate()
    assert all(r['accepted'] for r in result)
    assert any(abs(r['components_N']['top']) > 1. for r in result)
    assert any(abs(r['components_N']['bottom']) > 1. for r in result)


def test_general_stable_P1_static_contour_matches_independent_oracle_but_consumption_refused():
    inputs = columns(eta=(0., .2), bottom=-10., periodic=False,
                     temperature=lambda lo, hi: 15. + .2 * .5 * (lo + hi))
    profile, footprint, requests, walls = inputs
    with pytest.raises(UnsupportedGeometry, match='general_P1'):
        BarotropicOperator(*inputs)
    diagnostics = static_contours(*inputs)
    assert diagnostics and all(not r['force_consumption_qualified'] for r in diagnostics)
    for row in diagnostics:
        independent = contour(profile, row['segment'])
        assert abs(row['contour_force_N'] - independent['force_N']) <= row['bound_N']
    assert np.min(profile.density_slope[profile.state.wet_mask]) < 0.


def test_step_bottom_is_refused_without_hidden_ramp_or_force_correction():
    inputs = columns(eta=(0., .2), bottom=(-10., -12.), periodic=False)
    with pytest.raises(UnsupportedGeometry, match='step_bottom'):
        BarotropicOperator(*inputs)
    with pytest.raises(UnsupportedGeometry, match='step_bottom'):
        static_contours(*inputs)


def test_wrong_band_metadata_and_incomplete_footprint_are_refused():
    profile, footprint, requests, walls = columns()
    bad = p.reconstruct(replace(profile.state, band_bottom=profile.state.band_bottom + .01))
    with pytest.raises(ValueError, match='band_bottom'):
        BarotropicOperator(bad, footprint, requests, walls)
    with pytest.raises(ValueError, match='incomplete'):
        BarotropicOperator(profile, footprint, requests, ())


def test_input_state_unchanged_and_operator_owns_copies():
    profile, footprint, requests, walls = columns()
    original = profile.state.stocks.copy()
    op = BarotropicOperator(profile, footprint, requests, walls)
    footprint.area_m2[:] = 99.
    op.rates()
    op.midpoint_identity()
    np.testing.assert_array_equal(profile.state.stocks, original)
    np.testing.assert_array_equal(op.footprint.area_m2, [2., 2., 2.])
    with pytest.raises(ValueError):
        op.profile.state.stocks[:] = 0.


def test_arbitrary_velocity_C_and_transpose_use_identical_segment_weights():
    op = operator()
    velocity = np.arange(op.profile.state.h.size * 2).reshape(op.profile.state.h.shape + (2,)) * .001
    potential = np.linspace(-3., 2., len(op.segments))
    force = op.transpose(potential)
    left = np.sum(op.footprint.area_m2[:, None, None] * velocity * force)
    right = -np.dot(potential, op.shared_flux(velocity))
    assert left == pytest.approx(right, abs=1e-12)
    with pytest.raises(ValueError):
        op.shared_flux(np.full_like(velocity, np.nan))


def test_varying_TS_at_constant_EOS_density_preserves_PE_tangent():
    inputs = columns(eta=(-.2, .1, .4), bottom=-10., shear=True,
                     temperature=lambda lo, hi: 15. + .1 * .5 * (lo + hi),
                     salinity=lambda lo, hi: 35. + (2e-4 / 7.6e-4) * .1 * .5 * (lo + hi))
    op = BarotropicOperator(*inputs)
    rate = op.rates()
    direct = stock_pe_direction(op.profile, op.footprint.area_m2, rate.hdot, rate.stock_transport, rate.etadot)
    assert abs(direct - rate.energy['gravity_PE_rate_W']) < rate.energy['bound_W']
    assert np.max(abs(rate.stock_transport[:, 3:, :2])) > 1e-3


def test_known_dry_neighbor_retains_wall_pressure_and_has_no_water_or_stock_flux():
    profile, footprint, requests, walls = columns(periodic=False)
    state = profile.state
    arrays = {name: getattr(state, name).copy() for name in
              ['active_layers', 'wet_mask', 'eta', 'bottom', 'band_bottom', 'interfaces', 'h', 'stocks']}
    for value in arrays.values():
        value[1] = 0
    profile = p.reconstruct(replace(state, **arrays))
    op = BarotropicOperator(profile, footprint, requests, walls)
    assert not op.segments
    rate = op.rates()
    np.testing.assert_array_equal(rate.hdot, 0.)
    np.testing.assert_array_equal(rate.stock_transport, 0.)
    assert p.boundary_force_budget(profile, requests, walls)['dry_wall_force_N'][0] != 0.


def test_y_faces_use_the_same_bound_operator_and_complete_contours():
    profile, _, _, _ = columns(eta=(0., .3, -.1, .5), periodic=False, shear=True)
    state = profile.state
    state = replace(state, **{name: getattr(state, name).reshape((2, 2) + getattr(state, name).shape[1:])
                              for name in ['active_layers', 'wet_mask', 'eta', 'bottom', 'band_bottom', 'interfaces', 'h', 'stocks']})
    profile = p.reconstruct(state)
    footprint = p.RectangularFootprint(np.array([[[0., 1., 0., 2.], [1., 2., 0., 2.]],
                                               [[0., 1., 2., 4.], [1., 2., 2., 4.]]]), np.full((2, 2), 2.))
    requests = [p.FaceRequest((i, 0), (i, 1), 2., (1., 0.)) for i in range(2)]
    requests += [p.FaceRequest((0, j), (1, j), 1., (0., 1.)) for j in range(2)]
    walls = [p.OuterWall((i, j), 2., (2. * j - 1., 0.)) for i in range(2) for j in range(2)]
    walls += [p.OuterWall((i, j), 1., (0., 2. * i - 1.)) for i in range(2) for j in range(2)]
    op = BarotropicOperator(profile, footprint, requests, walls)
    assert any(s.normal == (0., 1.) for s in op.segments)
    rate = op.rates()
    assert np.max(abs(rate.pressure_momentum[..., 1])) > 1.
    assert rate.energy['pairing_residual_ratio'] < 1.
    assert op.midpoint_identity()['work_roundoff_ratio'] < 1.


def test_full_P1_PE_oracle_handles_density_direction_and_refuses_invalid_shapes():
    profile, footprint, _, _ = columns(temperature=lambda lo, hi: 15. + .1 * .5 * (lo + hi))
    assert stock_pe_direction(profile, footprint.area_m2, np.zeros_like(profile.state.h),
                             np.zeros_like(profile.state.stocks), np.zeros_like(profile.state.eta)) == 0.
    with pytest.raises(ValueError, match='shape-matched'):
        stock_pe_direction(profile, footprint.area_m2, np.zeros_like(profile.state.h),
                           np.zeros((1, 1, 4)), np.zeros_like(profile.state.eta))
    op = operator()
    rate = op.rates()
    bad = rate.stock_transport.copy()
    bad[..., 0] += .1
    assert abs(stock_pe_direction(op.profile, op.footprint.area_m2, rate.hdot, bad, rate.etadot)
               - rate.energy['gravity_PE_rate_W']) > .01


def test_nearly_uniform_density_cannot_bypass_full_stock_PE_direction_gate():
    profile, footprint, requests, walls = columns(eta=(0., 0., 0.), bottom=-10., shear=True)
    stocks = profile.state.stocks.copy()
    stocks[1, :, 0] += profile.state.h[1] * 1e-10
    stocks[2, :, 0] -= profile.state.h[2] * 1e-10
    profile = p.reconstruct(replace(profile.state, stocks=stocks))
    op = BarotropicOperator(profile, footprint, requests, walls)
    with pytest.raises(UnsupportedGeometry, match='stock_PE_tangent'):
        op.rates()


def test_finite_extreme_velocity_refuses_nonfinite_intermediates_and_ledgers():
    op = operator(eta=(0.,), periodic=False, velocity=(1e180, 0.))
    with pytest.raises(ValueError, match='squared speed'):
        op.rates()
    with pytest.raises(ValueError):
        op.midpoint_identity()


@pytest.mark.parametrize('left,right,dl,dr,expected', [
    (1., 1., 2., -3., -3.), (-1., -1., 2., -3., 2.),
    (0., 2., -1., 3., 0.), (0., -2., -1., 3., -1.),
    (0., 0., 2., 3., 2.), (0., 0., -2., 3., 0.), (1., -1., 2., 3., 0.)])
def test_minmod_one_sided_kinks(left, right, dl, dr, expected):
    assert _minmod_direction(left, right, dl, dr) == expected


def test_full_P1_stock_PE_direction_includes_slope_moment():
    profile, footprint, _, _ = columns(eta=(0.,), periodic=False, bottom=-10.,
                                      temperature=lambda lo, hi: 15. + .2 * .5 * (lo + hi))
    state, eos = profile.state, profile.eos
    direction = np.zeros_like(state.stocks)
    centers = .5 * (state.interfaces[..., :-1] + state.interfaces[..., 1:])
    direction[..., 0] = state.h * .04 * centers
    actual = stock_pe_direction(profile, footprint.area_m2, np.zeros_like(state.h), direction, np.zeros_like(state.eta))
    expected = 2. * eos.gravity * (-eos.rho0 * eos.alpha * .04) * 1000. / 3.
    assert actual == pytest.approx(expected, abs=1e-11)


def test_zero_slope_at_density_bound_has_clipped_direction_not_zero_direction():
    profile, footprint, _, _ = columns(eta=(0.,), slots=2, periodic=False, temperature=45., salinity=0.)
    state, eos = profile.state, profile.eos
    density_rate = np.array([[.01, 3.]])
    stockdot = np.zeros_like(state.stocks)
    stockdot[..., 1] = state.h * density_rate / (eos.rho0 * eos.beta)
    actual = stock_pe_direction(profile, footprint.area_m2, np.zeros_like(state.h), stockdot, np.zeros_like(state.eta))
    # widths=.25,.75; raw ds=-5.98; first bound clips it to -2*.01/.25.
    expected = 2. * eos.gravity * (.01 * .25 * -.125 + 3. * .75 * -.625
                                  + (-.08 * .25**3 - 5.98 * .75**3) / 12.)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_mean_outside_density_bound_within_input_roundoff_has_inactive_room_derivative():
    profile, footprint, _, _ = columns(eta=(0.,), slots=2, periodic=False, temperature=45. + 2e-12, salinity=0.)
    state, eos = profile.state, profile.eos
    stockdot = np.zeros_like(state.stocks)
    stockdot[..., 1] = state.h * np.array([[.01, 3.]]) / (eos.rho0 * eos.beta)
    actual = stock_pe_direction(profile, footprint.area_m2, np.zeros_like(state.h), stockdot, np.zeros_like(state.eta))
    # Strictly negative raw room stays inactive for an infinitesimal direction.
    expected = 2. * eos.gravity * (.01 * .25 * -.125 + 3. * .75 * -.625)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_fixed_physical_affine_density_direction_on_moving_top_cells():
    profile, footprint, _, _ = columns(eta=(.3,), bottom=-10., periodic=False,
                                      temperature=lambda lo, hi: 15. + .2 * .5 * (lo + hi))
    state, eos = profile.state, profile.eos
    etadot = np.array([.04])
    hdot = np.zeros_like(state.h)
    hdot[:, :3] = state.h[:, :3] / (state.eta - state.band_bottom)[:, None] * etadot[:, None]
    center_rate = etadot[:, None] - np.cumsum(hdot, axis=-1) + .5 * hdot
    stockdot = np.zeros_like(state.stocks)
    stockdot[..., 0] = hdot * profile.temperature_mean + state.h * .2 * center_rate
    stockdot[..., 1] = 35. * hdot
    actual = stock_pe_direction(profile, footprint.area_m2, hdot, stockdot, etadot)
    b = -eos.rho0 * eos.alpha * .2
    expected = 2. * eos.gravity * .3 * (eos.rho0 + b * .3) * .04
    assert actual == pytest.approx(expected, abs=1e-10)
