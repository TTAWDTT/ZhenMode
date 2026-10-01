"""Static-only pressure representation and independent boundary-force gates."""

from dataclasses import replace

import numpy as np
import pytest

from research.experiments.material_top_band import inventory_pressure as p
from research.experiments.material_top_band import pressure_oracle as oracle
from research.experiments.material_top_band import pressure_static_evidence as evidence
from research.experiments.material_top_band.real_geometry import ColumnStocks


def state_from_edges(edge_columns, temperature, salinity=35., velocity=(.02, -.01)):
    count = np.array([len(z) - 1 for z in edge_columns])
    nz = max(count.max(), 2)
    z = np.zeros((len(count), nz + 1))
    h = np.zeros((len(count), nz))
    stocks = np.zeros(h.shape + (4,))
    for i, edges in enumerate(edge_columns):
        z[i, :count[i] + 1] = edges
        z[i, count[i] + 1:] = edges[-1]
        h[i, :count[i]] = -np.diff(edges)
        for k in range(count[i]):
            lower, upper = edges[k + 1], edges[k]
            T = temperature(lower, upper) if callable(temperature) else temperature
            S = salinity(lower, upper) if callable(salinity) else salinity
            stocks[i, k] = h[i, k] * np.array([T, S, 1025. * velocity[0], 1025. * velocity[1]])
    active = np.arange(nz) < count[:, None]
    return ColumnStocks(count, active, z[:, 0].copy(), z[np.arange(len(count)), count].copy(),
                        z[np.arange(len(count)), np.minimum(count, 3)].copy(), z, h, stocks)


def affine_mean(lower, upper):
    return 15. + .2 * .5 * (lower + upper)


def face(left=(0,), right=(1,), normal=(1., 0.), length=2.):
    return p.FaceRequest(left, right, length, normal)


def outer_for_two(length=2.):
    return [p.OuterWall((0,), length, (-1., 0.)), p.OuterWall((1,), length, (1., 0.)),
            p.OuterWall((0,), 1., (0., 1.)), p.OuterWall((0,), 1., (0., -1.)),
            p.OuterWall((1,), 1., (0., 1.)), p.OuterWall((1,), 1., (0., -1.))]


def two_footprint():
    return p.RectangularFootprint(np.array([[0., 1., 0., 2.], [1., 2., 0., 2.]]), np.array([2., 2.]))


def isolated_footprint(height=2.):
    return p.RectangularFootprint(np.array([[0., 1., 0., height]]), np.array([height]))


def isolated_walls(height=2.):
    return [p.OuterWall((0,), height, (-1., 0.)), p.OuterWall((0,), height, (1., 0.)),
            p.OuterWall((0,), 1., (0., -1.)), p.OuterWall((0,), 1., (0., 1.))]


@pytest.mark.parametrize('eta, bottoms', [(0., (-2., -2.)), (-.3, (-2., -3.))])
def test_affine_physical_profile_different_partitions_has_no_pressure_force(eta, bottoms):
    state = state_from_edges([[eta, -1., bottoms[0]], [eta, -.5, -1.5, bottoms[1]]], affine_mean)
    profile = p.reconstruct(state)
    expected_slope = -1025. * 2e-4 * .2
    np.testing.assert_allclose(profile.density_slope[state.wet_mask], expected_slope, rtol=0, atol=1e-14)
    faces = p.pressure_faces(profile, [face()])
    assert max(abs(item.pressure_jump_Pa) for item in faces) < 1e-12
    certificate = p.certify_force_consumption(profile, [face()], outer_for_two(), footprint=two_footprint())
    assert certificate['accepted']


def test_p0_counterexample_exposes_partition_force():
    state = state_from_edges([[0., -1., -2.], [0., -.5, -2.]], affine_mean)
    p0 = p.reconstruct(state, representation='p0')
    jump = p.pressure(p0, (1,), -.5) - p.pressure(p0, (0,), -.5)
    b = -1025. * 2e-4 * .2
    assert jump == pytest.approx(9.81 * b / 8., abs=1e-12)
    assert abs(jump) > .01


def test_uniform_gauge_changes_neither_force_nor_boundary_balance():
    state = state_from_edges([[0., -.5, -1.], [1., 0., -1.]], 15.)
    base = p.reconstruct(state)
    shifted = p.reconstruct(state, external_pressure_Pa=1234.)
    for z in [-.8, -.3]:
        assert p.pressure(shifted, (0,), z) - p.pressure(base, (0,), z) == pytest.approx(1234.)
    f0, f1 = (p.pressure_faces(profile, [face()]) for profile in [base, shifted])
    np.testing.assert_allclose([f.pressure_jump_Pa for f in f0], [f.pressure_jump_Pa for f in f1], atol=1e-12)
    b0, b1 = (p.boundary_force_budget(profile, [face()], outer_for_two()) for profile in [base, shifted])
    np.testing.assert_allclose(b0['physical_boundary_force_N'], b1['physical_boundary_force_N'], atol=1e-10)
    np.testing.assert_allclose(b0['surface_geometry_difference_N'], b1['surface_geometry_difference_N'], atol=1e-10)


def test_free_surface_cap_cannot_be_laundered_as_dry_wall_reaction():
    state = state_from_edges([[0., -.5, -1.], [1., 0., -1.]], 15.)
    profile = p.reconstruct(state)
    faces = p.pressure_faces(profile, [face()])
    boundary = p.boundary_force_budget(profile, [face()], outer_for_two())
    certificate = p.certify_force_consumption(profile, [face()], outer_for_two(), footprint=two_footprint())
    rho_g_length = 1025. * 9.81 * 2.
    np.testing.assert_allclose(certificate['Ctranspose_force_N'], [-rho_g_length, 0.], atol=1e-10)
    np.testing.assert_allclose(boundary['physical_boundary_force_N'], [-1.5 * rho_g_length, 0.], atol=1e-10)
    np.testing.assert_allclose(boundary['surface_geometry_difference_N'], [.5 * rho_g_length, 0.], atol=1e-10)
    assert boundary['dry_wall_force_N'] == [0., 0.]
    assert not certificate['accepted']
    with pytest.raises(p.PressureForceIncompatibility):
        p.require_force_consumption(profile, [face()], outer_for_two(), footprint=two_footprint())
    # A fixed-mass algebraic work identity can hold despite physical rejection.
    trial = p.algebraic_midpoint_work(profile, faces, np.array([2., 2.]))
    assert trial['work_roundoff_ratio'] <= 1
    assert not trial['accepted_state_returned']


def test_staircase_solid_wall_is_integrated_from_its_own_pressure():
    state = state_from_edges([[0., -.5, -1.], [0., -1., -2.]], 15.)
    profile = p.reconstruct(state)
    budget = p.boundary_force_budget(profile, [face()], outer_for_two())
    np.testing.assert_allclose(budget['staircase_solid_force_N'], [1.5 * 1025. * 9.81 * 2., 0.], atol=1e-10)
    np.testing.assert_allclose(budget['physical_boundary_force_N'], [0., 0.], atol=1e-10)
    assert budget['wall_work_J'] == 0.
    assert p.certify_force_consumption(profile, [face()], outer_for_two(), footprint=two_footprint())['accepted']


def test_dry_neighbor_has_pressure_force_reaction_but_zero_q():
    state = state_from_edges([[0., -.5, -1.], [0.]], 15.)
    profile = p.reconstruct(state)
    assert p.pressure_faces(profile, [face()]) == ()
    budget = p.boundary_force_budget(profile, [face()], [])
    np.testing.assert_allclose(budget['dry_wall_force_N'], [-.5 * 1025. * 9.81 * 2., 0.], atol=1e-10)
    np.testing.assert_allclose(budget['solid_reaction_N'], [.5 * 1025. * 9.81 * 2., 0.], atol=1e-10)
    assert budget['wall_work_J'] == 0.


def test_slope_limit_preserves_mean_inventory_and_physical_endpoints():
    state = state_from_edges([[0., -1., -2.]], lambda lo, hi: 44.9 if hi == 0 else 20.)
    before = state.stocks.tobytes()
    profile = p.reconstruct(state)
    for k in range(2):
        center = .5 * (state.interfaces[0, k] + state.interfaces[0, k + 1])
        for z in state.interfaces[0, k:k + 2]:
            T = profile.temperature_mean[0, k] + profile.temperature_slope[0, k] * (z - center)
            assert -5. <= T <= 45. + 1e-13
    assert state.stocks.tobytes() == before
    assert profile.temperature_mean[0, 0] == 44.9
    assert profile.slope_limited_count > 0


@pytest.mark.parametrize('keyword,value', [('Tref', 10.), ('beta', 8e-4), ('rho0', 1000.), ('gravity', 9.8)])
def test_eos_is_bound_to_original_contract(keyword, value):
    state = state_from_edges([[0., -1., -2.]], 15.)
    with pytest.raises(ValueError, match='EOS'):
        p.reconstruct(state, eos=p.EOS(**{keyword: value}))


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_nonfinite_stock_fails_before_pressure(bad):
    state = state_from_edges([[0., -1., -2.]], 15.)
    state.stocks[0, 0, 0] = bad
    with pytest.raises(ValueError, match='finite'):
        p.reconstruct(state)


def test_segment_quadratic_pressure_and_pe_match_independent_seven_gauss_oracle():
    state = state_from_edges([[-.3, -1., -2.], [-.1, -.6, -1.4, -3.]], affine_mean,
                             salinity=lambda lo, hi: 35. + .01 * .5 * (lo + hi))
    state.stocks[1, :, 0] += state.h[1] * .04
    profile = p.reconstruct(state)
    for column in [(0,), (1,)]:
        lo, hi = float(state.bottom[column]), float(state.eta[column])
        assert p.pressure_integral(profile, column, lo, hi) == pytest.approx(oracle.pressure_integral(profile, column, lo, hi), rel=2e-14)
    for item in p.pressure_faces(profile, [face()]):
        jump = (oracle.pressure_integral(profile, item.right, item.lower_z_m, item.upper_z_m, reduced=True)
                - oracle.pressure_integral(profile, item.left, item.lower_z_m, item.upper_z_m, reduced=True)) / (item.upper_z_m - item.lower_z_m)
        assert item.pressure_jump_Pa == pytest.approx(jump, abs=1e-10)
    areas = np.array([2., 3.])
    assert p.potential_energy(profile, areas) == pytest.approx(oracle.potential_energy(profile, areas), rel=2e-14)
    budget, independent = p.boundary_force_budget(profile, [face()], outer_for_two()), oracle.boundary_forces(profile, [face()], outer_for_two())
    np.testing.assert_allclose(budget['physical_boundary_force_N'], independent['physical'], atol=1e-10)
    np.testing.assert_allclose(budget['surface_geometry_difference_N'], independent['cap'], atol=1e-10)


def test_unclosed_footprint_is_rejected_even_when_pressure_jumps_are_zero():
    state = state_from_edges([[0., -1., -2.], [0., -.5, -2.]], affine_mean)
    profile = p.reconstruct(state)
    with pytest.raises(ValueError, match='footprint'):
        p.certify_force_consumption(profile, [face()], [], footprint=two_footprint())


def test_small_algebraic_impulse_work_uses_full_kinetic_roundoff_scale():
    state = state_from_edges([[0., -100., -200.], [0., -100., -200.]], 15., velocity=(.03, .01))
    state.stocks[1, :, 0] += state.h[1] * 1e-10
    profile = p.reconstruct(state)
    report = p.algebraic_midpoint_work(profile, p.pressure_faces(profile, [face()]), np.array([1e10, 1e10]))
    assert report['work_roundoff_ratio'] <= 1


def test_nonlinear_pressure_error_is_separate_from_roundoff_identities():
    report = evidence.nonlinear_truncation_scan()
    errors = report['maximum_pressure_error_Pa']
    assert all(np.isfinite(errors)) and all(value > 1e-8 for value in errors)
    assert errors[0] > errors[1] > errors[2]
    assert report['full_method_order_claimed'] is False


def test_bare_acceptance_receipt_cannot_authorize_force_consumption():
    with pytest.raises(ValueError, match='Profile'):
        p.require_force_consumption({'accepted': True})


def test_forged_profile_coefficients_are_recomputed_from_inventory():
    state = state_from_edges([[0., -.5, -1.], [1., 0., -1.]], 15.)
    original = p.reconstruct(state)
    forged = replace(original, density_mean=np.full_like(state.h, -1025.),
                     density_slope=np.zeros_like(state.h))
    certificate = p.certify_force_consumption(forged, [face()], outer_for_two(), footprint=two_footprint())
    assert not certificate['accepted']
    with pytest.raises(p.PressureForceIncompatibility):
        p.require_force_consumption(forged, [face()], outer_for_two(), footprint=two_footprint())


def test_closed_oblique_footprint_uses_absolute_supplied_length_scale():
    profile = p.reconstruct(state_from_edges([[0., -.5, -1.]], 15.))
    walls = isolated_walls(1e10)
    # Add a roundoff-sized closure defect on a large supplied footprint.
    walls[0] = p.OuterWall((0,), 1e10 + 1e-5, (-1., 0.))
    assert p.certify_force_consumption(profile, [], walls, footprint=isolated_footprint(1e10))['accepted']


def test_representation_pe_difference_keeps_small_signal_beside_deep_background():
    state = state_from_edges([[0., -1., -2., -3., -1e6]],
                             lambda lo, hi: 16. if lo < -2. else affine_mean(lo, hi))
    before, after = p.reconstruct(state, representation='p0'), p.reconstruct(state)
    area = np.array([1e10])
    difference = p.potential_energy_difference(before, after, area)
    expected = 9.81 * 1e10 * sum(float(after.density_slope[0, k]) * state.h[0, k]**3 / 12. for k in range(4))
    assert difference == pytest.approx(expected, rel=1e-14)
    assert difference == pytest.approx(oracle.potential_energy_difference(before, after, area), rel=1e-12)


def test_representation_pe_difference_requires_identical_physical_domains():
    before = p.reconstruct(state_from_edges([[0., -1., -2.]], 15.))
    after = p.reconstruct(state_from_edges([[.1, -1., -2.]], 15.))
    with pytest.raises(ValueError, match='same physical'):
        p.potential_energy_difference(before, after, np.array([1.]))


def test_direct_pe_difference_bound_excludes_unchanged_deep_background():
    ledgers = []
    for bottom in [-10., -1e6]:
        state = state_from_edges([[0., -1., -2., -3., bottom]],
                                 lambda lo, hi: 16. if lo < -2. else affine_mean(lo, hi))
        before, after = p.reconstruct(state, representation='p0'), p.reconstruct(state)
        area = np.array([1e10])
        ledger = p.potential_energy_difference_ledger(before, after, area)
        independent = oracle.potential_energy_difference(before, after, area)
        evidence._check_ratio(ledger['change_J'] - independent,
                              ledger['absolute_contribution_scale_J'], 'direct PE change')
        ledgers.append(ledger)
    assert ledgers[0] == ledgers[1]


def test_wet_profile_cannot_consume_pressure_without_geometry_footprint():
    profile = p.reconstruct(state_from_edges([[0., -.5, -1.]], 15.))
    with pytest.raises(ValueError, match='footprint'):
        p.certify_force_consumption(profile)
    with pytest.raises(ValueError, match='footprint'):
        p.require_force_consumption(profile)


def test_vector_closed_opposite_walls_do_not_prove_complete_footprint():
    profile = p.reconstruct(state_from_edges([[0., -.5, -1.]], 15.))
    walls = [p.OuterWall((0,), 2., (-1., 0.)), p.OuterWall((0,), 2., (1., 0.))]
    with pytest.raises(ValueError, match='footprint'):
        p.certify_force_consumption(profile, [], walls)


def test_empty_all_dry_domain_has_no_wet_scientific_qualification():
    profile = p.reconstruct(state_from_edges([[0.]], 15.))
    assert p.certify_force_consumption(profile)['accepted'] is False
    with pytest.raises(p.PressureForceIncompatibility):
        p.require_force_consumption(profile)


@pytest.mark.parametrize('eta', [0., -.2])
def test_affine_density_from_nonlinear_tracers_is_partition_well_balanced(eta):
    def temperature(lo, hi):
        return 15. + .1 * (lo**2 + lo * hi + hi**2) / 3.

    def salinity(lo, hi):
        return 35. + (2e-4 / 7.6e-4) * (
            .1 * (lo**2 + lo * hi + hi**2) / 3. + .3 * .5 * (lo + hi))
    state = state_from_edges([[eta, -1., -2., -3.], [eta, -.5, -1.5, -3.]], temperature, salinity)
    before = state.stocks.tobytes()
    profile = p.reconstruct(state)
    jump = p.pressure(profile, (1,), -1.) - p.pressure(profile, (0,), -1.)
    assert abs(jump) < 1e-12, f'partition pressure jump: {jump} Pa'
    np.testing.assert_allclose(profile.density_slope[state.wet_mask], .0615, rtol=0, atol=1e-13)
    assert state.stocks.tobytes() == before
    for item in p.pressure_faces(profile, [face()]):
        assert abs(item.pressure_jump_Pa) < 1e-12
    for column in [(0,), (1,)]:
        for depth in [-.4, -1., -2.7]:
            exact = 1025. * 9.81 * eta + 9.81 * .0615 * (eta**2 - depth**2) / 2.
            assert p.pressure(profile, column, depth, reduced=True) == pytest.approx(exact, abs=1e-12)
            assert oracle.pressure(profile, column, depth, reduced=True) == pytest.approx(exact, abs=1e-12)
    assert p.potential_energy(profile, np.array([1., 1.])) == pytest.approx(
        oracle.potential_energy(profile, np.array([1., 1.])), abs=1e-12)
    exact_pe = 2. * 9.81 * (1025. * eta**2 / 2. + .0615 * (eta**3 + 27.) / 3.)
    assert p.potential_energy(profile, np.array([1., 1.])) == pytest.approx(exact_pe, abs=1e-12)


@pytest.mark.parametrize('defect', ['empty', 'missing_pair', 'duplicate', 'normal', 'length', 'column'])
def test_explicit_footprint_rejects_incomplete_duplicate_or_misbound_faces(defect):
    profile = p.reconstruct(state_from_edges([[0., -.5, -1.]], 15.))
    walls = isolated_walls()
    if defect == 'empty':
        walls = []
    elif defect == 'missing_pair':
        walls = walls[:2]
    elif defect == 'duplicate':
        walls += [walls[0]]
    elif defect == 'normal':
        walls[0] = p.OuterWall((0,), 2., (0., 1.))
    elif defect == 'length':
        walls[0] = p.OuterWall((0,), 3., (-1., 0.))
    else:
        walls[0] = p.OuterWall((1,), 2., (-1., 0.))
    with pytest.raises(ValueError):
        p.certify_force_consumption(profile, [], walls, footprint=isolated_footprint())


@pytest.mark.parametrize('defect', ['area', 'overlap', 'partial', 'degenerate'])
def test_footprint_geometry_itself_is_bound_and_nondegenerate(defect):
    profile = p.reconstruct(state_from_edges([[0., -.5, -1.], [0., -.5, -1.]], 15.))
    bounds, areas = two_footprint().bounds_m.copy(), two_footprint().area_m2.copy()
    if defect == 'area':
        areas[0] = 3.
    elif defect == 'overlap':
        bounds[1, :2] -= .5
    elif defect == 'partial':
        bounds[1, 2:] += .5
    else:
        bounds[0, 1] = bounds[0, 0]
    with pytest.raises(ValueError):
        p.certify_force_consumption(profile, [face()], outer_for_two(),
                                    footprint=p.RectangularFootprint(bounds, areas))


def test_known_dry_neighbor_cannot_be_relabelled_an_outer_wall():
    profile = p.reconstruct(state_from_edges([[0., -.5, -1.], [0.]], 15.))
    with pytest.raises(ValueError, match='adjacency'):
        p.certify_force_consumption(profile, [], isolated_walls(), footprint=two_footprint())
    assert p.certify_force_consumption(profile, [face()], outer_for_two(), footprint=two_footprint())['accepted']


def test_complete_isolated_wet_footprint_qualifies():
    profile = p.reconstruct(state_from_edges([[0., -.5, -1.]], 15.))
    certificate = p.certify_force_consumption(profile, [], isolated_walls(), footprint=isolated_footprint())
    assert certificate['accepted'] and certificate['footprint_geometry']['covered_wet_sides'] == 4


def test_periodic_footprint_binds_exact_images_and_has_no_outer_periodic_wall():
    profile = p.reconstruct(state_from_edges([[0., -.5, -1.], [0., -.5, -1.]], 15.))
    footprint = replace(two_footprint(), periodic_extent_m=(2., 0.))
    requests = [face(), p.FaceRequest((0,), (1,), 2., (-1., 0.), (-2., 0.))]
    walls = [wall for wall in outer_for_two() if wall.normal[0] == 0]
    assert p.certify_force_consumption(profile, requests, walls, footprint=footprint)['accepted']
    for invalid in [requests[:1], [requests[0], face(normal=(-1., 0.))], requests + [requests[1]]]:
        with pytest.raises(ValueError):
            p.certify_force_consumption(profile, invalid, walls, footprint=footprint)
    with pytest.raises(ValueError):
        p.certify_force_consumption(profile, requests, outer_for_two(), footprint=footprint)


def test_unsupported_periodic_self_face_is_refused():
    profile = p.reconstruct(state_from_edges([[0., -.5, -1.]], 15.))
    footprint = replace(isolated_footprint(), periodic_extent_m=(1., 0.))
    with pytest.raises(ValueError, match='self'):
        p.certify_force_consumption(profile, [], isolated_walls(), footprint=footprint)


def test_direct_density_endpoints_use_eos_envelope_without_clipping_means():
    state = state_from_edges([[0., -1., -2.]], lambda lo, hi: 44.9 if hi == 0 else -4.9, 0.)
    profile = p.reconstruct(state)
    rho_min, rho_max = -33.415, 15.785
    excursion = .5 * state.h * abs(profile.density_slope)
    assert np.all(profile.density_mean - excursion >= rho_min - 1e-12)
    assert np.all(profile.density_mean + excursion <= rho_max + 1e-12)
    assert profile.density_slope_limited_count > 0
    np.testing.assert_array_equal(profile.temperature_mean, state.stocks[..., 0] / state.h)
    assert not profile.pointwise_density_equals_auxiliary_TS_reconstruction


def test_finite_huge_declared_area_mismatch_cannot_overflow_geometry_tolerance():
    profile = p.reconstruct(state_from_edges([[0., -.5, -1.]], 15.))
    footprint = p.RectangularFootprint(np.array([[0., 1e154, 0., 1e154]]), np.array([1.7e308]))
    walls = [p.OuterWall((0,), 1e154, normal) for normal in [(-1., 0.), (1., 0.), (0., -1.), (0., 1.)]]
    with pytest.raises(ValueError, match='area'):
        p.certify_force_consumption(profile, [], walls, footprint=footprint)


def test_pressure_work_and_pe_cannot_consume_area_outside_bound_footprint():
    profile = p.reconstruct(state_from_edges([[0., -.5, -1.], [0., -.5, -1.]], 15.))
    footprint = two_footprint()
    with pytest.raises(ValueError, match='area'):
        p.algebraic_midpoint_work(profile, p.pressure_faces(profile, [face()]), np.array([1., 2.]), footprint=footprint)
    with pytest.raises(ValueError, match='area'):
        p.potential_energy(profile, np.array([1., 2.]), footprint=footprint)


def test_periodic_three_height_wave_remains_physically_force_blocked():
    profile = p.reconstruct(state_from_edges([[0., -.5, -1.], [1., 0., -1.], [2., 0., -1.]], 15.))
    bounds = np.array([[0., 1., 0., 2.], [1., 2., 0., 2.], [2., 3., 0., 2.]])
    footprint = p.RectangularFootprint(bounds, np.full(3, 2.), (3., 0.))
    requests = [p.FaceRequest((0,), (1,), 2., (1., 0.)), p.FaceRequest((1,), (2,), 2., (1., 0.)),
                p.FaceRequest((2,), (0,), 2., (1., 0.), (3., 0.))]
    walls = [p.OuterWall((i,), 1., normal) for i in range(3) for normal in [(0., -1.), (0., 1.)]]
    certificate = p.certify_force_consumption(profile, requests, walls, footprint=footprint)
    assert certificate['footprint_complete']
    np.testing.assert_allclose(certificate['physical_boundary_force_N'], [0., 0.], atol=1e-12)
    np.testing.assert_allclose(certificate['Ctranspose_force_N'], [-1025. * 9.81 * 2., 0.], atol=1e-10)
    assert not certificate['accepted']
