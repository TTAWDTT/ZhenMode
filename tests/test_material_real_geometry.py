"""Static inventory and shared-face contracts; never advance a real state."""

from dataclasses import replace

import numpy as np
import pytest

from research.experiments.material_top_band import accepted_geometry_audit as archive_audit
from research.experiments.material_top_band import real_geometry as g


def packet(counts=(11, 12, 11, 12, 0)):
    depth = np.array([0., 5., 15., 30., 50., 75., 100., 150., 200., 300.,
                      500., 1000., 2000., 4000.])
    edges = np.r_[0., .5 * (depth[:-1] + depth[1:]), depth[-1]]
    counts = np.asarray(counts)
    mask = (np.arange(depth.size) < counts[:, None]).astype(float)
    eta = np.where(counts > 0, -2.49 + .15 * np.arange(counts.size), 0.)
    values = np.empty(mask.shape + (4,))
    values[..., 0] = 15. - .25 * np.arange(depth.size)
    values[..., 1] = 35. + .01 * np.arange(depth.size)
    values[..., 2] = .1 / (1. + np.arange(depth.size))
    values[..., 3] = -.02 * np.cos(np.arange(depth.size))
    values[mask == 0] = [999., -999., 999., -999.]
    return dict(depth=depth, reference_weights=np.diff(edges), wet_mask=mask,
                eta=eta, values=values, column_geometry='nodal_dual_v1')


def test_full_fourteen_level_inventory_and_deep_bytes_preserved():
    inputs = packet()
    before = {key: value.tobytes() for key, value in inputs.items() if isinstance(value, np.ndarray)}
    bridge = g.bridge_nodal_dual(**inputs)
    original, candidate = bridge.original, bridge.candidate
    assert candidate.h.shape == (5, 14)
    np.testing.assert_array_equal(candidate.active_layers, [11, 12, 11, 12, 0])
    np.testing.assert_array_equal(candidate.h[:, 3:], original.h[:, 3:])
    np.testing.assert_array_equal(candidate.stocks[:, 3:], original.stocks[:, 3:])
    np.testing.assert_allclose(candidate.stocks.sum(-2), original.stocks.sum(-2), rtol=2e-15, atol=1e-10)
    for key, snapshot in before.items():
        assert inputs[key].tobytes() == snapshot
    assert bridge.contract == 'nodal_dual_stock_mean_bridge_v1'
    assert not bridge.original_physical_profile_recovered


def test_reference_momentum_is_preserved_without_actual_mass_substitution():
    inputs = packet()
    bridge = g.bridge_nodal_dual(**inputs)
    expected = 1025. * inputs['reference_weights'][0] * inputs['values'][0, 0, 2]
    assert bridge.original.stocks[0, 0, 2] == expected
    wrong = 1025. * bridge.original.h[0, 0] * inputs['values'][0, 0, 2]
    assert abs(expected - wrong) > 100.
    assert bridge.reference_to_actual_kinetic_J_m2[0] > 1000.
    assert bridge.remap_kinetic_change_J_m2[0] < 0.


def test_top_fractions_and_fixed_deep_bottom_per_column():
    bridge = g.bridge_nodal_dual(**packet())
    state = bridge.candidate
    fraction = np.array([1., 3., 5.]) / 9.
    np.testing.assert_allclose(state.h[:4, :3], (state.eta[:4] + 22.5)[:, None] * fraction,
                               rtol=0, atol=4e-15)
    np.testing.assert_array_equal(state.band_bottom, [-22.5, -22.5, -22.5, -22.5, 0.])
    np.testing.assert_array_equal(state.bottom, [-750., -1500., -750., -1500., 0.])
    assert np.all(state.h[state.wet_mask] > 0)
    np.testing.assert_array_equal(state.h[~state.wet_mask], 0.)
    np.testing.assert_array_equal(state.stocks[~state.wet_mask], 0.)
    np.testing.assert_allclose(-np.diff(state.interfaces, axis=-1), state.h, atol=4e-14)


@pytest.mark.parametrize('count', [0, 1, 2, 3, 6, 14])
def test_shallow_and_dry_prefixes_have_their_own_band_and_bottom(count):
    inputs = packet((count,))
    bridge = g.bridge_nodal_dual(**inputs)
    edges = np.r_[0., np.cumsum(inputs['reference_weights'])]
    assert bridge.candidate.bottom[0] == -edges[count]
    assert bridge.candidate.band_bottom[0] == -edges[min(count, 3)]
    assert bridge.candidate.active_layers[0] == count
    np.testing.assert_allclose(bridge.candidate.stocks.sum(-2), bridge.original.stocks.sum(-2), rtol=2e-15, atol=1e-10)


def test_negative_eta_keeps_material_node_zero_but_refuses_physical_sample_claim():
    bridge = g.bridge_nodal_dual(**packet())
    assert bridge.node0_above_surface[0]
    assert bridge.original.h[0, 0] > 0
    assert bridge.original.stocks[0, 0, 0] > 0
    with pytest.raises(ValueError, match='surface boundary|physical sample'):
        g.require_original_profile(bridge, (0,), bridge.candidate.eta[0], -2.5)
    g.require_original_profile(bridge, (0,), -10., -22.5)


def test_unsampled_bottom_interval_cannot_be_invented():
    bridge = g.bridge_nodal_dual(**packet())
    assert bridge.unsampled_bottom_m[0] == 250.
    with pytest.raises(ValueError, match='bottom profile|physical sample'):
        g.require_original_profile(bridge, (0,), -500., -750.)


def test_dry_west_face_has_exact_zero_q_and_no_segment():
    bridge = g.bridge_nodal_dual(**packet())
    face = g.shared_face(bridge, (0,), (4,), length=1000.)
    assert face.total_Q_m3_s == 0.
    assert face.segments == ()
    assert face.common_upper_z_m is None
    assert face.common_lower_z_m is None


def test_staircase_face_uses_max_bottom_and_min_surface():
    bridge = g.bridge_nodal_dual(**packet())
    face = g.shared_face(bridge, (0,), (1,), length=1000.)
    assert face.common_lower_z_m == -750.
    assert face.common_upper_z_m == min(bridge.candidate.eta[:2])
    assert abs(sum(segment.upper_z_m - segment.lower_z_m for segment in face.segments)
               - (face.common_upper_z_m - face.common_lower_z_m)) < 1e-12
    assert all(segment.left_layer < 11 and segment.right_layer < 12 for segment in face.segments)
    assert any(not segment.original_profile_supported for segment in face.segments)
    assert face.total_Q_m3_s != 0.


def test_face_p0_stock_mean_q_constant_velocity_exact_integral():
    inputs = packet((11, 12))
    h = inputs['wet_mask'] * inputs['reference_weights']
    h[:, 0] += inputs['eta']
    inputs['values'][..., 2] = .02 * h / inputs['reference_weights']
    inputs['values'][..., 3] = 0.
    bridge = g.bridge_nodal_dual(**inputs)
    face = g.shared_face(bridge, (0,), (1,), length=321.)
    expected = 321. * (min(inputs['eta']) + 750.) * .02
    assert face.total_Q_m3_s == pytest.approx(expected, rel=2e-15)
    reverse = g.shared_face(bridge, (1,), (0,), length=321., normal=(-1., 0.))
    assert reverse.total_Q_m3_s == pytest.approx(-face.total_Q_m3_s, rel=2e-15)


@pytest.mark.parametrize('mutation, message', [
    ('geometry', 'nodal_dual'), ('weight', 'depth-derived'), ('mask', 'prefix'),
    ('nonbinary', 'binary'), ('negative_h', 'positive'), ('dry_eta', 'dry'),
    ('nonfinite', 'finite'), ('node_order', 'ordered'), ('shape', 'shape'),
])
def test_malformed_geometry_rejected_before_inventory_bridge(mutation, message):
    inputs = packet()
    if mutation == 'geometry':
        inputs['column_geometry'] = 'raw_ETOPO_partial_cells'
    elif mutation == 'weight':
        inputs['reference_weights'][0] += .1
    elif mutation == 'mask':
        inputs['wet_mask'][0, 2] = 0.
    elif mutation == 'nonbinary':
        inputs['wet_mask'][0, 2] = .5
    elif mutation == 'negative_h':
        inputs['eta'][0] = -2.5
    elif mutation == 'dry_eta':
        inputs['eta'][4] = .1
    elif mutation == 'nonfinite':
        inputs['values'][0, 0, 0] = np.nan
    elif mutation == 'node_order':
        inputs['depth'][1] = -5.
    elif mutation == 'shape':
        inputs['values'] = inputs['values'][..., :3]
    with pytest.raises(ValueError, match=message):
        g.bridge_nodal_dual(**inputs)


def test_p0_remap_requires_actual_overlap_not_just_column_stock_match():
    inputs = packet((6,))
    bridge = g.bridge_nodal_dual(**inputs)
    old, new = bridge.original, bridge.candidate
    for target in range(3):
        expected = np.zeros(4)
        for source in range(3):
            overlap = max(0., min(old.interfaces[0, source], new.interfaces[0, target])
                          - max(old.interfaces[0, source + 1], new.interfaces[0, target + 1]))
            expected += overlap * old.stocks[0, source] / old.h[0, source]
        np.testing.assert_allclose(new.stocks[0, target], expected, rtol=2e-15, atol=1e-12)
    assert np.max(abs(new.stocks[:, :3] - old.stocks[:, :3])) > 1.


@pytest.mark.parametrize('field', ['depth', 'reference_weights', 'wet_mask', 'eta', 'values'])
def test_masked_inputs_are_not_silently_unwrapped(field):
    inputs = packet()
    inputs[field] = np.ma.array(inputs[field], mask=np.zeros_like(inputs[field], dtype=bool))
    inputs[field].mask.flat[0] = True
    with pytest.raises(ValueError, match='masked'):
        g.bridge_nodal_dual(**inputs)


def test_two_node_column_default_band_covers_only_available_layers():
    inputs = packet((2, 1, 0))
    inputs['depth'] = np.array([0., 5.])
    inputs['reference_weights'] = np.array([2.5, 2.5])
    inputs['wet_mask'] = inputs['wet_mask'][:, :2]
    inputs['values'] = inputs['values'][:, :2]
    bridge = g.bridge_nodal_dual(**inputs)
    np.testing.assert_array_equal(bridge.candidate.active_layers, [2, 1, 0])
    np.testing.assert_allclose(bridge.candidate.stocks.sum(-2), bridge.original.stocks.sum(-2), rtol=2e-15, atol=1e-10)


def test_boolean_binary_mask_is_supported():
    inputs = packet()
    inputs['wet_mask'] = inputs['wet_mask'].astype(bool)
    bridge = g.bridge_nodal_dual(**inputs)
    np.testing.assert_array_equal(bridge.candidate.active_layers, [11, 12, 11, 12, 0])


def test_unused_finite_dry_sentinels_do_not_enter_inventory_or_kinetic():
    inputs = packet()
    control = g.bridge_nodal_dual(**inputs)
    inputs['values'][inputs['wet_mask'] == 0] = 1e200
    observed = g.bridge_nodal_dual(**inputs)
    np.testing.assert_array_equal(observed.candidate.stocks, control.candidate.stocks)
    np.testing.assert_array_equal(observed.reference_kinetic_J_m2, control.reference_kinetic_J_m2)


def test_masked_face_normal_cannot_erase_support_mask():
    bridge = g.bridge_nodal_dual(**packet())
    with pytest.raises(ValueError, match='masked'):
        g.shared_face(bridge, (0,), (1,), length=1000., normal=np.ma.array([1., 0.], mask=[True, False]))


def test_unverified_archive_rejected_before_numpy_or_bridge(tmp_path, monkeypatch):
    accepted, metadata = tmp_path / 'accepted', tmp_path / 'metadata'
    accepted.mkdir()
    metadata.mkdir()
    for role, (filename, _, _) in archive_audit.IDENTITIES.items():
        directory = accepted if role in ['accepted', 'report', 'source_archive'] else metadata
        (directory / filename).write_bytes(b'not the accepted archive')

    def forbidden(*args, **kwargs):
        pytest.fail('unverified private input must not reach NumPy or inventory bridge')

    monkeypatch.setattr(archive_audit, '_load', forbidden)
    monkeypatch.setattr(archive_audit, 'bridge_nodal_dual', forbidden)
    with pytest.raises(ValueError, match='identity mismatch'):
        archive_audit.audit_archive(accepted, metadata, tmp_path)


def test_unrepresentable_integer_source_inventory_is_not_rounded():
    inputs = packet((14,))
    inputs['values'] = np.full((1, 14, 4), 2**53 + 1, dtype=np.int64)
    with pytest.raises(ValueError, match='representable'):
        g.bridge_nodal_dual(**inputs)


@pytest.mark.parametrize('field', ['water', 'stock', 'eta'])
def test_independent_column_ledger_rejects_nonfinite_candidate(field):
    inputs = packet()
    bridge = g.bridge_nodal_dual(**inputs)
    state = bridge.candidate
    h, stocks, eta = state.h.copy(), state.stocks.copy(), state.eta.copy()
    if field == 'water':
        h[0, 0] = np.nan
    elif field == 'stock':
        stocks[0, 0, 0] = np.nan
    else:
        eta[0] = np.nan
    corrupted = replace(bridge, candidate=replace(state, h=h, stocks=stocks, eta=eta))
    with pytest.raises(ValueError, match='finite'):
        archive_audit._column_check(corrupted, (0,), inputs['reference_weights'], inputs['values'], np.ones(5))


def test_all_expected_original_profile_refusals_are_part_of_acceptance(monkeypatch):
    bridge = g.bridge_nodal_dual(**packet())
    selected = [('column_' + str(i), (i,)) for i in range(4)]
    report = archive_audit._profile_rejections(bridge, selected)
    assert all(set(value) == {'surface', 'bottom'} for value in report.values())
    monkeypatch.setattr(archive_audit, 'require_original_profile', lambda *args: None)
    with pytest.raises(ValueError, match='both expected'):
        archive_audit._profile_rejections(bridge, selected)


def test_unrelated_error_cannot_be_reported_as_physical_support_refusal(monkeypatch):
    bridge = g.bridge_nodal_dual(**packet())

    def malformed(*args):
        raise ValueError('unrelated error')

    monkeypatch.setattr(archive_audit, 'require_original_profile', malformed)
    with pytest.raises(ValueError, match='not established'):
        archive_audit._profile_rejections(bridge, [('center', (0,))])


@pytest.mark.parametrize('density', [None, True, '1025', 1000., float('nan'), 10**400])
def test_archived_density_convention_mismatch_is_refused(density):
    with pytest.raises(ValueError, match='rho0'):
        archive_audit._rho0({'rho0': density})


def test_density_fallback_is_bound_to_verified_historical_source():
    assert archive_audit._rho0({}) == 1025.
    assert archive_audit._rho0({'rho0': 1025.}) == 1025.
