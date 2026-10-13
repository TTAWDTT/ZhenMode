"""Development definitions keep the historical comparison immutable."""

import pytest

from tests.support.internal_wave import planted
from zhenmode.benchmarks.internal_wave import (
    contract,
    development_contract,
    initial_native,
    validate_contract,
)
from zhenmode.benchmarks.standing_wave import digest
from zhenmode.evaluation.internal_wave import score


@pytest.mark.parametrize('scheme', ('donor_cell', 'muscl_nodal_v1', 'muscl_endpoint_v2', 'muscl_band_v3'))
@pytest.mark.parametrize('integral', ('trapezoidal', 'cubic_uniform_v1'))
def test_new_protocol_is_bound_to_original_physics_and_independent_truth(scheme, integral):
    original = contract('zhenmode')
    candidate = development_contract(scheme=scheme, integral=integral)
    assert candidate['development']['base_contract_sha256'] == digest(original)
    assert 'vertical_tracer_scheme' not in contract('zhenmode')['ocean_options']
    validate_contract(candidate)
    a = planted(candidate)
    assert score(candidate, a)['engineering_screen_pass']
    assert initial_native(candidate, 'ocean-solver')['T'].shape == (8, 32, 16)


def test_development_cannot_change_timestep_threshold_or_base_identity():
    for field in ('dt', 'thresholds', 'development'):
        c = development_contract()
        c[field] = False
        with pytest.raises(ValueError):
            validate_contract(c)


@pytest.mark.parametrize('options,reason', [
    ({'scheme': 'material_point_v4'}, 'material_point_v4'),
    ({'momentum': 'point_primitive_v1'}, 'point_primitive_v1'),
    ({'mean': 'cubic_consistent_v1'}, 'cubic_consistent_v1'),
    ({'scheme': 'material_point_v4', 'arithmetic': 'integrated_reference_v1',
      'mean': 'cubic_consistent_v1', 'integral': 'cubic_uniform_v1'}, 'material_point_v4'),
    ({'momentum': 'point_primitive_v1', 'arithmetic': 'integrated_reference_v1',
      'mean': 'cubic_consistent_v1', 'integral': 'cubic_uniform_v1'}, 'point_primitive_v1'),
])
def test_development_rejects_unconstructible_selector_combinations(options, reason):
    with pytest.raises(ValueError, match=reason):
        development_contract(**options)


@pytest.mark.parametrize('selector,value,reason', [
    ('vertical_tracer_scheme', 'material_point_v4', 'material_point_v4'),
    ('momentum_vertical_velocity_scheme', 'point_primitive_v1', 'point_primitive_v1'),
    ('column_mean_scheme', 'cubic_consistent_v1', 'cubic_consistent_v1'),
])
def test_contract_validation_rejects_matching_but_unconstructible_declarations(selector, value, reason):
    candidate = development_contract()
    candidate['development'][selector] = value
    candidate['ocean_options'][selector] = value
    with pytest.raises(ValueError, match=reason):
        validate_contract(candidate)


@pytest.mark.parametrize('options', [
    {},
    {'mean': 'cubic_consistent_v1', 'integral': 'cubic_uniform_v1'},
    {'momentum': 'point_primitive_v1', 'arithmetic': 'integrated_reference_v1'},
    {'scheme': 'material_point_v4', 'arithmetic': 'integrated_reference_v1'},
    {'problem': 'thermal-wind', 'profile': 'coarse', 'scheme': 'material_point_v4',
     'arithmetic': 'integrated_reference_v1'},
    {'scheme': 'material_point_v4', 'arithmetic': 'integrated_reference_v1',
     'integral': 'cubic_uniform_v1', 'gradient': 'fourth_uniform_v1',
     'tangent_filter': 'gram_fourth_v1'},
    {'scheme': 'material_point_v4', 'arithmetic': 'integrated_reference_v1',
     'gradient': 'fourth_uniform_v1', 'horizontal': 'paired_product_fourth'},
    {'problem': 'thermal-wind', 'profile': 'coarse', 'scheme': 'material_point_v4',
     'arithmetic': 'integrated_reference_v1', 'gradient': 'fourth_uniform_v1',
     'horizontal': 'paired_product_fourth'},
])
def test_accepted_development_selectors_construct_the_actual_solver(options):
    from tests.support.uniform_tracer_channel import column_case

    candidate = development_contract(**options)
    validate_contract(candidate)
    resolved = dict(hydrostatic_integral_scheme='trapezoidal',
                    column_mean_scheme='nodal_capacity') | candidate['ocean_options']
    _, (_, _, _, params, _) = column_case(**resolved)
    assert params.vertical_tracer_scheme == candidate['development']['vertical_tracer_scheme']
    assert params.tracer_transport_scheme == candidate['ocean_options']['tracer_transport_scheme']
