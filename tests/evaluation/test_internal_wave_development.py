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


@pytest.mark.parametrize('scheme', ('donor_cell', 'muscl_nodal_v1'))
def test_new_protocol_is_bound_to_original_physics_and_independent_truth(scheme):
    original = contract('zhenmode')
    candidate = development_contract(scheme=scheme)
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
