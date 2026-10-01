"""Contract tampering and count checks without running a solver."""
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/experiments/industrial_flat_f0'))
sys.path.insert(0, str(ROOT / 'research/experiments/standing_wave_v0'))
import score as frozen_score
from diagnostic_contract import time_refinement, validate_time_contract
from diagnostic_scoring import isolated_time_scorer
from prepare_mom import settings
from score import contract


def test_refinement_preserves_thresholds_and_baseline():
    baseline = contract('coarse')
    refined = time_refinement(baseline)
    assert baseline == contract('coarse')
    assert refined['thresholds'] == baseline['thresholds']
    assert refined['nx'] == 64
    assert refined['period_s']/refined['dt'] == 640
    validate_time_contract(refined, baseline)


@pytest.mark.parametrize('field', ['thresholds', 'ocean_options', 'mom_time_options'])
def test_unregistered_changes_rejected(field):
    baseline = contract('coarse')
    refined = time_refinement(baseline)
    refined[field]['unregistered_change'] = 1
    with pytest.raises(ValueError, match='more than'):
        validate_time_contract(refined, baseline)


def test_mom_changes_only_four_clocks():
    baseline, refined = settings(False), settings(False, 50.)
    changed = {key for key in baseline if baseline[key] != refined[key]}
    assert changed == {'DT', 'DT_THERM', 'DT_FORCING', 'DTBT'}
    assert refined['DAYMAX'] == baseline['DAYMAX']


def test_unregistered_dt_rejected():
    with pytest.raises(ValueError):
        time_refinement(contract('coarse'), 25.)


def test_diagnostic_admission_does_not_modify_frozen_scorer():
    identity = hashlib.sha256(Path(frozen_score.__file__).read_bytes()).hexdigest()
    diagnostic = isolated_time_scorer(identity)
    refined = time_refinement(contract('coarse'))
    diagnostic.validate_contract(refined)
    with pytest.raises(ValueError, match='frozen'):
        frozen_score.validate_contract(refined)
    with pytest.raises(ValueError):
        diagnostic.validate(refined, {})
    refined['thresholds']['energy'] = .03
    with pytest.raises(ValueError, match='more than'):
        diagnostic.validate_contract(refined)


def test_diagnostic_scorer_identity_change_rejected():
    with pytest.raises(ValueError, match='identity'):
        isolated_time_scorer('0'*64)
