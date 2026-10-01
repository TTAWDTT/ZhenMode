"""Resolved fast/main timesteps must both be visible in exported metadata."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'research/experiments/industrial_flat_f0'))
from export_score import mom_substeps_description


@pytest.mark.parametrize(('dtbt', 'dt', 'expected'), [
    (100., 100., 'actual DTBT=100s; DT=100s; native complete-state output'),
    (50., 50., 'actual DTBT=50s; DT=50s; native complete-state output'),
    (25., 50., 'actual DTBT=25s; DT=50s; native complete-state output'),
    (37.5, 50., 'actual DTBT=37.5s; DT=50s; native complete-state output'),
])
def test_mom_substeps_use_both_resolved_values(dtbt, dt, expected):
    assert mom_substeps_description({'DTBT': dtbt, 'DT': dt}) == expected
