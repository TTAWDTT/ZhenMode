"""Negative controls for equivalence receipts; missing records cannot pass."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load_tool(filename):
    path = Path(__file__).resolve().parents[1] / 'scripts' / filename
    specification = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_missing_restart_record_is_refused():
    tool = load_tool('capture_driver_modularization.py')
    with pytest.raises(ValueError, match='record names differ'):
        tool.assert_partner_records({'continuous/state': np.arange(2)})


@pytest.mark.parametrize('left,right', [(np.array([0.]), np.array([-0.])),
    (np.array([1.]), np.array([1.], dtype='float32')),
    (np.asarray('{"counters":{"accepted_steps":6}}'),
     np.asarray('{"counters":{"accepted_steps":5}}'))])
def test_numeric_or_checkpoint_semantic_difference_is_refused(left, right):
    tool = load_tool('capture_driver_modularization.py')
    with pytest.raises(ValueError, match='record differs'):
        tool.assert_partner_records({'continuous/record': left, 'resumed/record': right})


def test_provenance_normalization_retains_checksum_and_relative_path(tmp_path):
    tool = load_tool('compare_modularization_captures.py')
    source = tmp_path / 'source' / 'src'
    reference = {'selected_files': {'wind': {'path': str(source.parent / 'data' / 'wind.npz'),
                                            'sha256': 'original'}}}
    original = tool.normalized_provenance(np.asarray(json.dumps(reference)), source)
    reference['selected_files']['wind']['sha256'] = 'changed'
    changed = tool.normalized_provenance(np.asarray(json.dumps(reference)), source)
    assert original.tobytes() != changed.tobytes()
    reference['selected_files']['wind']['sha256'] = 'original'
    reference['selected_files']['wind']['path'] = str(source.parent / 'data' / 'other.npz')
    changed = tool.normalized_provenance(np.asarray(json.dumps(reference)), source)
    assert original.tobytes() != changed.tobytes()
