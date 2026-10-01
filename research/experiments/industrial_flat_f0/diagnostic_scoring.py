"""Isolated dt50 admission; frozen scorer source and metric bodies stay unchanged.

Load a separate module namespace and replace only its contract-admission function.
Every array/metadata check and metric/threshold calculation remains the original.
The ordinary frozen scorer module continues to reject dt50 contracts.
"""
import hashlib
import importlib.util
from pathlib import Path

from diagnostic_contract import validate_time_contract


def isolated_time_scorer(expected_sha256):
    path = Path(__file__).parents[1] / 'standing_wave_v0/score.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('frozen scorer source identity changed')
    spec = importlib.util.spec_from_file_location('isolated_dt50_scorer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = module.contract('coarse')

    def admit_registered_dt50(candidate):
        validate_time_contract(candidate, baseline)
        if candidate['dt'] != 50.:
            raise ValueError('this diagnostic namespace admits dt50 only')

    module.validate_contract = admit_registered_dt50
    return module
