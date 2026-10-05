"""The MMS command must expose operator or convergence failure to CI."""
import runpy

import pytest

import zhenmode.model.verification as mms


@pytest.mark.parametrize('operator_ok,convergence_ok,status', [
    (True, True, 0), (False, True, 1), (True, False, 1),
])
def test_mms_cli_returns_failure_when_either_gate_fails(monkeypatch, operator_ok, convergence_ok, status):
    monkeypatch.setattr(mms, '_mms_run', lambda: {'operator': (0.01, operator_ok)})
    monkeypatch.setattr(mms, '_mms_convergence', lambda: {
        'ddy_coarse': 0.04, 'ddy_fine': 0.01, 'convergence_ratio': 4., 'passes': convergence_ok,
    })
    monkeypatch.setattr('sys.argv', ['zhenmode', 'mms'])
    with pytest.raises(SystemExit) as result:
        runpy.run_module('zhenmode', run_name='__main__')
    assert result.value.code == status
