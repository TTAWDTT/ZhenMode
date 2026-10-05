"""Candidate pins/configuration do not confer run or climate qualification."""
import pytest

from zhenmode.baselines.mom6.omip2 import PINS, corrected_configuration, preparation_plan


def test_pins_cover_coupled_driver_and_nested_dependencies():
    assert {'src/MOM6', 'src/SIS2', 'src/FMS2', 'src/coupler', 'src/ice_param'} <= PINS['components'].keys()
    assert len(PINS['nested']) == 2
    assert all(len(commit) == 40 for commit in (PINS['components'] | PINS['nested']).values())
    assert len(PINS['configuration_files']) == 5
    assert PINS['coupler_audit']['drag'] == 'LY2004'


def test_candidate_plan_remains_blocked():
    short = preparation_plan('integration-6h')
    long = preparation_plan('climate-6cycle')
    assert short['physics_sha256'] == long['physics_sha256']
    assert not short['execution_ready'] and not long['climate_qualification']
    assert short['build_plan']['jobs'] == 1
    assert short['mechanism_gaps'] and 'LY2004' in short['mechanism_gaps'][0]


def test_corrections_are_concrete_without_hiding_remaining_inputs():
    source = {'input.nml': "calendar = 'julian',\ndays = 1,\nhours = 0,\n",
              'MOM_saltrestore': 'FLUXCONST = .1667 ! m/day\nADJUST_NET_FRESH_WATER_TO_ZERO = True\nMAX_DELTA_SRESTORE = 5\nSALT_RESTORE_FILE = "old.nc"\n'}
    prepared = corrected_configuration(source, 'integration-6h')
    assert "calendar = 'gregorian'," in prepared['input.nml']
    assert 'hours = 6,' in prepared['input.nml']
    assert 'ADJUST_NET_FRESH_WATER_TO_ZERO = False' in prepared['MOM_saltrestore']
    assert 'old.nc' in prepared['MOM_saltrestore']
    assert '.1667' in source['MOM_saltrestore']  # upstream evidence is not rewritten
    with pytest.raises(ValueError, match='unambiguous'):
        corrected_configuration(source | {'input.nml': source['input.nml'] + 'days = 2,\n'}, 'integration-6h')
    with pytest.raises(ValueError, match='first-cycle'):
        corrected_configuration(source, 'climate-6cycle')
