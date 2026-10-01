import importlib.util
from pathlib import Path

import netCDF4
import numpy as np
import pytest

path = Path(__file__).parents[1] / 'research/experiments/industrial_flat_f0/resolved_config.py'
spec = importlib.util.spec_from_file_location('resolved_wave', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_text():
    lines = []
    for key, value in module.BASE_MOM_CONTRACT.items():
        text = ('"' + value + '"') if isinstance(value, str) else str(value)
        lines.append(key + ' = ' + text + ' ! actual resolved test field')
    return '\n'.join(lines)


def test_base_contract_and_dormant_legacy_flag():
    result = module.verify_resolved_settings(
        fixture_text() + '\nUSE_BT_CONT_TYPE = True\nNONLINEAR_BT_CONTINUITY = True', {})
    assert result['nonlinear_flag_applies'] is False
    assert result['qualification_passed'] is False


@pytest.mark.parametrize('change', ['GRID_CONFIG = "mercator"', 'F_0 = 1.e-4',
                                  'DRHO_DS = .8', 'AXIS_UNITS = "degrees"'])
def test_contradictory_resolved_contract_rejected(change):
    with pytest.raises(ValueError, match='contradictory'):
        module.verify_resolved_settings(fixture_text() + '\n' + change, {})


def test_unresolved_setting_not_assumed_default():
    with pytest.raises(ValueError, match='KV'):
        module.verify_resolved_settings(fixture_text(), {'KV': 0.})


def test_unknown_barotropic_contract_not_marked_dormant():
    with pytest.raises(ValueError, match='USE_BT_CONT_TYPE'):
        module.verify_resolved_settings(fixture_text(), {})
    with pytest.raises(ValueError, match='active NONLINEAR'):
        module.verify_resolved_settings(fixture_text() + '\nUSE_BT_CONT_TYPE = False', {})


def test_native_area_and_metre_contract(tmp_path):
    path = tmp_path / 'geometry.nc'
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('y', 8)
        ds.createDimension('x', 64)
        ds.createDimension('yq', 9)
        ds.createDimension('xq', 65)
        for name, dim in [('lonh', 'x'), ('lath', 'y')]:
            v = ds.createVariable(name, 'f8', (dim,))
            v.units = 'm'
            v[:] = np.arange(len(ds.dimensions[dim]))
        for name, value in [('D', 100.), ('wet', 1.), ('f', 0.),
                            ('Ah', 1002269.4248554128 / 64 * 100000 / 8),
                            ('dxT', 1002269.4248554128 / 64), ('dyT', 100000 / 8)]:
            dims = ('yq', 'xq') if name == 'f' else ('y', 'x')
            ds.createVariable(name, 'f8', dims)[:] = value
    assert module.verify_mom_geometry(path)['geometry_contract_passed']
    with netCDF4.Dataset(path, 'a') as ds:
        ds['Ah'][0, 0] *= 2.
    with pytest.raises(ValueError, match='Ah'):
        module.verify_mom_geometry(path)
