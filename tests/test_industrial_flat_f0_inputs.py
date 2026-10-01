import importlib.util
from pathlib import Path

import netCDF4
import numpy as np
import pytest

path = Path(__file__).parents[1] / 'research/experiments/industrial_flat_f0/prepare_inputs.py'
spec = importlib.util.spec_from_file_location('prepare_flat_inputs', path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_nodal_inventory_and_real_metre_metrics():
    arrays = adapter.native_arrays(0.)
    np.testing.assert_allclose(arrays['h0_m'], [100/6, 100/3, 100/3, 100/6])
    np.testing.assert_allclose(arrays['h0_m'].sum(), 100.)
    np.testing.assert_allclose(-np.diff(arrays['interfaces_m'], axis=-1).sum(axis=-1),
                               100. + arrays['eta_m'])
    np.testing.assert_allclose(arrays['area_m2'].sum(),
                               1002269.4248554128 * 100000.)
    assert arrays['x_m'][-1] > 360.
    assert np.all(arrays['cos_metric'] == 1.)


def test_native_mom_roundtrip_preserves_interfaces_and_units(tmp_path):
    arrays = adapter.native_arrays(0.)
    target = tmp_path / 'initial.nc'
    adapter.write_mom_initial(target, arrays)
    with netCDF4.Dataset(target) as ds:
        assert ds['x'].units == ds['y'].units == 'm'
        expected = arrays['interfaces_m'].copy()
        expected[..., 0] *= np.sinc(1. / 64.)
        np.testing.assert_array_equal(ds['eta'][:].transpose(2, 1, 0), expected)
        np.testing.assert_array_equal(arrays['interfaces_m'][..., 0], arrays['eta_m'])
        assert np.all(ds['PTEMP'][:] == 15.)
        assert np.all(ds['SALT'][:] == 35.)


def test_nonfinite_f0_rejected():
    with pytest.raises(ValueError):
        adapter.native_arrays(float('nan'))


def test_nonzero_f0_not_silently_accepted():
    with pytest.raises(ValueError, match='requires f0=0'):
        adapter.native_arrays(1.e-4)
