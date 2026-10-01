"""Synthetic small reader witnesses, never benchmark timing or dynamics."""
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'research/experiments/industrial_flat_f0'))
from export_score import native_quadrature
from native_mom import read_native_states


def write_state(path, time_s):
    with netCDF4.Dataset(path, 'w') as ds:
        for name, size in [('Time', 1), ('Layer', 4), ('lonh', 64), ('lath', 8), ('lonq', 65), ('latq', 9)]:
            ds.createDimension(name, size)
        for name, values in [('lonh', (np.arange(64) + .5) * 1002269.4248554128 / 64.),
                              ('lath', (np.arange(8) + .5) * 12500.),
                              ('lonq', np.arange(65) * 1002269.4248554128 / 64.),
                              ('latq', np.arange(9) * 12500.)]:
            variable = ds.createVariable(name, 'f8', (name,))
            variable.units = 'm'
            variable[:] = values
        time = ds.createVariable('Time', 'f8', ('Time',))
        time.units = 'days'
        time[:] = time_s / 86400.
        for name, dims, units, value in [
                ('Temp', ('Time', 'Layer', 'lath', 'lonh'), 'degC', 15.),
                ('Salt', ('Time', 'Layer', 'lath', 'lonh'), 'PPT', 35.),
                ('h', ('Time', 'Layer', 'lath', 'lonh'), 'm', 25.),
                ('u', ('Time', 'Layer', 'lath', 'lonq'), 'm s-1', .001),
                ('v', ('Time', 'Layer', 'latq', 'lonh'), 'm s-1', 0.)]:
            variable = ds.createVariable(name, 'f8', dims)
            variable.units = units
            variable[:] = value


def test_full_native_states_preserve_values_and_times(tmp_path):
    (tmp_path / 'RESTART').mkdir()
    write_state(tmp_path / 'native_initial.nc', 0.)
    write_state(tmp_path / 'RESTART/native.nc', 1000.)
    states = read_native_states(tmp_path)
    assert set(states) == {0, 1000}
    assert np.all(states[1000]['u'] == .001)
    assert np.all(states[1000]['v'] == 0.)


def test_only_byte_identical_duplicate_snapshot_is_an_alias(tmp_path):
    (tmp_path / 'RESTART').mkdir()
    write_state(tmp_path / 'native_initial.nc', 0.)
    first = tmp_path / 'RESTART/final.nc'
    alias = tmp_path / 'RESTART/timestamp.nc'
    write_state(first, 1000.)
    alias.write_bytes(first.read_bytes())
    assert len(read_native_states(tmp_path)[1000]['identical_file_aliases']) == 1
    with netCDF4.Dataset(alias, 'a') as ds:
        ds['u'][0, 0, 0, 0] = .002
    with pytest.raises(ValueError, match='conflicting duplicate'):
        read_native_states(tmp_path)


@pytest.mark.parametrize('change', ['units', 'masked', 'time', 'coordinate'])
def test_native_states_reject_wrong_identity(tmp_path, change):
    (tmp_path / 'RESTART').mkdir()
    path = tmp_path / 'native_initial.nc'
    write_state(path, 0.)
    with netCDF4.Dataset(path, 'a') as ds:
        if change == 'units':
            ds['h'].units = 'cm'
        elif change == 'masked':
            ds['v'][0, 0, 1, 1] = netCDF4.default_fillvals['f8']
        elif change == 'time':
            ds['Time'][:] = .5 / 86400.
        else:
            ds['lonq'][0] = 1.
    with pytest.raises((ValueError, AssertionError)):
        read_native_states(tmp_path)


def test_dual_quadrature_wall_half_cells_and_periodic_faces():
    h = np.full((1, 512, 4), 25.)
    area = np.full(512, 1002269.4248554128 / 64 * 12500.)
    faces = dict(x_u=np.zeros(512), y_u=np.zeros(512), x_v=np.zeros(576), y_v=np.zeros(576))
    quadrature = native_quadrature(h, np.zeros(512), np.zeros(512), area, 'cgrid', faces)
    v = quadrature['volume_v'].reshape(1, 64, 9, 4)
    assert np.all(v[:, :, 0] == .5 * v[:, :, 1])
    assert np.all(v[:, :, -1] == .5 * v[:, :, -2])
    assert quadrature['volume_u'].shape == (1, 512, 4)
    with pytest.raises(ValueError, match='actual face coordinates'):
        native_quadrature(h, np.zeros(512), np.zeros(512), area, 'cgrid')
