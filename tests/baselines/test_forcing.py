"""Shared forcing export values, native coordinates and input ownership."""
from argparse import ArgumentParser

import netCDF4
import numpy as np
import pytest

from ocean_solver.baselines import forcing
from tests.support.grid import all_wet_grid


def arguments(directory, kind, *options):
    parser = ArgumentParser()
    forcing.add_arguments(parser)
    return parser.parse_args(['--kind', kind, '--out', str(directory / 'forcing.nc'),
                              '--bathy', 'explicit-bathymetry.npz', *options])


@pytest.mark.parametrize('kind', ['wind', 'air-temperature', 'sensible-heat'])
def test_export_preserves_native_values_coordinates_and_month_mapping(tmp_path, monkeypatch, kind):
    grid = all_wet_grid(nx=4, ny=3, nz=4)
    spatial = np.arange(12, dtype=float).reshape(4, 3)
    air = np.stack([spatial + month + 15. for month in range(12)])
    temperatures = np.repeat((spatial + 10.)[:, :, None], 4, axis=2)
    calls = []

    def build(config, path, **options):
        assert path == 'explicit-bathymetry.npz'
        assert config.resolution == 0.5 and options['remap'] == 'area'
        return grid

    def wind(grid, month_idx, taper_cells):
        calls.append(month_idx)
        assert taper_cells == 8
        return spatial + month_idx, -spatial - month_idx

    monkeypatch.setattr(forcing, 'make_global_grid', build)
    monkeypatch.setattr(forcing, 'real_wind_forcing', wind)
    monkeypatch.setattr(forcing, 'load_monthly_mean_air_temp', lambda *a, **kw: air)
    monkeypatch.setattr(forcing, 'get_initial_fields', lambda grid: (temperatures, temperatures))
    result = forcing.export_forcing(arguments(tmp_path, kind))
    with netCDF4.Dataset(result['path']) as dataset:
        np.testing.assert_array_equal(dataset['x'][:], grid.lon)
        np.testing.assert_array_equal(dataset['y'][:], grid.lat)
        np.testing.assert_array_equal(dataset['time'][:], np.arange(12) * 30 + 16)
        assert dataset['time'].calendar == 'julian'
        assert dataset['time'].units == 'days since 0001-01-01 00:00:00'
        if kind == 'wind':
            assert calls == list(range(900, 912))
            expected = np.stack([spatial.T + index for index in range(900, 912)])
            np.testing.assert_array_equal(dataset['STRESS_X'][:], expected)
            np.testing.assert_array_equal(dataset['STRESS_Y'][:], -expected)
            assert dataset['STRESS_X'].units == 'N m-2'
        else:
            assert not calls
            variable = 'AIR_TEMP' if kind == 'air-temperature' else 'sensible'
            expected = (air.transpose(0, 2, 1) if kind == 'air-temperature' else
                        np.broadcast_to((np.arange(12) + 5.)[:, None, None] * 80., (12, 3, 4)))
            np.testing.assert_array_equal(dataset[variable][:], expected)
            assert dataset[variable].units == ('degC' if kind == 'air-temperature' else 'W m-2')
        assert all(dataset[name].dimensions == ('time', 'y', 'x') for name in
                   set(dataset.variables) - {'x', 'y', 'time'})


def test_export_rejects_existing_output_before_loading_inputs(tmp_path, monkeypatch):
    out = tmp_path / 'forcing.nc'
    out.write_bytes(b'previous result')
    monkeypatch.setattr(forcing, 'make_global_grid', lambda *a, **kw: pytest.fail('read input'))
    with pytest.raises(FileExistsError):
        forcing.export_forcing(arguments(tmp_path, 'wind'))
    assert out.read_bytes() == b'previous result'


@pytest.mark.parametrize('missing', ['--kind', '--out', '--bathy'])
def test_export_requires_explicit_input_and_output(missing):
    parser = ArgumentParser()
    forcing.add_arguments(parser)
    args = ['--kind', 'wind', '--out', 'forcing.nc', '--bathy', 'bathy.npz']
    index = args.index(missing)
    del args[index:index + 2]
    with pytest.raises(SystemExit) as error:
        parser.parse_args(args)
    assert error.value.code == 2
