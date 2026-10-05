"""Small manufactured CF files: independent ramp/interval integrals and rejection cases."""
import json
from datetime import datetime

import netCDF4
import numpy as np
import pytest

from zhenmode.execution.benchmark import main
from zhenmode.model.inputs.forcing.jra55 import TIME_UNITS, JRA55Forcing
from zhenmode.provenance.sources import sha256_file

# Explicit fixture specification; deliberately does not import the reader's FIELDS.
WEATHER = [
    ('wind_u', 'uas', 'm s-1', 6., 'point', 10800, 10),
    ('wind_v', 'vas', 'm s-1', 2., 'point', 10800, 10),
    ('temperature_k', 'tas', 'K', 290., 'point', 10800, 10),
    ('specific_humidity', 'huss', '1', .005, 'point', 10800, 10),
    ('pressure_pa', 'psl', 'Pa', 101325., 'point', 10800, None),
    ('shortwave_down', 'rsds', 'W m-2', 100., 'mean', 10800, None),
    ('longwave_down', 'rlds', 'W m-2', 320., 'mean', 10800, None),
    ('rain', 'prra', 'kg m-2 s-1', 1e-5, 'mean', 10800, None),
    ('snow', 'prsn', 'kg m-2 s-1', 0., 'mean', 10800, None),
    ('runoff', 'friver', 'kg m-2 s-1', 1e-5, 'mean', 86400, None),
    ('calving', 'licalvf', 'kg m-2 s-1', 0., 'mean', 86400, None),
]


@pytest.fixture
def forcing_files(tmp_path):
    entries = []
    for name, variable, units, value, kind, cadence, height in WEATHER:
        path = tmp_path / f'{name}.nc'
        with netCDF4.Dataset(path, 'w') as ds:
            ds.source_id = 'MRI-JRA55-do-1-4-0'
            ds.data_kind = 'manufactured'
            for dim, count in [('time', 3 if kind == 'point' else 2), ('lat', 2), ('lon', 2), ('bnds', 2)]:
                ds.createDimension(dim, count)
            for axis, values, axis_units in [('lon', [0., 1.], 'degrees_east'),
                                             ('lat', [-1., 1.], 'degrees_north')]:
                c = ds.createVariable(axis, 'f8', (axis,))
                c.units = axis_units
                c[:] = values
            t = ds.createVariable('time', 'f8', ('time',))
            t.units, t.calendar = TIME_UNITS, 'proleptic_gregorian'
            if kind == 'mean':
                t.bounds = 'time_bounds'
                b = ds.createVariable('time_bounds', 'f8', ('time', 'bnds'))
                b[:] = [[0, cadence], [cadence, 2 * cadence]]
                t[:] = [cadence / 2, 1.5 * cadence]
            else:
                t[:] = [0, cadence, 2 * cadence]
            if height:
                h = ds.createVariable('height', 'f8')
                h.units = 'm'
                h[...] = height
            v = ds.createVariable(variable, 'f8', ('time', 'lat', 'lon'))
            v.units, v.cell_methods = units, f'time: {kind}'
            v[:] = value
            if name == 'temperature_k':
                v[:] = np.array([290, 293, 296])[:, None, None]
            if name == 'shortwave_down':
                v[:] = np.array([100, 400])[:, None, None]
        entries.append(dict(field=name, path=path.name, sha256=sha256_file(path),
                            bytes=path.stat().st_size, source_url='urn:manufactured:unit-test',
                            license='project-test-fixture'))
    manifest = tmp_path / 'forcing.json'
    manifest.write_text(json.dumps(dict(schema_version=1, product='JRA55-do', version='1.4.0',
                                       data_kind='manufactured', files=entries)))
    return manifest


def reader(path, **options):
    return JRA55Forcing(path, **(dict(lon=[0., 1.], lat=[-1., 1.], wet_mask=[[1, 0], [1, 1]],
                                    start_seconds=0, end_seconds=21600) | options))


def modify(manifest, field, operation, *, rehash=True):
    document = json.loads(manifest.read_text())
    entry = next(e for e in document['files'] if e['field'] == field)
    path = manifest.parent / entry['path']
    with netCDF4.Dataset(path, 'a') as ds:
        operation(ds)
    if rehash:
        entry['sha256'], entry['bytes'] = sha256_file(path), path.stat().st_size
        manifest.write_text(json.dumps(document))


def test_interpolation_and_flux_overlap_are_distinct(forcing_files):
    source = reader(forcing_files)
    air = source.sample(3600, interval_end_seconds=14400)
    # 1/3 through the 3h weather ramp; 2h first flux + 1h second flux.
    np.testing.assert_array_equal(air.temperature_k, [[291, 0], [291, 291]])
    np.testing.assert_array_equal(air.shortwave_down, [[200, 0], [200, 200]])
    np.testing.assert_array_equal(air.runoff, [[1e-5, 0], [1e-5, 1e-5]])
    assert source.data_kind == 'manufactured'


@pytest.mark.parametrize('field,operation,error', [
    ('temperature_k', lambda ds: ds['tas'].setncattr('units', 'C'), 'units'),
    ('wind_u', lambda ds: ds['height'].assignValue(2), '10m'),
    ('temperature_k', lambda ds: ds['height'].assignValue(2), '10m'),
    ('specific_humidity', lambda ds: ds['height'].assignValue(2), '10m'),
    ('wind_u', lambda ds: ds['lon'].__setitem__(slice(None), [0, 2]), 'grid mismatch'),
    ('wind_u', lambda ds: ds['time'].setncattr('calendar', '360_day'), 'Gregorian'),
    ('wind_u', lambda ds: ds['time'].__setitem__(1, 11000), 'cadence'),
    ('wind_u', lambda ds: ds['time'].__setitem__(1, 0), 'duplicate'),
    ('shortwave_down', lambda ds: ds['rsds'].setncattr('cell_methods', 'time: point'), 'time: mean'),
    ('shortwave_down', lambda ds: ds['time_bounds'].__setitem__((1, 0), 10000), 'bounds'),
    ('wind_u', lambda ds: ds.setncattr('source_id', 'MRI-JRA55-do-1-5-0'), 'version'),
])
def test_reject_metadata(forcing_files, field, operation, error):
    modify(forcing_files, field, operation)
    with pytest.raises(ValueError, match=error):
        reader(forcing_files)


@pytest.mark.parametrize('field,var,value,error', [
    ('wind_u', 'uas', np.nan, 'nonfinite'),
    ('specific_humidity', 'huss', 50., 'humidity'),
    ('pressure_pa', 'psl', 1013., 'pressure'),
    ('temperature_k', 'tas', 17., 'Kelvin'),
    ('rain', 'prra', -1., 'negative'),
])
def test_reject_wet_values(forcing_files, field, var, value, error):
    modify(forcing_files, field, lambda ds: ds[var].__setitem__(slice(None), value))
    with pytest.raises(ValueError, match=error):
        reader(forcing_files).sample(0, interval_end_seconds=3600)


def test_identity_missing_files_and_window(forcing_files):
    source = reader(forcing_files)
    with pytest.raises(ValueError, match='outside'):
        source.sample(20000, interval_end_seconds=22000)
    with pytest.raises(ValueError, match='not covered'):
        reader(forcing_files, end_seconds=21601)
    modify(forcing_files, 'wind_u', lambda ds: ds['uas'].__setitem__(0, 9.), rehash=False)
    with pytest.raises(ValueError, match='changed'):
        source.sample(0, interval_end_seconds=3600)
    with pytest.raises(ValueError, match='identity'):
        reader(forcing_files)
    (forcing_files.parent / 'wind_u.nc').unlink()
    with pytest.raises(FileNotFoundError, match='missing forcing file'):
        reader(forcing_files)


@pytest.mark.parametrize('mask', [[[1, np.nan], [1, 1]], [[1, .5], [1, 1]], [[0, 0], [0, 0]]])
def test_reject_invalid_masks(forcing_files, mask):
    with pytest.raises(ValueError, match='mask|grid'):
        reader(forcing_files, wet_mask=mask)


@pytest.mark.parametrize('date', ['1958-12-31T21:00:00', '2000-02-28T21:00:00'])
def test_physical_seconds_cross_year_and_leap_day(forcing_files, date):
    for field, *_ in WEATHER:
        modify(forcing_files, field, lambda ds: ds['time'].setncattr('units', f'seconds since {date}'))
    shift = (datetime.fromisoformat(date) - datetime(1970, 1, 1)).total_seconds()
    source = reader(forcing_files, start_seconds=shift, end_seconds=shift + 21600)
    air = source.sample(shift + 3600, interval_end_seconds=shift + 14400)
    assert air.temperature_k[0, 0] == pytest.approx(291)
    assert air.shortwave_down[0, 0] == pytest.approx(200)


def test_installed_forcing_preflight_is_explicit_about_scope(forcing_files, capsys):
    grid = forcing_files.parent / 'grid.npz'
    np.savez(grid, lon=[0., 1.], lat=[-1., 1.], wet_mask=[[1, 0], [1, 1]])
    assert main(['check-forcing', '--manifest', str(forcing_files), '--grid', str(grid),
                 '--start', '1970-01-01T00:00:00', '--end', '1970-01-01T06:00:00']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['data_kind'] == 'manufactured'
    assert not report['all_records_values_scanned'] and not report['execution_ready']
