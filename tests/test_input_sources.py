"""Adversarial format/identity and deterministic read-change witnesses."""
import hashlib
import json
from io import BytesIO

import numpy as np
import pytest
from netCDF4 import Dataset
from test_input_quality import cli, example

from input_quality import audit_woa_variable, enforce_strict_quality
from input_sources import InputSnapshot, load_climatology_snapshot


def write_inputs(folder):
    raw, grid = example()
    np.savez(folder / 'temp.npz', **raw)
    np.savez(folder / 'salt.npz', **{**raw, 'variable': np.array('salinity'),
                                  'units': np.array('1')})
    np.savez(folder / 'grid.npz', **grid)
    np.savez(folder / 'initial.npz', T=raw['data'].transpose(2, 1, 0),
             S=raw['data'].transpose(2, 1, 0))
    return ['--temperature', str(folder / 'temp.npz'), '--salinity',
            str(folder / 'salt.npz'), '--grid', str(folder / 'grid.npz'),
            '--initial-state', str(folder / 'initial.npz'),
            '--output', str(folder / 'quality.json'), '--strict']


def test_typed_normal_twin_input_passes_strict(tmp_path):
    assert cli().main(write_inputs(tmp_path)) == 0
    report = json.loads((tmp_path / 'quality.json').read_text())
    for label in ('T', 'S'):
        enforce_strict_quality(report['variables'][label])
        assert report['variables'][label]['current_source_identity']['binding'] == (
            'immutable_read_snapshot')


@pytest.mark.parametrize('field,value', [('variable', 'salinity'), ('units', 'unknown'),
                                        ('source_format', 'unknown'),
                                        ('missing_encoding', 'unknown')])
def test_wrong_or_unknown_twin_metadata_rejected(tmp_path, field, value):
    arguments = write_inputs(tmp_path)
    raw, _ = example()
    raw[field] = np.array(value)
    np.savez(tmp_path / 'temp.npz', **raw)
    assert cli().main(arguments) == 2


def test_same_untyped_twin_cannot_supply_both_variables(tmp_path):
    arguments = write_inputs(tmp_path)
    raw, _ = example()
    for key in ('source_format', 'variable', 'units', 'missing_encoding'):
        del raw[key]
    np.savez(tmp_path / 'temp.npz', **raw)
    arguments[3] = arguments[1]
    assert cli().main(arguments) == 2


def test_same_typed_temperature_twin_cannot_supply_salinity(tmp_path):
    arguments = write_inputs(tmp_path)
    arguments[3] = arguments[1]
    assert cli().main(arguments) == 2


@pytest.mark.parametrize('changed', ['temp.npz', 'grid.npz', 'initial.npz'])
def test_cli_rejects_replacement_after_parse(tmp_path, monkeypatch, changed):
    arguments = write_inputs(tmp_path)
    module = cli()
    original_audit = module.audit_woa_variable
    invoked = False

    def replace_after_read(*args, **kwargs):
        nonlocal invoked
        result = original_audit(*args, **kwargs)
        if not invoked:
            invoked = True
            # The audit already consumed parsed arrays. Replace their source now.
            buffer = BytesIO()
            np.savez(buffer, changed=np.array([np.nan]))
            (tmp_path / changed).write_bytes(buffer.getvalue())
        return result

    monkeypatch.setattr(module, 'audit_woa_variable', replace_after_read)
    assert module.main(arguments) == 2
    assert invoked
    assert not (tmp_path / 'quality.json').exists()


def test_cli_detects_change_during_output_write(tmp_path, monkeypatch):
    arguments = write_inputs(tmp_path)
    module = cli()
    original_save = module.np.savez_compressed

    def mutate_after_save(*args, **kwargs):
        original_save(*args, **kwargs)
        (tmp_path / 'temp.npz').write_bytes(b'replaced-after-validation')

    monkeypatch.setattr(module.np, 'savez_compressed', mutate_after_save)
    assert module.main(arguments) == 2


def test_snapshot_hash_stays_bound_to_parsed_bytes_after_replacement(tmp_path):
    write_inputs(tmp_path)
    path = tmp_path / 'temp.npz'
    snapshot = InputSnapshot.read(path)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    raw, _ = example()
    raw['data'][:] = np.nan
    np.savez(path, **raw)
    parsed = load_climatology_snapshot(snapshot, 'T')
    assert np.all(parsed['data'] == 7.)
    assert snapshot.identity()['sha256'] == original_hash
    assert snapshot.identity()['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='changed'):
        snapshot.verify_unchanged()


def netcdf_source(path, *, units='degrees_celsius', fill=True, sentinel=False):
    with Dataset(path, 'w') as dataset:
        for name, values in [('time', [0]), ('depth', [0., 10.]),
                             ('lat', [-1., 0., 1.]), ('lon', [0., 1., 2., 3.])]:
            dataset.createDimension(name, len(values))
            coordinate = dataset.createVariable(name, 'f8', (name,))
            coordinate[:] = values
            if name != 'time':
                coordinate.units = {'depth': 'm', 'lat': 'degrees_north',
                                    'lon': 'degrees_east'}[name]
            if name == 'depth':
                coordinate.positive = 'down'
        variable = dataset.createVariable('t_an', 'f4', ('time', 'depth', 'lat', 'lon'),
                                          fill_value=np.float32(9.96921e36) if fill else False)
        variable.units = units
        variable[:] = np.full(variable.shape, 9.96921e36 if sentinel else 7., dtype=np.float32)


def test_netcdf_declared_finite_fill_is_missing(tmp_path):
    path = tmp_path / 'temp.nc'
    netcdf_source(path, sentinel=True)
    raw = load_climatology_snapshot(InputSnapshot.read(path), 'T')
    _, grid = example()
    report, _ = audit_woa_variable(raw, grid, variable='T')
    assert report['missing_wet_nodes'] == 24
    with pytest.raises(ValueError, match='unsupported'):
        enforce_strict_quality(report)


def test_normal_metadata_netcdf_passes_array_strict(tmp_path):
    path = tmp_path / 'temp.nc'
    netcdf_source(path)
    raw = load_climatology_snapshot(InputSnapshot.read(path), 'T')
    _, grid = example()
    report, _ = audit_woa_variable(raw, grid, variable='T')
    enforce_strict_quality(report)


@pytest.mark.parametrize('options', [{'units': 'kelvin'}, {'fill': False}])
def test_netcdf_unknown_units_or_encoding_rejected(tmp_path, options):
    path = tmp_path / 'temp.nc'
    netcdf_source(path, **options)
    with pytest.raises(ValueError, match='unsupported'):
        load_climatology_snapshot(InputSnapshot.read(path), 'T')


@pytest.mark.parametrize('field,value', [('missing_wet_nodes', 99),
                                        ('coordinate_mapping_supported', False),
                                        ('grid_contract_verified', False),
                                        ('changed_raw_valid_nodes', 1),
                                        ('nonfinite_initial_wet_nodes', 1),
                                        ('records_omitted', 1)])
def test_forged_supported_status_does_not_override_fields(field, value):
    raw, grid = example()
    report, _ = audit_woa_variable(raw, grid, variable='T')
    report[field] = value
    with pytest.raises(ValueError, match='unsupported'):
        enforce_strict_quality(report)


def test_forged_metadata_flags_do_not_override_wrong_role():
    raw, grid = example()
    report, _ = audit_woa_variable(raw, grid, variable='T')
    report['source_metadata']['variable'] = 'salinity'
    with pytest.raises(ValueError, match='unsupported'):
        enforce_strict_quality(report)


def test_npz_cannot_claim_netcdf_decoding(tmp_path):
    raw, _ = example()
    raw.update(source_format='woa-compatible-netcdf.v1', missing_encoding='netcdf_masked',
               fill_value=9.96921e36)
    path = tmp_path / 'spoof.npz'
    np.savez(path, **raw)
    with pytest.raises(ValueError, match='unsupported twin format'):
        load_climatology_snapshot(InputSnapshot.read(path), 'T')


@pytest.mark.parametrize('encoding,fill', [('nan', None), ('fill_value', 1.)])
def test_unexplained_known_woa_marker_is_not_supported(tmp_path, encoding, fill):
    raw, _ = example()
    raw['data'][:] = 9.96921e36
    raw['missing_encoding'] = encoding
    if fill is not None:
        raw['fill_value'] = fill
    path = tmp_path / 'unknown_encoding.npz'
    np.savez(path, **raw)
    with pytest.raises(ValueError, match='unsupported'):
        load_climatology_snapshot(InputSnapshot.read(path), 'T')


@pytest.mark.parametrize('coordinate,attribute,value', [('lon', 'units', 'radians'),
                                                      ('depth', 'units', 'cm'),
                                                      ('depth', 'positive', 'up')])
def test_wrong_coordinate_units_or_direction_rejected(tmp_path, coordinate, attribute, value):
    path = tmp_path / 'temp.nc'
    netcdf_source(path)
    with Dataset(path, 'a') as dataset:
        dataset.variables[coordinate].setncattr(attribute, value)
    with pytest.raises(ValueError, match='unsupported'):
        load_climatology_snapshot(InputSnapshot.read(path), 'T')


def test_packed_netcdf_valid_decoded_value_equal_to_encoded_fill_is_preserved(tmp_path):
    path = tmp_path / 'packed.nc'
    netcdf_source(path)
    with Dataset(path, 'a') as dataset:
        variable = dataset.createVariable('packed_temperature', 'i2',
                                          ('time', 'depth', 'lat', 'lon'), fill_value=0)
        variable.units = 'degrees_celsius'
        variable.scale_factor = .2
        variable.add_offset = -1.
        variable.set_auto_maskandscale(False)
        variable[:] = np.full(variable.shape, 5, dtype=np.int16)
        dataset.renameVariable('t_an', 'unused')
        dataset.renameVariable('packed_temperature', 't_an')
    raw = load_climatology_snapshot(InputSnapshot.read(path), 'T')
    _, grid = example()
    report, masks = audit_woa_variable(raw, grid, variable='T')
    assert report['missing_wet_nodes'] == 0
    assert not masks['raw_missing'].any()
    enforce_strict_quality(report)
