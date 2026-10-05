"""Adversarial format/identity and deterministic read-change witnesses."""

import base64
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from io import BytesIO

import numpy as np
import pytest
from netCDF4 import Dataset

from tests.support.data.input_quality import cli, example
from tests.support.data.input_sources import netcdf_source, write_inputs
from tests.support.paths import REPOSITORY_ROOT
from zhenmode.model.inputs.quality import audit_woa_variable, enforce_strict_quality
from zhenmode.model.inputs.sources import InputSnapshot, load_climatology_snapshot


def test_npz_readers_and_solver_import_without_netcdf4(tmp_path):
    bathy = tmp_path / 'relief.nc'
    np.savez(str(bathy) + '.npz', z=np.full((20, 40), -3000, dtype=np.int16),
             lon=np.arange(40) * .1, lat=-90. + (.5 + np.arange(20)) * .1)
    typed = tmp_path / 'temperature.npz'
    raw, _ = example()
    np.savez(typed, **raw)
    netcdf = tmp_path / 'temperature.nc'
    netcdf.write_bytes(b'CDF\x01')
    environment = {**os.environ, 'PYTHONPATH': str(REPOSITORY_ROOT / 'src'),
                   'OCEAN_SOLVER_BATHYMETRY': str(bathy)}
    probe = subprocess.run([sys.executable, '-c', textwrap.dedent('''
        import importlib.abc
        import sys
        import numpy as np
        class NoNetCDF(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'netCDF4' or fullname.startswith('netCDF4.'):
                    raise ModuleNotFoundError('netCDF4 unavailable in this witness')
        sys.meta_path.insert(0, NoNetCDF())
        from zhenmode.model.config import DEFAULT_CONFIG
        from zhenmode.model.inputs.bathymetry import Dataset, _read_etopo_global
        from zhenmode.model.inputs.sources import InputSnapshot, load_climatology_snapshot
        from zhenmode.model.solver.factory import make_solver_global
        assert callable(make_solver_global) and Dataset is None
        assert DEFAULT_CONFIG.bathymetry_file == sys.argv[1]
        depth, lon, lat = _read_etopo_global(sys.argv[1], resolution=1., lat_max=90.)
        assert depth.shape == (4, 2)
        np.testing.assert_array_equal(depth, 3000.)
        np.testing.assert_allclose(lon, [.5, 1.5, 2.5, 3.5])
        np.testing.assert_allclose(lat, [-89.5, -88.5])
        field = load_climatology_snapshot(InputSnapshot.read(sys.argv[2]), 'T')
        np.testing.assert_array_equal(field['data'], np.full((2, 3, 4), 7.))
        try:
            load_climatology_snapshot(InputSnapshot.read(sys.argv[3]), 'T')
        except ImportError as error:
            assert 'typed .npz twin' in str(error)
        else:
            raise AssertionError('NetCDF must require its reader')
        assert 'netCDF4' not in sys.modules
    '''), str(bathy), str(typed), str(netcdf)], cwd=tmp_path, env=environment,
        capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert probe.returncode == 0, probe.stdout + probe.stderr


def test_documented_synthetic_bathymetry_cli(tmp_path):
    probe = subprocess.run([sys.executable, str(REPOSITORY_ROOT / 'scripts/make_synthetic_bathymetry.py'),
                            '--out-dir', str(tmp_path)], cwd=tmp_path, capture_output=True,
                           text=True, encoding='utf-8', timeout=30)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert 'NOT ETOPO' in probe.stdout
    # Independent public format/sign contract, with no real data or repository output.
    with np.load(tmp_path / 'ETOPO_2022_v1_r3600x1800_surface.nc.npz') as saved:
        assert set(saved.files) == {'z', 'lon', 'lat'}
        assert saved['z'].shape == (1800, 3600) and saved['z'].dtype == np.int16
        assert .4 < np.mean(saved['z'] < 0) < .95
        assert np.max(saved['z']) == 0 and np.min(saved['z']) < -3000


def test_typed_normal_twin_input_passes_strict(tmp_path):
    assert cli().main(write_inputs(tmp_path)) == 0
    report = json.loads((tmp_path / 'quality.json').read_text())
    assert report['schema'] == 'ocean.input_quality_bundle.v2'
    assert report['publication_state'] == 'committed'
    artifact = report['private_mask_artifact']
    content = base64.b64decode(artifact['content_base64'], validate=True)
    assert len(content) == artifact['bytes']
    assert hashlib.sha256(content).hexdigest() == artifact['sha256']
    with np.load(BytesIO(content), allow_pickle=False) as masks:
        assert not masks['T_raw_missing'].any()
        assert not masks['S_raw_missing'].any()
    assert not (tmp_path / 'quality.masks.npz').exists()
    assert not list(tmp_path.glob('.input-preflight-*'))
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
    assert not (tmp_path / 'quality.json').exists()
    assert not (tmp_path / 'quality.masks.npz').exists()

def test_change_after_json_serialization_leaves_no_published_artifacts(tmp_path, monkeypatch):
    arguments = write_inputs(tmp_path)
    module = cli()
    original_dump = module.json.dump

    def change_after_json(*args, **kwargs):
        original_dump(*args, **kwargs)
        (tmp_path / 'temp.npz').write_bytes(b'changed-after-json-write')

    monkeypatch.setattr(module.json, 'dump', change_after_json)
    assert module.main(arguments) == 2
    assert not (tmp_path / 'quality.json').exists()
    assert not (tmp_path / 'quality.masks.npz').exists()
    assert not list(tmp_path.glob('.input-preflight-*'))

@pytest.mark.parametrize('state', [None, 'pending', 'failed'])
def test_strict_refuses_uncommitted_or_failed_report(state):
    raw, grid = example()
    report, _ = audit_woa_variable(raw, grid, variable='T')
    if state is not None:
        report['publication_state'] = state
    with pytest.raises(ValueError, match='unsupported'):
        enforce_strict_quality(report)

@pytest.mark.parametrize('name', ['quality.json', 'quality.masks.npz'])
def test_existing_user_artifact_is_preserved(tmp_path, name):
    arguments = write_inputs(tmp_path)
    target = tmp_path / name
    target.write_bytes(b'existing-user-artifact')
    with pytest.raises(SystemExit):
        cli().main(arguments)
    assert target.read_bytes() == b'existing-user-artifact'

@pytest.mark.parametrize('name', ['quality.json'])
def test_late_user_artifact_wins_without_overwrite(tmp_path, monkeypatch, name):
    arguments = write_inputs(tmp_path)
    module = cli()
    original_dump = module.json.dump

    def create_user_target_after_staging(*args, **kwargs):
        original_dump(*args, **kwargs)
        (tmp_path / name).write_bytes(b'late-user-artifact')

    monkeypatch.setattr(module.json, 'dump', create_user_target_after_staging)
    assert module.main(arguments) == 2
    assert (tmp_path / name).read_bytes() == b'late-user-artifact'
    other = 'quality.masks.npz' if name == 'quality.json' else 'quality.json'
    assert not (tmp_path / other).exists()
    assert not list(tmp_path.glob('.input-preflight-*'))

def test_strict_missing_input_does_not_publish_success_marker(tmp_path):
    arguments = write_inputs(tmp_path)
    raw, _ = example()
    raw['data'][0, 1, 1] = np.nan
    np.savez(tmp_path / 'temp.npz', **raw)
    assert cli().main(arguments) == 2
    assert not (tmp_path / 'quality.json').exists()
    assert not (tmp_path / 'quality.masks.npz').exists()

@pytest.mark.parametrize('create_unrelated_mask', [False, True])
def test_failed_publication_never_deletes_public_files(tmp_path, monkeypatch,
                                                     create_unrelated_mask):
    arguments = write_inputs(tmp_path)
    module = cli()

    def fail_publication(source, destination):
        assert destination.name == 'quality.json'
        if create_unrelated_mask:
            (tmp_path / 'quality.masks.npz').write_bytes(b'user-mask')
        raise OSError('injected report publication failure')

    monkeypatch.setattr(module.os, 'link', fail_publication)
    assert module.main(arguments) == 2
    assert not (tmp_path / 'quality.json').exists()
    if create_unrelated_mask:
        assert (tmp_path / 'quality.masks.npz').read_bytes() == b'user-mask'
    else:
        assert not (tmp_path / 'quality.masks.npz').exists()
    assert not list(tmp_path.glob('.input-preflight-*'))

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
    report['publication_state'] = 'committed'
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
    report['publication_state'] = 'committed'
    report[field] = value
    with pytest.raises(ValueError, match='unsupported'):
        enforce_strict_quality(report)

def test_forged_metadata_flags_do_not_override_wrong_role():
    raw, grid = example()
    report, _ = audit_woa_variable(raw, grid, variable='T')
    report['publication_state'] = 'committed'
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
    report['publication_state'] = 'committed'
    enforce_strict_quality(report)

def test_bundle_reader_rejects_failed_top_level(tmp_path):
    from zhenmode.model.inputs.quality import enforce_strict_quality_bundle
    assert cli().main(write_inputs(tmp_path)) == 0
    report = json.loads((tmp_path / 'quality.json').read_text())
    report['publication_state'] = 'failed'
    with pytest.raises(ValueError):
        enforce_strict_quality_bundle(report)

def test_bundle_reader_rejects_invalid_base64(tmp_path):
    from zhenmode.model.inputs.quality import enforce_strict_quality_bundle
    assert cli().main(write_inputs(tmp_path)) == 0
    report = json.loads((tmp_path / 'quality.json').read_text())
    report['private_mask_artifact']['content_base64'] = '@invalid'
    with pytest.raises(ValueError):
        enforce_strict_quality_bundle(report)
