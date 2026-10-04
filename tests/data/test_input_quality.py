"""Small input-support contracts, independent of real/private datasets."""

import base64
import copy
from io import BytesIO

import numpy as np
import pytest

from tests.support.data.input_quality import cli, example
from zhenmode.model.io.climatology import _fill_nan_horizontal_per_level
from zhenmode.model.io.data_quality import audit_woa_variable, enforce_strict_quality


def test_original_valid_input_is_supported_and_unchanged():
    raw, grid = example()
    before = copy.deepcopy(raw)
    report, masks = audit_woa_variable(raw, grid, variable='T')
    report['publication_state'] = 'committed'
    enforce_strict_quality(report)
    assert report['missing_wet_nodes'] == 0
    assert report['status'] == 'supported'
    assert not masks['raw_missing'].any()
    for key in raw:
        np.testing.assert_array_equal(raw[key], before[key])

def test_valid_first_pass_donors_have_weights_and_wet_paths():
    raw, grid = example()
    raw['data'][0, 1, 1] = np.nan
    initial = _fill_nan_horizontal_per_level(raw['data']).transpose(2, 1, 0)
    report, _ = audit_woa_variable(raw, grid, variable='T', initial_field=initial)
    record = report['records'][0]
    assert record['iteration'] == 1
    assert record['first_pass_matches_initial'] is True
    assert sum(d['weight'] for d in record['donors']) == 1.
    assert all(d['depth_compatible'] for d in record['donors'])
    assert all(d['wet_path'] for d in record['donors'])
    # Consistent reconstruction is still not observed support.
    with pytest.raises(ValueError, match='unsupported'):
        enforce_strict_quality(report)

def test_dry_donor_is_reported_not_silently_removed():
    raw, grid = example()
    raw['data'][0, 1, 1] = np.nan
    grid['wet_mask_3d'][0, 1] = 0
    grid['depth'][0, 1] = 0
    report, _ = audit_woa_variable(raw, grid, variable='T')
    donors = report['records'][0]['donors']
    assert any(d['depth_compatible'] is False for d in donors)
    assert len(donors) == 8

def test_outside_domain_donor_is_reported():
    raw, grid = example()
    grid['lat'] = np.array([0., 1.])
    grid['depth'] = grid['depth'][:, 1:]
    grid['wet_mask_3d'] = grid['wet_mask_3d'][:, 1:]
    raw['data'][0, 1, 1] = np.nan
    report, _ = audit_woa_variable(raw, grid, variable='T')
    assert any(not d['boundary_compatible'] for d in report['records'][0]['donors'])

def test_multiple_rounds_are_explicitly_unimplemented():
    raw, grid = example()
    raw['data'][0] = np.nan
    raw['data'][0, 0, 0] = 7.
    report, _ = audit_woa_variable(raw, grid, variable='T')
    center = next(r for r in report['records'] if r['target_index'] == [2, 2, 0])
    assert center['trace_status'] == 'multi_round_or_vertical_unimplemented'
    assert center['iteration'] is None
    assert center['donors'] is None

def test_no_donor_and_all_missing_column_fail_closed():
    raw, grid = example()
    raw['data'][:] = np.nan
    report, masks = audit_woa_variable(raw, grid, variable='S')
    assert report['all_missing_wet_columns'] == 12
    assert report['unsupported_wet_columns'] == 12
    assert masks['target_missing'].all()
    with pytest.raises(ValueError, match='unsupported'):
        enforce_strict_quality(report)

def test_off_grid_mapping_is_unknown_and_strictly_rejected():
    raw, grid = example()
    grid['lon'] += .25
    report, masks = audit_woa_variable(raw, grid, variable='T')
    assert report['coordinate_mapping_supported'] is False
    assert report['missing_wet_nodes'] is None
    assert not masks['target_support_known'].any()
    with pytest.raises(ValueError, match='unsupported'):
        enforce_strict_quality(report)

def test_record_limit_does_not_hide_missing_or_support_status():
    raw, grid = example()
    raw['data'][:] = np.nan
    report, _ = audit_woa_variable(raw, grid, variable='T', max_records=1)
    assert len(report['records']) == 1
    assert report['records_omitted'] == 23
    assert report['missing_wet_nodes'] == 24

def test_legacy_fill_numbers_unchanged_by_diagnostic():
    raw, grid = example()
    raw['data'][0, 1, 1] = np.nan
    raw['data'][1, 1, 1] = np.nan
    expected = np.full(raw['data'].shape, 7.)
    before = _fill_nan_horizontal_per_level(raw['data'])
    audit_woa_variable(raw, grid, variable='T')
    after = _fill_nan_horizontal_per_level(raw['data'])
    np.testing.assert_array_equal(before, expected)
    np.testing.assert_array_equal(after, before)

def test_bad_masks_are_rejected_instead_of_coerced():
    raw, grid = example()
    grid['wet_mask_3d'][1, 1, 0] = np.nan
    with pytest.raises(ValueError, match='binary'):
        audit_woa_variable(raw, grid, variable='T')

@pytest.mark.parametrize('dtype', [bool, np.uint8])
def test_binary_mask_dtypes_accept_valid_wet_prefix(dtype):
    raw, grid = example()
    grid['wet_mask_3d'][1, 1, 1] = 0
    grid['wet_mask_3d'] = grid['wet_mask_3d'].astype(dtype)
    report, _ = audit_woa_variable(raw, grid, variable='T')
    report['publication_state'] = 'committed'
    enforce_strict_quality(report)

def test_masked_raw_value_never_passes_strict_mode():
    raw, grid = example()
    raw['data'] = np.ma.array(raw['data'], mask=False)
    raw['data'].mask[0, 1, 1] = True
    report, _ = audit_woa_variable(raw, grid, variable='T')
    assert report['missing_wet_nodes'] == 1
    with pytest.raises(ValueError, match='unsupported'):
        enforce_strict_quality(report)

def test_masked_initial_value_never_passes_strict_mode():
    raw, grid = example()
    initial = np.ma.array(raw['data'].transpose(2, 1, 0), mask=False)
    initial.mask[1, 1, 0] = True
    report, _ = audit_woa_variable(raw, grid, variable='T', initial_field=initial)
    with pytest.raises(ValueError, match='unsupported'):
        enforce_strict_quality(report)

def test_masked_geometry_is_not_silently_unmasked():
    raw, grid = example()
    grid['wet_mask_3d'] = np.ma.array(grid['wet_mask_3d'], mask=False)
    grid['wet_mask_3d'].mask[1, 1, 0] = True
    with pytest.raises(ValueError, match='masked geometry'):
        audit_woa_variable(raw, grid, variable='T')

def test_cli_records_actual_twin_identity_and_independent_variable_masks(tmp_path):
    import json

    raw, grid = example()
    np.savez(tmp_path / 'grid.npz', **grid)
    salt = {**raw, 'variable': np.array('salinity'), 'units': np.array('1')}
    np.savez(tmp_path / 'salt.nc.npz', **salt)
    raw['data'][0, 1, 1] = np.nan
    np.savez(tmp_path / 'temp.nc.npz', **raw)
    arguments = ['--temperature', str(tmp_path / 'temp.nc'), '--salinity',
                 str(tmp_path / 'salt.nc'), '--grid', str(tmp_path / 'grid.npz'),
                 '--output', str(tmp_path / 'quality.json')]
    assert cli().main(arguments) == 0
    report = json.loads((tmp_path / 'quality.json').read_text())
    assert report['variables']['T']['current_source_identity']['filename'] == 'temp.nc.npz'
    assert report['variables']['T']['missing_wet_nodes'] == 1
    assert report['variables']['S']['missing_wet_nodes'] == 0
    with np.load(BytesIO(base64.b64decode(
            report['private_mask_artifact']['content_base64']))) as masks:
        assert masks['T_raw_missing'].sum() == 1
        assert masks['S_raw_missing'].sum() == 0

def test_cli_diagnostic_preserves_input_and_refuses_overwrites(tmp_path):
    raw, grid = example()
    np.savez(tmp_path / 'grid.npz', **grid)
    raw['data'][0, 1, 1] = np.nan
    np.savez(tmp_path / 'source.nc.npz', **raw)
    salt = {**raw, 'variable': np.array('salinity'), 'units': np.array('1')}
    np.savez(tmp_path / 'salt.nc.npz', **salt)
    original = (tmp_path / 'source.nc.npz').read_bytes()
    arguments = ['--temperature', str(tmp_path / 'source.nc'), '--salinity',
                 str(tmp_path / 'salt.nc'), '--grid', str(tmp_path / 'grid.npz'),
                 '--output', str(tmp_path / 'quality.json')]
    assert cli().main(arguments) == 0
    assert (tmp_path / 'source.nc.npz').read_bytes() == original
    with pytest.raises(SystemExit):
        cli().main(arguments)
    arguments[-1] = str(tmp_path / 'grid.npz')
    with pytest.raises(SystemExit):
        cli().main(arguments)

def test_missing_variable_identity_cannot_pass_strict():
    raw, grid = example()
    del raw['variable']
    report, _ = audit_woa_variable(raw, grid, variable='T')
    with pytest.raises(ValueError, match='unsupported'):
        enforce_strict_quality(report)

def test_finite_woa_fill_value_cannot_pass_strict():
    raw, grid = example()
    raw['data'][:] = 9.96921e36
    raw.update(source_format='ocean.woa_twin.v1', variable='temperature',
               units='degrees_celsius', missing_encoding='fill_value', fill_value=9.96921e36)
    report, _ = audit_woa_variable(raw, grid, variable='T')
    assert report['missing_wet_nodes'] == 24
    with pytest.raises(ValueError, match='unsupported'):
        enforce_strict_quality(report)

def test_wet_node_below_bathymetry_is_rejected():
    raw, grid = example()
    grid['depth'][:] = 1.
    with pytest.raises(ValueError, match='bathymetry'):
        audit_woa_variable(raw, grid, variable='T')

def test_periodic_aliasing_is_rejected():
    raw, grid = example()
    grid['lon'] = np.array([0., 1., 2., 360.])
    with pytest.raises(ValueError, match='longitude'):
        audit_woa_variable(raw, grid, variable='T')

def test_strict_checks_fields_instead_of_status_only():
    report = {'schema': 'ocean.input_quality.v1', 'status': 'supported',
              'missing_wet_nodes': 99, 'coordinate_mapping_supported': False}
    with pytest.raises(ValueError, match='unsupported'):
        enforce_strict_quality(report)
