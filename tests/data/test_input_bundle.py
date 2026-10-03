"""Bounded v2 bundle round trips and tampering witnesses; no private input data."""
import base64
import copy
import hashlib
import json
import zipfile
from io import BytesIO

import numpy as np
import pytest

from ocean_solver.io.data_quality import enforce_strict_quality_bundle, load_strict_quality_bundle
from tests.support.data.input_quality import cli
from tests.support.data.input_sources import write_inputs


@pytest.fixture
def bundle(tmp_path):
    assert cli().main(write_inputs(tmp_path)) == 0
    return json.loads((tmp_path / 'quality.json').read_text())


def replace_masks(bundle, masks):
    buffer = BytesIO()
    np.savez_compressed(buffer, **masks)
    replace_payload(bundle, buffer.getvalue())


def replace_payload(bundle, content):
    bundle['private_mask_artifact'].update(
        content_base64=base64.b64encode(content).decode('ascii'), bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest())


def test_normal_bundle_file_and_mapping_round_trip(tmp_path, bundle):
    report, masks = load_strict_quality_bundle(tmp_path / 'quality.json')
    assert report == bundle
    assert set(masks) == {f'{label}_{name}' for label in ('T', 'S') for name in (
        'raw_missing', 'target_missing', 'target_support_known', 'target_wet')}
    assert all(mask.dtype == bool for mask in masks.values())
    assert np.array_equal(masks['T_target_wet'], masks['S_target_wet'])


@pytest.mark.parametrize('field,value', [
    ('schema', 'unknown'), ('publication_state', 'failed'),
    ('publication_state', 'uncommitted')])
def test_top_level_rejected(bundle, field, value):
    bundle[field] = value
    with pytest.raises(ValueError):
        enforce_strict_quality_bundle(bundle)


@pytest.mark.parametrize('change', ['swap', 'missing_S', 'extra', 'failed_T', 'S_is_T'])
def test_variable_identity_or_state_rejected(bundle, change):
    variables = bundle['variables']
    if change == 'swap':
        variables['T'], variables['S'] = variables['S'], variables['T']
    elif change == 'missing_S':
        del variables['S']
    elif change == 'extra':
        variables['X'] = copy.deepcopy(variables['T'])
    elif change == 'failed_T':
        variables['T']['publication_state'] = 'failed'
    else:
        variables['S']['variable'] = 'T'
    with pytest.raises(ValueError):
        enforce_strict_quality_bundle(bundle)


@pytest.mark.parametrize('field,value', [
    ('content_base64', '@invalid'), ('content_base64', '\u00e9'),
    ('bytes', 1), ('bytes', True), ('sha256', '0' * 64), ('storage', 'external')])
def test_artifact_integrity_rejected(bundle, field, value):
    bundle['private_mask_artifact'][field] = value
    with pytest.raises(ValueError):
        enforce_strict_quality_bundle(bundle)


@pytest.mark.parametrize('change', ['missing_key', 'extra_key', 'dtype', 'shape',
                                  'missing', 'unknown', 'wet_mismatch', 'raw_count'])
def test_masks_and_counts_rejected_even_with_recomputed_hash(bundle, change):
    masks = enforce_strict_quality_bundle(bundle)
    if change == 'missing_key':
        del masks['T_raw_missing']
    elif change == 'extra_key':
        masks['extra'] = np.zeros(1, dtype=bool)
    elif change == 'dtype':
        masks['T_raw_missing'] = masks['T_raw_missing'].astype(float)
    elif change == 'shape':
        masks['T_raw_missing'] = masks['T_raw_missing'].reshape(-1)
    elif change == 'missing':
        masks['T_target_missing'][0, 0, 0] = True
    elif change == 'unknown':
        masks['T_target_support_known'][0, 0, 0] = False
    elif change == 'wet_mismatch':
        masks['S_target_wet'][0, 0, :] = False
    else:
        masks['T_raw_missing'][0, 0, 0] = True
    replace_masks(bundle, masks)
    with pytest.raises(ValueError):
        enforce_strict_quality_bundle(bundle)


@pytest.mark.parametrize('field,value', [('raw_shape_depth_lat_lon', [2, 3, 99]),
                                       ('target_shape_lon_lat_depth', [4, 3, 99]),
                                       ('raw_missing_nodes', 1), ('wet_nodes', 0),
                                       ('target_source_indices_lon_lat_depth', None)])
def test_report_shape_count_contradictions_rejected(bundle, field, value):
    bundle['variables']['T'][field] = value
    with pytest.raises(ValueError):
        enforce_strict_quality_bundle(bundle)


def test_expansion_bound_before_numpy_allocation(bundle, monkeypatch):
    called = False

    def unexpected_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError('must reject before array allocation')

    monkeypatch.setattr(np, 'load', unexpected_load)
    with pytest.raises(ValueError, match='expansion limit'):
        enforce_strict_quality_bundle(bundle, max_uncompressed_bytes=1)
    assert not called


def test_compressed_and_report_size_bounds(tmp_path, bundle):
    with pytest.raises(ValueError, match='compressed mask limit'):
        enforce_strict_quality_bundle(bundle, max_compressed_bytes=1)
    with pytest.raises(ValueError, match='report size limit'):
        load_strict_quality_bundle(tmp_path / 'quality.json', max_report_bytes=1)


def test_forged_huge_npy_header_rejected_before_allocation(bundle, monkeypatch):
    original = base64.b64decode(bundle['private_mask_artifact']['content_base64'])
    content = BytesIO()
    with zipfile.ZipFile(BytesIO(original)) as source, zipfile.ZipFile(content, 'w') as target:
        for item in source.infolist():
            if item.filename == 'T_raw_missing.npy':
                header = BytesIO()
                np.lib.format.write_array_header_1_0(header, {
                    'descr': '|b1', 'fortran_order': False, 'shape': (2**40, 3, 4)})
                target.writestr(item.filename, header.getvalue())
            else:
                target.writestr(item.filename, source.read(item))
    replace_payload(bundle, content.getvalue())
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('unbounded allocation'))
    with pytest.raises(ValueError, match='dtype, shape or payload'):
        enforce_strict_quality_bundle(bundle)


def test_duplicate_json_key_rejected(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"publication_state":"failed","publication_state":"committed"}')
    with pytest.raises(ValueError, match='duplicate JSON key'):
        load_strict_quality_bundle(path)
