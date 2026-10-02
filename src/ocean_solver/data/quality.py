"""Read-only WOA support audit; never fills, filters, or changes scientific inputs.

Only exact source/target nodes are supported in this first implementation.
First-pass donor records describe the legacy horizontal-fill candidate, not
complete reconstruction provenance. Any missing wet node fails strict mode.
"""

import base64
import binascii
import hashlib
import json
import math
import zipfile
import zlib
from collections import deque
from io import BytesIO

import numpy as np

from ocean_solver._compat import preserve_legacy_names

_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1))

_UNITS = {'temperature': {'degrees_celsius'}, 'salinity': {'1', 'psu'}}


def _scalar(fields, key):
    value = np.asarray(fields.get(key))
    if value.shape != ():
        raise ValueError(f'{key} must be scalar metadata')
    return value.item()


def normalize_source(raw, variable):
    """Recognize explicit formats/roles/encoding; never infer them from filenames."""
    data = np.ma.asarray(raw['data'], dtype=float).filled(np.nan)
    try:
        role = {'T': 'temperature', 'S': 'salinity'}[variable]
        format_name = _scalar(raw, 'source_format')
        encoding = _scalar(raw, 'missing_encoding')
        identity_ok = (format_name in ('ocean.woa_twin.v1', 'woa-compatible-netcdf.v1')
                       and _scalar(raw, 'variable') == role
                       and _scalar(raw, 'units') in _UNITS[role]
                       and _scalar(raw, 'longitude_units') == 'degrees_east'
                       and _scalar(raw, 'latitude_units') == 'degrees_north'
                       and _scalar(raw, 'depth_units') in ('m', 'meters')
                       and _scalar(raw, 'depth_positive') == 'down')
        encoding_ok = encoding in ('nan', 'fill_value', 'netcdf_masked')
        if format_name == 'ocean.woa_twin.v1' and encoding == 'netcdf_masked':
            encoding_ok = False
        if format_name == 'woa-compatible-netcdf.v1' and encoding != 'netcdf_masked':
            encoding_ok = False
        if encoding in ('fill_value', 'netcdf_masked'):
            fill = float(_scalar(raw, 'fill_value'))
            encoding_ok = encoding_ok and np.isfinite(fill)
            if encoding == 'fill_value':
                source_dtype = np.ma.asarray(raw['data']).dtype
                if source_dtype.kind in 'iu':
                    info = np.iinfo(source_dtype)
                    encoding_ok = encoding_ok and fill.is_integer() and info.min <= fill <= info.max
                elif source_dtype.kind == 'f':
                    encoding_ok = encoding_ok and abs(fill) <= np.finfo(source_dtype).max
                else:
                    encoding_ok = False
                if encoding_ok:
                    encoded_fill = np.asarray(fill, dtype=source_dtype).item()
                    data = np.where(data == encoded_fill, np.nan, data)
        # A known WOA marker left unexplained by the declared encoding is unknown support.
        encoding_ok = encoding_ok and not np.isin(
            data, [9.96921e36, float(np.float32(9.96921e36))]).any()
    except (KeyError, TypeError, ValueError, OverflowError):
        identity_ok = encoding_ok = False
    return data, bool(identity_ok), bool(encoding_ok)


def source_metadata(raw):
    metadata = {}
    for key in ('source_format', 'variable', 'units', 'missing_encoding', 'fill_value',
                'longitude_units', 'latitude_units', 'depth_units', 'depth_positive'):
        try:
            value = _scalar(raw, key)
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            metadata[key] = value if isinstance(value, (str, int, float, bool)) else None
        except ValueError:
            metadata[key] = None
    return metadata


def _exact_indices(source, target):
    """Reject extrapolated/interpolated coordinates instead of inventing lineage."""
    indices = np.abs(source[:, None] - target[None, :]).argmin(axis=0)
    return indices if (len(np.unique(indices)) == len(indices)
                       and np.all(np.abs(source[indices] - target) <= 1e-10)) else None


def _wet_components(mask):
    """Cardinal wet connectivity, periodic longitude and closed latitude."""
    labels = np.full(mask.shape, -1, dtype=int)
    component = 0
    nx, ny = mask.shape
    for i, j in np.argwhere(mask):
        if labels[i, j] >= 0:
            continue
        labels[i, j] = component
        pending = deque([(int(i), int(j))])
        while pending:
            x, y = pending.popleft()
            for xx, yy in (((x - 1) % nx, y), ((x + 1) % nx, y),
                           (x, y - 1), (x, y + 1)):
                if 0 <= yy < ny and mask[xx, yy] and labels[xx, yy] < 0:
                    labels[xx, yy] = component
                    pending.append((xx, yy))
        component += 1
    return labels


def _direct_wet_path(target, donor, wet):
    """Witness a cardinal or two-edge diagonal path; no inferred basin names."""
    i, j = target
    x, y = donor
    if not wet[x, y]:
        return None
    if i == x or j == y:
        return [[i, j], [x, y]]
    for intermediate in ((i, y), (x, j)):
        if wet[intermediate]:
            return [[i, j], list(intermediate), [x, y]]
    return None


def _validate(raw, grid):
    for fields, keys in ((raw, ('lon', 'lat', 'depth')),
                         (grid, ('lon', 'lat', 'z', 'depth', 'wet_mask_3d'))):
        if any(np.ma.getmaskarray(fields[key]).any() for key in keys):
            raise ValueError('masked geometry is unsupported')
    for key in ('lon', 'lat', 'depth'):
        axis = np.asarray(raw[key])
        if (axis.ndim != 1 or not axis.size or not np.isfinite(axis).all()
                or not np.all(np.diff(axis) > 0)):
            raise ValueError(f'raw {key} must be finite and strictly increasing')
    expected = tuple(len(raw[key]) for key in ('depth', 'lat', 'lon'))
    if np.asarray(raw['data']).shape != expected:
        raise ValueError('raw data shape must be depth/lat/lon')
    for key in ('lon', 'lat'):
        axis = np.asarray(grid[key])
        if (axis.ndim != 1 or not axis.size or not np.isfinite(axis).all()
                or not np.all(np.diff(axis) > 0)):
            raise ValueError(f'grid {key} must be finite and strictly increasing')
    z = np.asarray(grid['z'])
    if (z.ndim != 1 or not z.size or not np.isfinite(z).all()
            or np.any(z > 0) or not np.all(np.diff(z) < 0)):
        raise ValueError('grid z must be finite, nonpositive and decreasing')
    wet = np.asarray(grid['wet_mask_3d'])
    if (wet.shape != (len(grid['lon']), len(grid['lat']), len(z))
            or not np.isfinite(wet).all() or not np.isin(wet, [0, 1]).all()):
        raise ValueError('wet_mask_3d must be a finite binary lon/lat/depth mask')
    if np.any(np.diff(wet.astype(np.int8), axis=-1) > 0):
        raise ValueError('wet mask must be a surface-connected prefix')
    depth = np.asarray(grid['depth'])
    if depth.shape != wet.shape[:2] or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError('grid depth must be finite, nonnegative and lon/lat shaped')
    if np.any(wet.astype(bool) & ((-z)[None, None, :] > depth[..., None] + 1e-9)):
        raise ValueError('wet node lies below grid bathymetry')
    if np.any(wet.astype(bool) & (depth[..., None] <= 0)):
        raise ValueError('wet node has nonpositive grid bathymetry')
    for lon in (np.asarray(raw['lon']), np.asarray(grid['lon'])):
        normalized = np.sort((lon + 180) % 360 - 180)
        if len(normalized) > 1 and np.any(np.diff(normalized) <= 1e-10):
            raise ValueError('periodic longitude coordinates must be unique')


def audit_woa_variable(raw, grid, *, variable, initial_field=None, max_records=10000):
    """Return JSON metadata and boolean masks, without modifying any arrays.

    Raw data is depth/lat/lon; grid wet nodes are lon/lat/depth. Longitude is
    periodic, latitude closed. Wet-graph connectivity is not a basin/front
    classifier. Full multi-round and vertical-fallback lineage is unimplemented.
    """
    _validate(raw, grid)
    if not isinstance(max_records, int) or max_records < 0:
        raise ValueError('max_records must be a nonnegative integer')
    wet = np.asarray(grid['wet_mask_3d'], dtype=bool)
    data, identity_ok, encoding_ok = normalize_source(raw, variable)
    raw_missing = ~np.isfinite(data)
    indices = [
        _exact_indices(np.asarray(raw['lon']), (np.asarray(grid['lon']) + 180) % 360 - 180),
        _exact_indices(np.asarray(raw['lat']), np.asarray(grid['lat'])),
        _exact_indices(np.asarray(raw['depth']), -np.asarray(grid['z'])),
    ]
    mapped = all(index is not None for index in indices)
    masks = {'raw_missing': raw_missing.copy(),
             'target_wet': wet.copy(),
             'target_support_known': np.full(wet.shape, mapped, dtype=bool),
             'target_missing': np.zeros(wet.shape, dtype=bool)}
    report = {
        'schema': 'ocean.input_quality.v1', 'variable': variable,
        'raw_shape_depth_lat_lon': list(data.shape),
        'target_shape_lon_lat_depth': list(wet.shape),
        'raw_missing_nodes': int(np.count_nonzero(raw_missing)),
        'wet_nodes': int(np.count_nonzero(wet)),
        'wet_columns': int(np.count_nonzero(wet.any(axis=-1))),
        'target_source_indices_lon_lat_depth': (
            [index.tolist() for index in indices] if mapped else None),
        'coordinate_mapping_supported': mapped,
        'grid_contract_verified': True,
        'source_format_identity_verified': identity_ok,
        'missing_encoding_verified': encoding_ok,
        'source_metadata': source_metadata(raw),
        'historical_raw_identity_verified': False,
        'status': 'unsupported', 'records': [],
        'publication_state': 'uncommitted',
        'unimplemented': ['multi_round_donor_tree', 'vertical_fallback_lineage',
                          'interpolated_or_extrapolated_grid_lineage'],
        'policy': 'diagnostic only; strict rejects every missing wet node or unknown mapping',
        'missing_wet_nodes': None, 'all_missing_wet_columns': None,
        'unsupported_wet_columns': int(np.count_nonzero(wet.any(axis=-1))),
        'records_omitted': 0,
        'nonfinite_initial_wet_nodes': 0, 'changed_raw_valid_nodes': 0,
    }
    if not mapped:
        return report, masks
    ii, jj, kk = indices
    sampled = data[np.ix_(kk, jj, ii)].transpose(2, 1, 0)
    missing = ~np.isfinite(sampled)
    masks['target_missing'] = missing.copy()
    affected = wet & missing
    missing_count = int(np.count_nonzero(affected))
    report.update(
        missing_wet_nodes=missing_count,
        all_missing_wet_columns=int(np.count_nonzero(
            wet.any(axis=-1) & np.all(missing | ~wet, axis=-1))),
        unsupported_wet_columns=int(np.count_nonzero(affected.any(axis=-1))),
        records_omitted=max(0, missing_count - max_records),
    )
    if initial_field is not None:
        initial_field = np.ma.asarray(initial_field, dtype=float).filled(np.nan)
        if initial_field.shape != wet.shape:
            raise ValueError('initial_field must have the wet-node shape')
        invalid = int(np.count_nonzero(wet & ~np.isfinite(initial_field)))
        valid = wet & ~missing
        changed = int(np.count_nonzero(valid & (initial_field != sampled)))
        report.update(nonfinite_initial_wet_nodes=invalid, changed_raw_valid_nodes=changed)
        report['initial_field_comparison'] = 'provided'
    else:
        invalid = changed = 0
        report['initial_field_comparison'] = 'not_provided'
    if missing_count == 0 and invalid == 0 and changed == 0 and identity_ok and encoding_ok:
        report['status'] = 'supported'
    lon_lookup = {int(source): target for target, source in enumerate(ii)}
    lat_lookup = {int(source): target for target, source in enumerate(jj)}
    lat = np.asarray(grid['lat'])
    lower = lat[0] - .5 * (lat[1] - lat[0]) if len(lat) > 1 else lat[0]
    upper = lat[-1] + .5 * (lat[-1] - lat[-2]) if len(lat) > 1 else lat[-1]
    components = {}
    for i, j, k in np.argwhere(affected)[:max_records]:
        i, j, k = int(i), int(j), int(k)
        if k not in components:
            components[k] = _wet_components(wet[:, :, k])
        donors = []
        for di, dj in _OFFSETS:
            x = (int(ii[i]) + di) % len(raw['lon'])
            y = int(np.clip(int(jj[j]) + dj, 0, len(raw['lat']) - 1))
            if raw_missing[kk[k], y, x]:
                continue
            mi, mj = lon_lookup.get(x), lat_lookup.get(y)
            on_grid = mi is not None and mj is not None
            compatible = bool(wet[mi, mj, k]) if on_grid else None
            path = _direct_wet_path((i, j), (mi, mj), wet[:, :, k]) if on_grid else None
            donors.append({
                'source_index_depth_lat_lon': [int(kk[k]), y, x],
                'weight': None, 'iteration': 1, 'source_original_valid': True,
                'lon': float(raw['lon'][x]), 'lat': float(raw['lat'][y]),
                'depth_m': float(raw['depth'][kk[k]]),
                'boundary_compatible': bool(lower <= raw['lat'][y] <= upper),
                'on_model_grid': on_grid, 'depth_compatible': compatible,
                'same_wet_component': bool(compatible and
                    components[k][mi, mj] == components[k][i, j]) if on_grid else None,
                'wet_path': path,
            })
        for donor in donors:
            donor['weight'] = 1. / len(donors)
        match = None
        if donors and initial_field is not None:
            mean = sum(float(data[tuple(d['source_index_depth_lat_lon'])])
                       for d in donors) / len(donors)
            match = bool(abs(mean - initial_field[i, j, k]) <= 1e-12)
        report['records'].append({
            'target_index': [i, j, k], 'raw_missing': True,
            'input_support': 'unsupported',
            'trace_status': ('first_pass_candidate' if donors else
                             'multi_round_or_vertical_unimplemented'),
            'iteration': 1 if donors else None,
            'first_pass_matches_initial': match,
            'donors': donors if donors else None,
            'vertical_fallback': 'unknown',
        })
    return report, masks


def validate_quality_fields(report):
    """Validate scientific input support before publishing a sidecar."""
    if not isinstance(report, dict):
        raise ValueError('unsupported input quality: report must be a mapping')
    flags = ('coordinate_mapping_supported', 'grid_contract_verified',
             'source_format_identity_verified', 'missing_encoding_verified')
    counts = ('missing_wet_nodes', 'all_missing_wet_columns', 'unsupported_wet_columns',
              'records_omitted', 'nonfinite_initial_wet_nodes', 'changed_raw_valid_nodes')
    metadata = report.get('source_metadata')
    if not isinstance(metadata, dict):
        raise ValueError('unsupported input quality: missing source format metadata')
    _, identity_ok, encoding_ok = normalize_source(
        {**metadata, 'data': np.empty(0)}, report.get('variable'))
    if (report.get('schema') != 'ocean.input_quality.v1'
            or report.get('status') != 'supported'
            or not identity_ok or not encoding_ok
            or any(report.get(key) is not True for key in flags)
            or any(type(report.get(key)) is not int or report[key] != 0 for key in counts)
            or report.get('records') != []
            or report.get('variable') not in ('T', 'S')
            or report.get('initial_field_comparison') not in ('provided', 'not_provided')):
        raise ValueError('unsupported input quality: missing, changed, or unknown provenance')


def enforce_strict_quality(report):
    """Local variable support check only; use the bundle reader for artifacts."""
    if not isinstance(report, dict) or report.get('publication_state') != 'committed':
        raise ValueError('unsupported input quality: report is not committed')
    validate_quality_fields(report)


_MAX_COMPRESSED_MASK_BYTES = 32 * 1024 * 1024
_MAX_UNCOMPRESSED_MASK_BYTES = 128 * 1024 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024
_MASK_FIELDS = ('raw_missing', 'target_missing', 'target_support_known', 'target_wet')


def _shape_field(report, name):
    shape = report.get(name)
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(size) is not int or size <= 0 for size in shape)):
        raise ValueError('unsupported bundle: invalid mask shape')
    return tuple(shape)


def _read_bounded_masks(content, expected_shapes, limit):
    """Read only eight boolean NPY members after checking expansion/header bounds."""
    masks = {}
    with zipfile.ZipFile(BytesIO(content)) as archive:
        members = archive.infolist()
        expected = {key + '.npy' for key in expected_shapes}
        if (len(members) != len(expected) or {item.filename for item in members} != expected
                or sum(item.file_size for item in members) > limit):
            raise ValueError('unsupported bundle: mask members or expansion limit')
        for item in members:
            if (item.flag_bits & 1 or item.compress_type not in
                    (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise ValueError('unsupported bundle: mask compression')
            # Central-directory size is checked first, and actual reads are bounded.
            with archive.open(item) as stream:
                member = stream.read(item.file_size + 1)
            if len(member) != item.file_size:
                raise ValueError('unsupported bundle: mask member length')
            buffer = BytesIO(member)
            version = np.lib.format.read_magic(buffer)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(buffer)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(buffer)
            else:
                raise ValueError('unsupported bundle: NPY version')
            key = item.filename[:-4]
            if (dtype != np.dtype(bool) or shape != expected_shapes[key]
                    or math.prod(shape) > limit
                    or buffer.tell() + math.prod(shape) != len(member)):
                raise ValueError('unsupported bundle: mask dtype, shape or payload')
            # Bool payload must be canonical, not arbitrary nonzero bytes.
            if any(value > 1 for value in memoryview(member)[buffer.tell():]):
                raise ValueError('unsupported bundle: noncanonical boolean mask')
            buffer.seek(0)
            masks[key] = np.load(buffer, allow_pickle=False)
    return masks


def enforce_strict_quality_bundle(bundle, *,
                                  max_compressed_bytes=_MAX_COMPRESSED_MASK_BYTES,
                                  max_uncompressed_bytes=_MAX_UNCOMPRESSED_MASK_BYTES):
    """Validate this v2 artifact contract and return masks; no scientific qualification.

    Limits bound compressed bytes, total ZIP expansion and NPY array allocation.
    This is a narrow format/integrity check, not a general hostile-input sandbox.
    """
    for limit in (max_compressed_bytes, max_uncompressed_bytes):
        if type(limit) is not int or limit <= 0:
            raise ValueError('unsupported bundle: invalid size limit')
    if (not isinstance(bundle, dict)
            or bundle.get('schema') != 'ocean.input_quality_bundle.v2'
            or bundle.get('publication_state') != 'committed'):
        raise ValueError('unsupported bundle: schema or publication state')
    variables = bundle.get('variables')
    if not isinstance(variables, dict) or set(variables) != {'T', 'S'}:
        raise ValueError('unsupported bundle: T/S variable identities')
    shapes = {}
    for label, report in variables.items():
        enforce_strict_quality(report)
        if report.get('variable') != label:
            raise ValueError('unsupported bundle: variable identity mismatch')
        raw_shape = _shape_field(report, 'raw_shape_depth_lat_lon')
        target_shape = _shape_field(report, 'target_shape_lon_lat_depth')
        shapes.update({f'{label}_{name}': raw_shape if name == 'raw_missing' else target_shape
                       for name in _MASK_FIELDS})
    if shapes['T_target_wet'] != shapes['S_target_wet']:
        raise ValueError('unsupported bundle: T/S target shapes differ')
    artifact = bundle.get('private_mask_artifact')
    if not isinstance(artifact, dict) or artifact.get('storage') != 'embedded_npz_base64':
        raise ValueError('unsupported bundle: mask storage')
    encoded = artifact.get('content_base64')
    if (not isinstance(encoded, str) or len(encoded) > 4 * ((max_compressed_bytes + 2) // 3)
            or type(artifact.get('bytes')) is not int
            or not 0 < artifact['bytes'] <= max_compressed_bytes):
        raise ValueError('unsupported bundle: compressed mask limit or length')
    try:
        content = base64.b64decode(encoded, validate=True)
        if (len(content) != artifact['bytes'] or len(content) > max_compressed_bytes
                or hashlib.sha256(content).hexdigest() != artifact.get('sha256')):
            raise ValueError('unsupported bundle: mask bytes or hash mismatch')
        masks = _read_bounded_masks(content, shapes, max_uncompressed_bytes)
    except (binascii.Error, zipfile.BadZipFile, EOFError, UnicodeError, OSError,
            zlib.error, NotImplementedError, RuntimeError) as error:
        raise ValueError('unsupported bundle: invalid embedded masks') from error
    for label, report in variables.items():
        raw_missing = masks[f'{label}_raw_missing']
        wet = masks[f'{label}_target_wet']
        missing = masks[f'{label}_target_missing']
        known = masks[f'{label}_target_support_known']
        indices = report.get('target_source_indices_lon_lat_depth')
        if not isinstance(indices, list) or len(indices) != 3:
            raise ValueError('unsupported bundle: missing exact-node mapping')
        for axis, index in enumerate(indices):
            raw_axis = (2, 1, 0)[axis]
            if (not isinstance(index, list) or len(index) != wet.shape[axis]
                    or any(type(value) is not int or not 0 <= value < raw_missing.shape[raw_axis]
                           for value in index) or len(set(index)) != len(index)):
                raise ValueError('unsupported bundle: invalid exact-node mapping')
        ii, jj, kk = indices
        if not np.array_equal(missing, raw_missing[np.ix_(kk, jj, ii)].transpose(2, 1, 0)):
            raise ValueError('unsupported bundle: raw/target missing masks disagree')
        affected = wet & missing
        counts = {
            'wet_nodes': int(np.count_nonzero(wet)),
            'wet_columns': int(np.count_nonzero(wet.any(axis=-1))),
            'raw_missing_nodes': int(np.count_nonzero(raw_missing)),
            'missing_wet_nodes': int(np.count_nonzero(affected)),
            'unsupported_wet_columns': int(np.count_nonzero(affected.any(axis=-1))),
            'all_missing_wet_columns': int(np.count_nonzero(
                wet.any(axis=-1) & np.all(missing | ~wet, axis=-1))),
        }
        if (not known.all() or np.any(np.diff(wet.astype(np.int8), axis=-1) > 0)
                or any(type(report.get(key)) is not int or report[key] != value
                       for key, value in counts.items())):
            raise ValueError('unsupported bundle: mask/count contradiction')
    if not np.array_equal(masks['T_target_wet'], masks['S_target_wet']):
        raise ValueError('unsupported bundle: T/S wet masks differ')
    return masks


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('unsupported bundle: duplicate JSON key')
        result[key] = value
    return result


def load_strict_quality_bundle(path, *, max_report_bytes=_MAX_REPORT_BYTES, **limits):
    """Bound file reading, validate the complete v2 bundle, return report and masks."""
    if type(max_report_bytes) is not int or max_report_bytes <= 0:
        raise ValueError('unsupported bundle: invalid report size limit')
    with open(path, 'rb') as stream:
        content = stream.read(max_report_bytes + 1)
    if len(content) > max_report_bytes:
        raise ValueError('unsupported bundle: report size limit')
    try:
        bundle = json.loads(content, object_pairs_hook=_unique_json_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('unsupported bundle: invalid JSON') from error
    return bundle, enforce_strict_quality_bundle(bundle, **limits)

preserve_legacy_names(globals(), 'input_quality')
