"""Audit the fixed accepted-353 archive without executing historical code.

Only scalar geometry/inventory receipts and identities leave this reader.
Private paths are supplied at invocation, never copied into the report. The
candidate state remains in memory; no original state or arrays are published.
"""

import argparse
import hashlib
import io
import json
import math
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from .real_geometry import bridge_nodal_dual, require_original_profile, shared_face

HISTORICAL_SOURCE = '212df951c351f82dba74fbc43db5e52b0ad34c47'
IDENTITIES = {
    'accepted': ('cap256_last_authoritative_state.npz', 'e93d4de5a4522407d510e3d473e306a5eab4f62e708a74dac46b9525b61a75d7', 13140109),
    'report': ('report.json', '3e565a7af6bac9c973781393afbfc2a318a71f89a9ec5fc4c40526f0cad2cc6a', 20857),
    'grid': ('grid_used.npz', '336899f3b5a7bf26299453e0b7811cb3c4b00d0350b7fa75ed258ca1640eec98', 277555),
    'params': ('parameter_arrays_used.npz', 'be8a8dd07dcb04d3ad2a80eb98245b6336a2c883405358e673015c6b1ce56320', 1194571),
    'manifest': ('manifest_start.json', '3b06005f5fd497708a5b7455dc1efde60efc9bcdf25e541067bbfd55495c2342', 14941),
    'source_archive': ('source_used.tar.gz', '1445c4cdcc1552ebfe1849015696317d0624d76ed2840d73a6da865273fd02d2', 284077),
}
SELECTED = [('center', (302, 2)), ('east', (303, 2)), ('south', (302, 1)),
            ('north', (302, 3)), ('west_dry', (301, 2))]


def _rho0(scalar_params):
    # The exact historical source modules bind the fallback constant; any
    # explicit manifest density must agree with that inventory convention.
    density = scalar_params.get('rho0', 1025.)
    if type(density) not in (int, float) or density != 1025. or not math.isfinite(density):
        raise ValueError('archived rho0 must match the verified 1025 kg/m3 convention')
    return float(density)


def _load(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as packet:
        return {key: packet[key].copy() for key in packet.files}


def _verify_sources(blob, report, git_repository):
    matched = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
        for name, digest in report['frozen_sha256'].items():
            if not name.startswith('src/'):
                continue
            if not name.endswith('.py') or '..' in name.split('/'):
                raise ValueError('unsafe historical source member')
            member = archive.getmember(name)
            if not member.isfile():
                raise ValueError('regular historical source member required')
            archived = archive.extractfile(member).read()
            if hashlib.sha256(archived).hexdigest() != digest:
                raise ValueError('frozen source digest mismatch')
            historical = subprocess.check_output(
                ['git', '-C', str(git_repository), 'show', HISTORICAL_SOURCE + ':' + name],
                stderr=subprocess.PIPE, timeout=10)
            if archived != historical:
                raise ValueError('historical source object mismatch')
            matched += 1
    if matched != 35:
        raise ValueError('all 35 historical source modules must match')
    return matched


def _column_check(bridge, index, weights, values, area):
    """Independent scalar fsum inventory check, not the bridge intermediates."""
    candidate = bridge.candidate
    count = int(candidate.active_layers[index])
    eta = float(candidate.eta[index])
    expected_h = [float(weights[k]) + (eta if k == 0 else 0.) for k in range(count)]
    expected_water = math.fsum(expected_h)
    expected_stocks = [math.fsum(
        (expected_h[k] if field < 2 else bridge.rho0 * float(weights[k])) * float(values[index][k, field])
        for k in range(count)) for field in range(4)]
    observed_water = math.fsum(float(h) for h in candidate.h[index])
    observed_stocks = [math.fsum(float(candidate.stocks[index][k, field]) for k in range(count))
                       for field in range(4)]
    residual = [observed_stocks[f] - expected_stocks[f] for f in range(4)]
    bounds = [256 * np.finfo(float).eps * max(1., math.fsum(
        abs((expected_h[k] if f < 2 else bridge.rho0 * float(weights[k])) * float(values[index][k, f]))
        for k in range(count))) for f in range(4)]
    if not all(math.isfinite(value) for value in
               [eta, expected_water, observed_water, *expected_stocks,
                *observed_stocks, *bounds, float(area[index])]):
        raise ValueError('finite independent selected-column ledger required')
    if abs(observed_water - expected_water) > 256 * np.finfo(float).eps * max(1., abs(expected_water)) or any(
        abs(error) > bound for error, bound in zip(residual, bounds)
    ):
        raise ValueError('independent selected-column inventory check failed')
    return dict(wet_layers=count, eta_m=eta, discrete_bottom_z_m=float(candidate.bottom[index]),
                band_bottom_z_m=float(candidate.band_bottom[index]), water_m=observed_water,
                minimum_wet_h_m=None if not count else float(candidate.h[index][:count].min()),
                inventory_residual_per_m2=residual, inventory_bounds_per_m2=bounds,
                cell_area_m2=float(area[index]),
                node0_above_surface=bool(bridge.node0_above_surface[index]),
                unsampled_bottom_m=float(bridge.unsampled_bottom_m[index]),
                reference_kinetic_J_m2=float(bridge.reference_kinetic_J_m2[index]),
                original_actual_kinetic_J_m2=float(bridge.original_actual_kinetic_J_m2[index]),
                candidate_kinetic_J_m2=float(bridge.candidate_kinetic_J_m2[index]),
                reference_to_actual_kinetic_cost_J_m2=float(bridge.reference_to_actual_kinetic_J_m2[index]),
                remap_kinetic_change_J_m2=float(bridge.remap_kinetic_change_J_m2[index]))


def _profile_rejections(bridge, selected):
    rejections = {}
    for label, index in selected:
        missing = {}
        for endpoint, upper, lower in [
            ('surface', bridge.candidate.eta[index], bridge.original.interfaces[index][1]),
            ('bottom', bridge.sample_lower_z[index], bridge.candidate.bottom[index])]:
            try:
                require_original_profile(bridge, index, upper, lower)
            except ValueError as error:
                required_message = 'surface boundary' if endpoint == 'surface' else 'bottom profile'
                if required_message not in str(error):
                    raise ValueError('expected physical-support refusal was not established') from error
                missing[endpoint] = str(error)
        if set(missing) != {'surface', 'bottom'}:
            raise ValueError('both expected surface and bottom support refusals are required')
        rejections[label] = missing
    return rejections


def audit_archive(accepted_dir, metadata_dir, git_repository):
    paths = {key: Path(accepted_dir if key in ['accepted', 'report', 'source_archive'] else metadata_dir) / value[0]
             for key, value in IDENTITIES.items()}
    blobs = {key: path.read_bytes() for key, path in paths.items()}
    for key, blob in blobs.items():
        if len(blob) != IDENTITIES[key][2] or hashlib.sha256(blob).hexdigest() != IDENTITIES[key][1]:
            raise ValueError('fixed accepted archive identity mismatch: ' + key)
    report, manifest = (json.loads(blobs[key]) for key in ['report', 'manifest'])
    if report['last_accepted_absolute_step'] != 353 or report['case'] != '1deg_dt300' or not report['frozen_files_unchanged']:
        raise ValueError('only the unchanged accepted step353 archive is authorized')
    matched = _verify_sources(blobs['source_archive'], report, git_repository)
    rho0 = _rho0(manifest['scalar_params'])
    state, grid, params = (_load(blobs[key]) for key in ['accepted', 'grid', 'params'])
    if set(state) != {'u', 'v', 'T', 'S', 'eta', 'ice'} or state['eta'].shape != (360, 132):
        raise ValueError('accepted full-state schema mismatch')
    for name in ['u', 'v', 'T', 'S']:
        if state[name].shape != (360, 132, 14):
            raise ValueError('all original fourteen state levels required')
    if state['ice'].shape != (360, 132) or not np.isfinite(state['ice']).all():
        raise ValueError('finite original ice inventory required')
    depth = -grid['z']
    weights = params['dz_node'].reshape(-1)
    mask = params['wet_mask_z']
    if not np.array_equal(grid['wet_mask_3d'], mask) or not np.array_equal(grid['wet_mask'], mask[..., 0]) or not np.array_equal(params['wet_mask'], mask[..., 0]):
        raise ValueError('archived grid/parameter wet masks must match')
    if not np.array_equal(mask, ((depth <= grid['depth'][..., None]) & (grid['wet_mask'][..., None] == 1)).astype(float)):
        raise ValueError('archived terrain node cutoff differs from nodal wet prefix')
    if not np.array_equal(np.diff(depth), grid['dz']) or not np.array_equal(np.diff(depth), params['dz_3d'].reshape(-1)):
        raise ValueError('archived depth spacing mismatch')
    values = np.stack([state[key] for key in ['T', 'S', 'u', 'v']], axis=-1)
    source_value_hashes = {key: hashlib.sha256(state[key].tobytes()).hexdigest() for key in state}
    bridge = bridge_nodal_dual(depth, weights, mask, state['eta'], values,
                              column_geometry=manifest['scalar_params']['column_geometry'], rho0=rho0)
    candidate, original = bridge.candidate, bridge.original
    reference_column = (mask * weights).sum(axis=-1)
    # Historical H_sw uses 1 m only as a division guard on dry columns.
    # That guard never becomes candidate water or a wet layer.
    guarded_reference_column = np.where(reference_column > 0., reference_column, 1.)
    if not np.array_equal(params['H_sw'], guarded_reference_column):
        raise ValueError('archived reference column depth mismatch')
    if not np.array_equal(params['dz_norm'], mask * weights / guarded_reference_column[..., None]):
        raise ValueError('archived reference normalization mismatch')
    expected_bottom = np.zeros_like(mask)
    live = candidate.active_layers > 0
    ii, jj = np.nonzero(live)
    expected_bottom[ii, jj, candidate.active_layers[live] - 1] = 1.
    if not np.array_equal(params['bottom_mask'], expected_bottom):
        raise ValueError('archived bottom indicator mismatch')
    area = grid['dx_2d'] * float(grid['dy'])
    if area.shape != state['eta'].shape or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError('positive archived cell areas required')
    # Independent global source inventories from source fields/weights, not
    # original/candidate stock totals used by the bridge's own receipt.
    actual_h = mask * weights
    actual_h[..., 0] += state['eta']
    expected_water = math.fsum((area[..., None] * actual_h).ravel())
    observed_water = math.fsum((area[..., None] * candidate.h).ravel())
    expected_stocks, observed_stocks, bounds = [], [], []
    for field, name in enumerate(['T', 'S', 'u', 'v']):
        capacity = actual_h if field < 2 else rho0 * mask * weights
        terms = area[..., None] * capacity * state[name]
        expected_stocks.append(math.fsum(terms.ravel()))
        observed_stocks.append(math.fsum((area[..., None] * candidate.stocks[..., field]).ravel()))
        bounds.append(256 * np.finfo(float).eps * max(1., math.fsum(np.abs(terms).ravel())))
    residual = np.array(observed_stocks) - expected_stocks
    water_bound = 256 * np.finfo(float).eps * max(1., abs(expected_water))
    if not all(math.isfinite(value) for value in
               [expected_water, observed_water, water_bound, *expected_stocks,
                *observed_stocks, *bounds, *residual]):
        raise ValueError('finite independent full-domain ledger required')
    if abs(observed_water - expected_water) > water_bound or np.any(abs(residual) > bounds):
        raise ValueError('independent full-domain inventory check failed')
    if not np.array_equal(candidate.h[..., 3:], original.h[..., 3:]) or not np.array_equal(candidate.stocks[..., 3:, :], original.stocks[..., 3:, :]):
        raise ValueError('all deep source inventory slots must remain unchanged')
    columns = {label: _column_check(bridge, index, weights, values, area) for label, index in SELECTED}
    if [columns[label]['wet_layers'] for label, _ in SELECTED] != [11, 12, 11, 12, 0]:
        raise ValueError('selected accepted wet/dry column identity mismatch')
    faces = {}
    for label, index in SELECTED[1:]:
        normal = (-1., 0.) if label == 'west_dry' else ((1., 0.) if label == 'east' else (0., -1.) if label == 'south' else (0., 1.))
        # Static Q witness only. No production metric/operator equivalence is
        # asserted: explicit diagnostic face length uses archived dual width.
        length = float(grid['dy']) if label in ['east', 'west_dry'] else float(grid['dx_2d'][302, 2])
        face = shared_face(bridge, (302, 2), index, length=length, normal=normal)
        if label == 'west_dry' and (face.segments or face.total_Q_m3_s != 0.
                                   or face.common_upper_z_m is not None or face.common_lower_z_m is not None):
            raise ValueError('dry west must have no common domain and exactly zero Q')
        faces[label] = dict(common_upper_z_m=face.common_upper_z_m, common_lower_z_m=face.common_lower_z_m,
                            segment_count=len(face.segments), total_Q_m3_s=face.total_Q_m3_s,
                            unsupported_original_profile_segments=sum(not segment.original_profile_supported for segment in face.segments),
                            interpretation=face.interpretation)
    rejections = _profile_rejections(bridge, SELECTED[:4])
    for key, path in paths.items():
        if path.read_bytes() != blobs[key]:
            raise ValueError('original archive changed during audit')
    if any(hashlib.sha256(state[key].tobytes()).hexdigest() != digest for key, digest in source_value_hashes.items()):
        raise ValueError('original accepted state arrays changed during audit')
    count_values, count_occurrences = np.unique(candidate.active_layers, return_counts=True)
    return dict(schema='ocean.accepted353_geometry_inventory.v1', contract=bridge.contract,
                accepted_step=353, archived_dt_s=300, real_timestep_executed=False,
                production_step_invoked=False, full_domain_shape=list(candidate.h.shape),
                active_layer_histogram={str(int(n)): int(total) for n, total in zip(count_values, count_occurrences)},
                source_identities={key: dict(filename=name, sha256=digest, bytes=size) for key, (name, digest, size) in IDENTITIES.items()},
                historical_source_sha=HISTORICAL_SOURCE, historical_modules_matched=matched,
                rho0_kg_m3=rho0, rho0_manifest_explicit='rho0' in manifest['scalar_params'],
                all_original_snapshots_unchanged=True, all_original_state_arrays_unchanged=True,
                original_ice_unchanged=True, private_arrays_published=False, private_paths_published=False,
                all_deep_inventory_slots_unchanged=True, candidate_minimum_wet_h_m=float(candidate.h[candidate.wet_mask].min()),
                independent_global_water_residual_m3=observed_water - expected_water, independent_global_water_bound_m3=water_bound,
                independent_global_stock_residuals=residual.tolist(), independent_global_stock_bounds=bounds,
                global_stock_units=['degC*m3', 'psu*m3', 'kg*m/s', 'kg*m/s'],
                maximum_column_stock_roundoff_ratio=float(bridge.stock_roundoff_ratio.max()),
                selected_columns=columns, static_shared_faces=faces, original_profile_rejections=rejections,
                decisions=[
                    'Original nodal-dual discrete inventory is authoritative. Negative eta leaves node0 inventory valid but its FD sample above the physical surface unsupported.',
                    'Reference momentum is preserved. Its conversion to actual-h velocity exposes a kinetic cost, separately from conservative P0 remap energy change.',
                    'Candidate P0 common faces use max discrete bottom / min surface. Dry west has no segments and exactly zero Q; staircase faces exclude deeper unmatched water.'
                ],
                minimum_missing_information=[
                    'A surface boundary/reconstruction rule consistent with the node0 material inventory when node0 lies above eta.',
                    'A specified supported bottom profile or new boundary closure from deepest wet node to the discrete dual bottom.',
                    'Original WOA missing-input masks and donor lineage; finite archived values alone do not recover observational support.'
                ],
                original_physical_profile_recovered=False, pressure_consumer_qualified=False,
                coastal_dynamic_integration_qualified=False, biharmonic_enabled=False,
                momentum_filter_enabled=False, physical_total_energy_budget_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accepted-dir', type=Path, required=True)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--git-repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_archive(args.accepted_dir, args.metadata_dir, args.git_repository)
    result['source_sha256'] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ['real_geometry.py', 'accepted_geometry_audit.py']}
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('Accepted353 static geometry audit passed; no time step or private arrays published.')


if __name__ == '__main__':
    main()
