"""Fixed accepted353 static pressure audit; outputs scalar receipts only."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import inventory_pressure as p
from . import pressure_oracle as oracle
from .accepted_geometry_audit import IDENTITIES, SELECTED, _load, audit_archive
from .real_geometry import bridge_nodal_dual


def nonlinear_truncation_scan():
    """A static quadratic-density truncation probe, not a time-order claim."""
    errors = []
    for layers in [4, 8, 16]:
        z = np.linspace(0., -2., layers + 1)[None, :]
        h = -np.diff(z, axis=-1)
        top, base = z[:, :-1], z[:, 1:]
        mean_T = 15. + .1 * (top + base) + .05 * (top**2 + top * base + base**2) / 3.
        stocks = np.zeros(h.shape + (4,))
        stocks[..., 0], stocks[..., 1] = h * mean_T, h * 35.
        state = p.ColumnStocks(np.array([layers]), np.ones_like(h, dtype=bool), np.array([0.]), np.array([-2.]),
                               np.array([z[0, min(3, layers)]]), z, h, stocks)
        profile = p.reconstruct(state)
        differences = []
        for depth in np.linspace(-2., 0., 129):
            exact = profile.eos.gravity * profile.eos.rho0 * profile.eos.alpha * (.1 * depth**2 + .05 * depth**3 / 3.)
            differences.append(abs(p.pressure(profile, (0,), float(depth), reduced=True) - exact))
        errors.append(max(differences))
    return dict(case='exact CV means of quadratic T(z), constant S; static analytic hydrostatic pressure',
                layers=[4, 8, 16], maximum_pressure_error_Pa=errors,
                errors_are_physical_truncation=True, full_method_order_claimed=False)


def _check_ratio(error, scale, label):
    if not math.isfinite(error) or not math.isfinite(scale) or scale < 0:
        raise ValueError('finite independent oracle ledger required: ' + label)
    ratio = abs(error) / (256. * np.finfo(float).eps * max(1., scale))
    if ratio > 1:
        raise ValueError('independent pressure oracle failed: ' + label)
    return ratio


def _requests_and_walls(dx, dy):
    requests = [p.FaceRequest((0,), (1,), dy, (1., 0.)), p.FaceRequest((0,), (2,), dx, (0., -1.)),
                p.FaceRequest((0,), (3,), dx, (0., 1.)), p.FaceRequest((0,), (4,), dy, (-1., 0.))]
    # Explicit static-patch sidewalls, not a claim that outer sampled neighbors
    # are land in the original global domain. West dry is an archived dry coast.
    walls = [p.OuterWall((1,), dy, (1., 0.)), p.OuterWall((1,), dx, (0., 1.)), p.OuterWall((1,), dx, (0., -1.)),
             p.OuterWall((2,), dy, (1., 0.)), p.OuterWall((2,), dy, (-1., 0.)), p.OuterWall((2,), dx, (0., -1.)),
             p.OuterWall((3,), dy, (1., 0.)), p.OuterWall((3,), dy, (-1., 0.)), p.OuterWall((3,), dx, (0., 1.))]
    return requests, walls


def _static_footprint(dx, dy):
    bounds = np.array([[0., dx, 0., dy], [dx, 2. * dx, 0., dy],
                       [0., dx, -dy, 0.], [0., dx, dy, 2. * dy], [-dx, 0., 0., dy]])
    return p.RectangularFootprint(bounds, np.full(5, dx * dy))


def audit(accepted_dir, metadata_dir, git_repository):
    geometry_receipt = audit_archive(accepted_dir, metadata_dir, git_repository)
    paths = {role: Path(accepted_dir if role in ['accepted', 'report', 'source_archive'] else metadata_dir) / identity[0]
             for role, identity in IDENTITIES.items()}
    snapshots = {key: path.read_bytes() for key, path in paths.items()}
    if any(hashlib.sha256(blob).hexdigest() != IDENTITIES[key][1] for key, blob in snapshots.items()):
        raise ValueError('verified accepted archive changed before pressure consumption')
    state, grid, params = (_load(snapshots[key]) for key in ['accepted', 'grid', 'params'])
    manifest = json.loads(snapshots['manifest'])
    source_hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in state.items()}
    columns = [index for _, index in SELECTED]
    values = np.stack([np.column_stack([state[key][index] for key in ['T', 'S', 'u', 'v']]) for index in columns])
    eta = np.array([state['eta'][index] for index in columns])
    mask = np.stack([params['wet_mask_z'][index] for index in columns])
    weights, depth = params['dz_node'].reshape(-1), -grid['z']
    bridge = bridge_nodal_dual(depth, weights, mask, eta, values,
                              column_geometry=manifest['scalar_params']['column_geometry'])
    eos = p.EOS(Tref=manifest['scalar_params']['T_ref'], Sref=manifest['scalar_params']['S_ref'])
    profile = p.reconstruct(bridge.candidate, eos=eos)
    old_p0, new_p0 = p.reconstruct(bridge.original, eos=eos, representation='p0'), p.reconstruct(bridge.candidate, eos=eos, representation='p0')
    area = np.array([float(grid['dx_2d'][index] * grid['dy']) for index in columns])
    requests, walls = _requests_and_walls(float(grid['dx_2d'][columns[0]]), float(grid['dy']))
    footprint = _static_footprint(float(grid['dx_2d'][columns[0]]), float(grid['dy']))
    faces = p.pressure_faces(profile, requests)
    boundary = p.boundary_force_budget(profile, requests, walls)
    certificate = p.certify_force_consumption(profile, requests, walls, footprint=footprint)
    # Original inventory costs use their archived column areas. The declared
    # equal-width diagnostic patch has its own bound areas for pressure work.
    work_trial = p.algebraic_midpoint_work(profile, faces, footprint.area_m2, footprint=footprint)
    if work_trial['work_roundoff_ratio'] > 1:
        raise ValueError('fixed-mass algebraic midpoint work check failed')
    independent_boundary = oracle.boundary_forces(profile, requests, walls)
    boundary_scale = boundary['absolute_traction_scale_N']
    boundary_ratio = max(_check_ratio(float(got - want), boundary_scale, 'boundary force') for got, want in
                         zip(boundary['physical_boundary_force_N'], independent_boundary['physical']))
    cap_ratio = max(_check_ratio(float(got - want), boundary_scale, 'surface geometric cap') for got, want in
                    zip(boundary['surface_geometry_difference_N'], independent_boundary['cap']))
    oracle_pressure_ratio = 0.
    summaries = {}
    for neighbor, request in zip(['east', 'south', 'north', 'west_dry'], requests):
        selected = [item for item in faces if item.right == request.right]
        for item in selected:
            left = oracle.pressure_integral(profile, item.left, item.lower_z_m, item.upper_z_m, reduced=True)
            right = oracle.pressure_integral(profile, item.right, item.lower_z_m, item.upper_z_m, reduced=True)
            expected = (right - left) / (item.upper_z_m - item.lower_z_m)
            oracle_pressure_ratio = max(oracle_pressure_ratio, _check_ratio(item.pressure_jump_Pa - expected,
                (abs(left) + abs(right)) / (item.upper_z_m - item.lower_z_m), 'shared segment'))
        summaries[neighbor] = dict(segment_count=len(selected), sum_Q_m3_s=math.fsum(item.volume_flux_m3_s for item in selected),
                                   maximum_abs_pressure_jump_Pa=max([abs(item.pressure_jump_Pa) for item in selected], default=0.),
                                   pressure_force_N=[math.fsum(-item.area_m2 * item.pressure_jump_Pa * item.normal[axis] for item in selected) for axis in range(2)])
    pe_ratios, pe_values = [], []
    for representation in [old_p0, new_p0, profile]:
        computed, independent = p.potential_energy(representation, area), oracle.potential_energy(representation, area)
        pe_ratios.append(_check_ratio(computed - independent, abs(computed) + abs(independent), 'PE first moment'))
        pe_values.append(computed)
    pe_changes = []
    pe_change_ratios = []
    pe_change_scales = []
    for before, after in [(old_p0, new_p0), (new_p0, profile)]:
        ledger = p.potential_energy_difference_ledger(before, after, area)
        computed, scale = ledger['change_J'], ledger['absolute_contribution_scale_J']
        independent = oracle.potential_energy_difference(before, after, area)
        pe_change_ratios.append(_check_ratio(computed - independent, scale, 'direct PE representation change'))
        pe_changes.append(computed)
        pe_change_scales.append(scale)
    diagnostic_pe = [p.potential_energy(representation, footprint.area_m2, footprint=footprint)
                     for representation in [old_p0, new_p0, profile]]
    diagnostic_pe_changes = [p.potential_energy_difference(before, after, footprint.area_m2)
                             for before, after in [(old_p0, new_p0), (new_p0, profile)]]
    diagnostic_change_ratios = []
    for before, after in [(old_p0, new_p0), (new_p0, profile)]:
        ledger = p.potential_energy_difference_ledger(before, after, footprint.area_m2)
        independent = oracle.potential_energy_difference(before, after, footprint.area_m2)
        diagnostic_change_ratios.append(_check_ratio(ledger['change_J'] - independent,
            ledger['absolute_contribution_scale_J'], 'diagnostic patch direct PE change'))
    for representation, computed in zip([old_p0, new_p0, profile], diagnostic_pe):
        independent = oracle.potential_energy(representation, footprint.area_m2)
        _check_ratio(computed - independent, abs(computed) + abs(independent), 'diagnostic patch PE')
    node_differences = {}
    for i, (label, _) in enumerate(SELECTED[:4]):
        n = int(bridge.original.active_layers[i])
        anomaly = eos.rho0 * (-eos.alpha * (values[i, :n, 0] - eos.Tref) + eos.beta * (values[i, :n, 1] - eos.Sref))
        old_pressure = [eos.rho0 * eos.gravity * eta[i]]
        for k in range(1, n):
            old_pressure.append(old_pressure[-1] + eos.gravity * .5 * (anomaly[k - 1] + anomaly[k]) * float(depth[k] - depth[k - 1]))
        differences = [p.pressure(profile, (i,), -float(depth[k]), reduced=True) - old_pressure[k]
                       for k in range(n) if -depth[k] <= eta[i]]
        node_differences[label] = dict(compared_physical_wet_nodes=len(differences),
                                      node0_above_surface_excluded=bool(eta[i] < 0),
                                      minimum_new_minus_old_Pa=min(differences), maximum_new_minus_old_Pa=max(differences),
                                      maximum_abs_new_minus_old_Pa=max(abs(value) for value in differences))
    try:
        p.require_force_consumption(profile, requests, walls, footprint=footprint)
        force_status = 'accepted_static_force_contract'
    except p.PressureForceIncompatibility:
        force_status = 'blocked_physical_force_incompatibility'
    # This consistency check diagnoses the missing cap term. It does not make
    # that term a solid wall or change physical rejection into acceptance.
    explained_ratio = max(_check_ratio(float(error - cap), boundary_scale, 'force/cap diagnosis') for error, cap in
                          zip(certificate['incompatibility_N'], boundary['surface_geometry_difference_N']))
    for key, path in paths.items():
        if path.read_bytes() != snapshots[key]:
            raise ValueError('original accepted archive changed during static pressure audit')
    if any(hashlib.sha256(state[key].tobytes()).hexdigest() != digest for key, digest in source_hashes.items()):
        raise ValueError('original arrays changed during static pressure audit')
    return dict(schema='ocean.accepted353_inventory_pressure.v1', profile_contract=p.CONTRACT,
                profile_authority='inventory_mean_control_volume_representation', observational_profile_recovered=False,
                original_FD_profile_equivalence=False, accepted_step=353, input_slots=[5, 14, 4],
                active_layers=bridge.candidate.active_layers.tolist(), all_original_snapshots_unchanged=True,
                all_original_arrays_unchanged=True, deep_stocks_unchanged=bool(np.array_equal(bridge.original.stocks[:, 3:], bridge.candidate.stocks[:, 3:])),
                private_arrays_published=False, private_paths_published=False, time_step_executed=False,
                source_identities=geometry_receipt['source_identities'], historical_source_sha=geometry_receipt['historical_source_sha'],
                historical_modules_matched=geometry_receipt['historical_modules_matched'],
                eos={item.name: getattr(eos, item.name) for item in p.fields(p.EOS)},
                original_observational_profile_rejections=geometry_receipt['original_profile_rejections'],
                node0_above_surface=bridge.node0_above_surface.tolist(), unsampled_bottom_m=bridge.unsampled_bottom_m.tolist(),
                slope_limited_components=profile.slope_limited_count, static_shared_pressure=summaries,
                density_authority=profile.density_authority,
                density_slope_limited_cells=profile.density_slope_limited_count,
                pointwise_density_equals_auxiliary_TS_reconstruction=profile.pointwise_density_equals_auxiliary_TS_reconstruction,
                new_vs_old_FD_physical_wet_node_pressure=node_differences,
                PE_definition='rho0 free surface plus rho-prime gravitational first moment, excluding constant fixed-bottom datum',
                inventory_PE_KE_area_scope='Original archived per-column dx_2d[index]*dy; separate from the diagnostic pressure patch.',
                pressure_work_PE_area_scope='Declared equal-center-width rectangular footprint.area_m2, checked against physical box geometry.',
                maximum_relative_original_vs_diagnostic_area_difference=float(np.max(abs(area - footprint.area_m2) / footprint.area_m2)),
                diagnostic_patch_PE_J=diagnostic_pe, diagnostic_patch_PE_changes_J=diagnostic_pe_changes,
                original_inventory_P0_PE_J=pe_values[0], remapped_inventory_P0_PE_J=pe_values[1], remapped_inventory_P1_PE_J=pe_values[2],
                P0_remap_PE_change_J=pe_changes[0], P1_representation_moment_PE_change_J=pe_changes[1],
                PE_change_computation='Direct rho-prime first-moment difference on union physical intervals; free-surface term cancels on identical eta/bottom domains.',
                direct_PE_change_absolute_contribution_scale_J=pe_change_scales,
                total_PE_subtraction_roundoff_difference_J=[pe_values[1] - pe_values[0] - pe_changes[0], pe_values[2] - pe_values[1] - pe_changes[1]],
                reference_M_to_actual_mass_KE_cost_J=math.fsum(area * bridge.reference_to_actual_kinetic_J_m2),
                P0_momentum_remap_KE_change_J=math.fsum(area * bridge.remap_kinetic_change_J_m2),
                algebraic_fixed_mass_trial=work_trial, physical_boundary_traction=boundary,
                force_consumption_certificate=certificate, pressure_force_consumer_status=force_status,
                independent_oracle_ratios=dict(shared_pressure=oracle_pressure_ratio, PE=max(pe_ratios), direct_PE_change=max(pe_change_ratios), diagnostic_direct_PE_change=max(diagnostic_change_ratios), physical_boundary=boundary_ratio,
                                              geometric_cap=cap_ratio, incompatibility_explained_by_cap=explained_ratio),
                face_metric_scope='Same explicitly bound archived center dual-width diagnostic faces as PR20; no production operator equivalence.',
                outer_boundary_scope='Four wet-column closed local Cartesian star with explicit static outer sidewalls; outer sampled neighbors are not declared original land.',
                source_array_or_model_advanced=False, moving_pressure_enabled=False, physical_total_energy_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accepted-dir', type=Path, required=True)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--git-repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.accepted_dir, args.metadata_dir, args.git_repository)
    result['synthetic_nonlinear_truncation'] = nonlinear_truncation_scan()
    result['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ['inventory_pressure.py', 'pressure_footprint.py', 'pressure_oracle.py', 'pressure_static_evidence.py']}
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('Static pressure profile consumed; force status: ' + result['pressure_force_consumer_status'])


if __name__ == '__main__':
    main()
