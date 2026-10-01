"""Scalarize private observed updates; retain unclosed conversions explicitly."""
import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np


def unpack(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('archive_dir', 'probe_dir', 'protocol', 'original_protocol', 'output'):
        parser.add_argument('--'+key.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    original_bytes = args.original_protocol.read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != protocol['original_protocol_sha256']:
        raise ValueError('original protocol identity mismatch')
    original = json.loads(original_bytes)
    content = {}
    for name in ('parameter_arrays_used.npz', 'manifest_start.json'):
        content[name] = (args.archive_dir / name).read_bytes()
        if hashlib.sha256(content[name]).hexdigest() != original['base_inputs'][name]:
            raise ValueError('summary input identity mismatch')
    arrays = unpack(BytesIO(content['parameter_arrays_used.npz']))
    scalars = json.loads(content['manifest_start.json'])['scalar_params']
    rho, gravity = 1025., 9.81  # Matched historical config; report below.
    area = arrays['dx_2d'] * scalars['dy']
    wet = arrays['wet_mask']
    wet3 = arrays['wet_mask_z']
    h0 = arrays['dz_node'] * wet3
    height = np.sum(h0, axis=-1)
    mass = rho * area[..., None] * h0
    bt_mass = rho * area * height
    normal = arrays['interior_mask_z'][..., 0]
    factor = 4096 * np.finfo(np.float64).eps
    expected = np.divide(h0, height[..., None], out=np.zeros_like(h0), where=height[..., None] > 0)
    norm_error = float(np.max(np.abs(expected-arrays['dz_norm'])))
    height_error = float(np.max(np.abs(height-arrays['H_sw'])[wet > 0]))
    if norm_error > factor or height_error > factor * max(1., float(height.max())):
        raise ValueError('reference momentum weight contract mismatch')

    def kinetic(u, v, weights):
        return float(.5 * np.sum(weights * (u*u + v*v)))

    def potential(eta):
        return float(.5 * rho * gravity * np.sum(area * wet * eta*eta))

    def mean_faces(u, v):
        common_x = np.sum(h0*np.roll(wet3, -1, axis=0), axis=-1)
        common_y = np.sum(h0*np.roll(wet3, -1, axis=1), axis=-1)
        cosine_face = .5*(arrays['cos_lat']+np.roll(arrays['cos_lat'], -1))
        x = .5*(u+np.roll(u, -1, axis=0))*common_x
        y = .5*(v+np.roll(v, -1, axis=1))*common_y*cosine_face[None, :]
        y[:, -1] = 0
        return x, y

    def divergence(faces):
        x, y = faces
        incoming = np.roll(y, 1, axis=1)
        incoming[:, 0] = 0
        return ((x-np.roll(x,1,axis=0))*arrays['inv_dx'][...,0]
                +(y-incoming)*scalars['inv_dy']/arrays['cos_lat'][None,:])

    def sum_work(u, v, du, dv, weights):
        terms = weights * (u * du + v * dv)
        return float(np.sum(terms)), float(np.sum(np.abs(weights*u*du) + np.abs(weights*v*dv)))

    def bound(*values):
        return float(factor * max(1., sum(abs(float(value)) for value in values)))

    def map_record(ua, va, ub, vb, weights):
        before, after = kinetic(ua, va, weights), kinetic(ub, vb, weights)
        work, absolute = sum_work(.5*(ua+ub), .5*(va+vb), ub-ua, vb-va, weights)
        tolerance = bound(before, after, absolute)
        residual = after-before-work
        return dict(delta_K_J=after-before, midpoint_increment_work_J=work,
                    identity_residual_J=residual, arithmetic_bound_J=tolerance,
                    identity_check=bool(abs(residual) <= tolerance))

    records = json.loads((args.probe_dir / 'instrumented' / 'records.json').read_text())
    output = dict(protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
                  reference_norm_error=norm_error, reference_height_error_m=height_error,
                  constants=dict(rho0_kg_m3=rho, g_m_s2=gravity),
                  resources=json.loads((args.probe_dir/'resources.json').read_text()), results=[])
    for record in records:
        if not record['equivalence']['passed']:
            raise ValueError('no valid equivalence witness')
        path = args.probe_dir / 'instrumented' / (record['label']+'-stages-private.npz')
        fields = unpack(path)
        if not all(np.all(np.isfinite(value)) for value in fields.values()):
            raise ValueError('nonfinite stage observation')
        names = ['incoming', 'first_drag', 'first_material_L', 'N_predictor', 'predictor_L',
                 'fast', 'tracer_finalize_velocity_unchanged', 'final_drag', 'final_wall']
        stages, inventory = [], []
        for index, name in enumerate(names):
            u, v, eta = [fields[f'outer_{index}_{key}'] for key in ('u', 'v', 'eta')]
            mean_u, mean_v = np.sum(u*arrays['dz_norm'], axis=-1), np.sum(v*arrays['dz_norm'], axis=-1)
            reference = kinetic(u, v, mass)
            moving_extra = float(.5*rho*np.sum(area*wet*eta*(u[..., 0]**2+v[..., 0]**2)))
            bt_energy = kinetic(mean_u, mean_v, bt_mass)
            inventory.append(dict(stage=name, Kref_J=reference, Kbt_J=bt_energy, Kshear_J=reference-bt_energy,
                                  Eeta_J=potential(eta), Kmat_minus_Kref_J=moving_extra,
                                  component_impulse_kg_m_s=[float(np.sum(mass*u)), float(np.sum(mass*v))]))
            if index:
                previous_u, previous_v = fields[f'outer_{index-1}_u'], fields[f'outer_{index-1}_v']
                item = map_record(previous_u, previous_v, u, v, mass)
                item.update(stage=name, delta_Eeta_J=potential(eta)-inventory[index-1]['Eeta_J'],
                            delta_moving_weight_extra_J=moving_extra-inventory[index-1]['Kmat_minus_Kref_J'])
                if name in ('first_drag', 'final_drag', 'final_wall'):
                    item['nonpositive_reference_energy_check'] = bool(item['delta_K_J'] <= item['arithmetic_bound_J'])
                stages.append(item)
        fast, totals = [], {key: 0. for key in ['density_work_J','wind_work_J','eta_pressure_work_J',
                                              'coriolis_work_J','delta_Eeta_drift_J','eta_filter_delta_E_J',
                                              'velocity_postmap_delta_K_J','kick_mask_delta_K_J','delta_Kbt_J']}
        duration = scalars['dt_bt']
        for index in range(scalars['n_subcyc']):
            def field(name):
                return fields['fast_'+name][index]
            ua, va, ub, vb = [field(name) for name in ('u_old','v_old','u_kick','v_kick')]
            midpoint_u, midpoint_v = .5*(ua+ub), .5*(va+vb)
            components = {
                'eta_pressure': (-duration*gravity*field('gradient_x')*wet,
                                 -duration*gravity*field('gradient_y')*wet*normal),
                'density': (duration*field('density_x')*wet, duration*field('density_y')*wet*normal),
                'wind': (duration*arrays['tau_x_2d']/(rho*arrays['H_sw'])*wet,
                         duration*arrays['tau_y_2d']/(rho*arrays['H_sw'])*wet*normal),
                'coriolis': (duration*arrays['f']*normal*midpoint_v*wet,
                             -duration*arrays['f']*normal*midpoint_u*wet)}
            increments = [sum(value[axis] for value in components.values()) for axis in (0, 1)]
            increment_error = max(float(np.max(np.abs((ub-ua)-increments[0]))),
                                  float(np.max(np.abs((vb-va)-increments[1]))))
            velocity_scale = max(1., *(float(np.max(np.abs(value))) for value in [ua, va, ub, vb, *increments]))
            component_work, absolute_work, impulses = {}, {}, {}
            for name, (du, dv) in components.items():
                component_work[name], absolute_work[name] = sum_work(midpoint_u, midpoint_v, du, dv, bt_mass)
                impulses[name] = [float(np.sum(bt_mass*du)), float(np.sum(bt_mass*dv))]
            kick = map_record(ua, va, ub, vb, bt_mass)
            reconstruction = kick['delta_K_J'] - sum(component_work.values())
            tolerance = bound(kinetic(ua,va,bt_mass), kinetic(ub,vb,bt_mass), sum(absolute_work.values()))
            eta_in, eta_mid, eta_transport, eta_out = [field(key) for key in ('eta_in','eta_mid','eta_transport','eta_out')]
            drift = potential(eta_transport)-potential(eta_in)
            eta_filter = potential(eta_out)-potential(eta_transport)
            bar_faces_first, bar_faces_second = mean_faces(ua,va), mean_faces(ub,vb)
            actual_first, actual_second = (field('first_x'),field('first_y')), (field('second_x'),field('second_y'))
            shear_first = tuple(a-b for a,b in zip(actual_first,bar_faces_first,strict=True))
            shear_second = tuple(a-b for a,b in zip(actual_second,bar_faces_second,strict=True))
            def drift_energy(first_faces, second_faces):
                return float(-.5*duration*rho*gravity*np.sum(area*wet*(
                    .5*(eta_in+eta_mid)*divergence(first_faces)
                    +.5*(eta_mid+eta_transport)*divergence(second_faces))))
            shear_drift, bar_drift = drift_energy(shear_first,shear_second), drift_energy(bar_faces_first,bar_faces_second)
            drift_residual = drift-shear_drift-bar_drift
            drift_bound = bound(potential(eta_in),potential(eta_transport),shear_drift,bar_drift)
            postmap = map_record(ub, vb, field('u_out'), field('v_out'), bt_mass)
            kick_mask = map_record(field('u_in'), field('v_in'), ua, va, bt_mass)
            north_flux = max(float(np.max(np.abs(field('first_y')[:, -1]))),
                             float(np.max(np.abs(field('second_y')[:, -1]))))
            item = dict(substep=index+1, kick=kick, work_J=component_work, component_impulse_kg_m_s=impulses,
                        component_work_reconstruction_residual_J=reconstruction, work_bound_J=tolerance,
                        work_reconstruction_check=bool(abs(reconstruction) <= tolerance),
                        velocity_increment_error_m_s=increment_error,
                        velocity_increment_bound_m_s=float(factor*velocity_scale),
                        increment_check=bool(increment_error <= factor*velocity_scale),
                        coriolis_zero_work_check=bool(abs(component_work['coriolis']) <= bound(absolute_work['coriolis'])),
                        delta_Eeta_first_drift_J=potential(eta_mid)-potential(eta_in),
                        delta_Eeta_second_drift_J=potential(eta_transport)-potential(eta_mid),
                        eta_pressure_plus_drift_defect_J=component_work['eta_pressure']+drift,
                        frozen_shear_eta_drift_J=shear_drift, barotropic_eta_drift_J=bar_drift,
                        eta_pressure_plus_barotropic_drift_defect_J=component_work['eta_pressure']+bar_drift,
                        drift_reconstruction_residual_J=drift_residual, drift_bound_J=drift_bound,
                        drift_reconstruction_check=bool(abs(drift_residual) <= drift_bound),
                        velocity_postmap=postmap, kick_input_mask=kick_mask,
                        eta_filter_delta_E_J=eta_filter,
                        closed_north_face_max_abs_m2_s=north_flux,
                        closed_face_check=bool(north_flux == 0),
                        south_exterior_definition='incoming y face explicitly zero in original divergence')
            fast.append(item)
            for key in ('density','wind','eta_pressure','coriolis'):
                totals[key+'_work_J'] += component_work[key]
            totals['delta_Eeta_drift_J'] += drift
            totals['eta_filter_delta_E_J'] += eta_filter
            totals['velocity_postmap_delta_K_J'] += postmap['delta_K_J']
            totals['kick_mask_delta_K_J'] += kick_mask['delta_K_J']
            totals['delta_Kbt_J'] += kinetic(field('u_out'),field('v_out'),bt_mass)-kinetic(field('u_in'),field('v_in'),bt_mass)
        base_u, base_v = fields['outer_4_u'], fields['outer_4_v']*arrays['interior_mask_z']
        initial_fast_wall = map_record(fields['outer_4_u'], fields['outer_4_v'], base_u, base_v, mass)
        lifted_u = base_u + (fields['fast_u_out'][-1]-fields['fast_u_in'][0])[...,None]*wet3
        lifted_v = base_v + (fields['fast_v_out'][-1]-fields['fast_v_in'][0])[...,None]*wet3
        lift_error = max(float(np.max(np.abs(lifted_u-fields['outer_5_u']))),float(np.max(np.abs(lifted_v-fields['outer_5_v']))))
        fast_reference_change = inventory[5]['Kref_J']-inventory[4]['Kref_J']
        fast_lift_residual = fast_reference_change-initial_fast_wall['delta_K_J']-totals['delta_Kbt_J']
        fast_lift_bound = len(fast)*bound(inventory[4]['Kref_J'],inventory[5]['Kref_J'],totals['delta_Kbt_J'])
        endpoint_delta = inventory[-1]['Kref_J']-inventory[0]['Kref_J']
        telescope_residual = endpoint_delta-sum(item['delta_K_J'] for item in stages)
        checks = dict(outer_midpoint_identities=all(item['identity_check'] for item in stages),
                      isolated_drag_wall_nonpositive=all(item.get('nonpositive_reference_energy_check',True) for item in stages),
                      fast_kick_increment_and_work=all(item['increment_check'] and item['work_reconstruction_check'] for item in fast),
                      fast_coriolis_zero_work=all(item['coriolis_zero_work_check'] for item in fast),
                      closed_fast_faces=all(item['closed_face_check'] for item in fast),
                      drift_barotropic_shear_reconstruction=all(item['drift_reconstruction_check'] for item in fast),
                      fast_3d_lift_check=bool(lift_error <= 1e-11 and abs(fast_lift_residual) <= fast_lift_bound),
                      fast_initial_wall_nonpositive=bool(initial_fast_wall['delta_K_J'] <= initial_fast_wall['arithmetic_bound_J']),
                      outer_telescope_check=bool(abs(telescope_residual) <= len(stages)*bound(endpoint_delta,*[item['delta_K_J'] for item in stages])))
        output['results'].append(dict(label=record['label'], equivalence=record['equivalence'], valid=record['valid'],
                                      projection_trace_calls=record['projection_trace_calls'], inventory=inventory, stages=stages,
                                      fast=fast, fast_totals=totals, fast_initial_wall=initial_fast_wall,
                                      fast_3d_lift_error_m_s=lift_error, fast_lift_residual_J=fast_lift_residual,
                                      fast_lift_bound_J=fast_lift_bound, outer_telescope_residual_J=telescope_residual,
                                      checks=checks, private_stage_archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                      missing=protocol['missing_fields'], density_conversion=protocol['density_PE']))
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(output,stream,indent=2,allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
