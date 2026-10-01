"""Scalar actual N two-RHS operator accounting; no fitted residual or PE closure."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from summarize_mechanical_probe import unpack

FACTOR = 4096 * np.finfo(np.float64).eps


def component_work(first, second, u0, v0, un, vn, mass, dt):
    """Use one actual N endpoint midpoint for all stage/component work."""
    midpoint = (.5*(u0+un), .5*(v0+vn))
    result = {}
    for name in first:
        works, absolute, impulses = [], 0., np.zeros(2)
        for terms in (first, second):
            du, dv = (.5*dt*item for item in terms[name])
            x, y = mass*midpoint[0]*du, mass*midpoint[1]*dv
            works.append(float(np.sum(x+y)))
            absolute += float(np.sum(np.abs(x)+np.abs(y)))
            impulses += [np.sum(mass*du), np.sum(mass*dv)]
        result[name] = dict(work_J=sum(works), first_rhs_work_J=works[0],
                            second_rhs_work_J=works[1], absolute_products_J=absolute,
                            component_impulse_kg_m_s=impulses.tolist())
    return result


def rhs_terms(fields, label, state, arrays, scalars):
    """Reconstruct only recorded locals and explicitly observed original maps."""
    wet = arrays['wet_mask_z']

    def pair(key):
        return tuple(fields[f'{label}_meta_{key}_{axis}'] for axis in (0, 1))

    def masked(values):
        return tuple(value*wet for value in values)

    raw = masked(pair('terms_advection_raw'))
    filtered = masked(pair('terms_advection_filtered'))
    terms = dict(advection_raw=raw, advection_dealias_difference=tuple(b-a for a,b in zip(raw,filtered)))
    for public, recorded in [('pressure_full','pressure_full'), ('wind_surface','wind_surface'),
                             ('horizontal_diffusion_full','horizontal_diffusion'),
                             ('vertical_diffusion_full','vertical_diffusion'),
                             ('coriolis_full','coriolis'), ('bottom_drag_full','bottom_drag')]:
        terms[public] = masked(pair('terms_'+recorded))
    for name in ('horizontal_diffusion','vertical_diffusion','coriolis'):
        terms[name+'_removal'] = tuple(-item for item in pair('terms_'+name))
    terms['eta_mean_removal'] = tuple(9.81*item[...,None] for item in pair('eta_mean_gradient'))
    terms['density_mean_removal'] = tuple(-item[...,None] for item in pair('density_mean'))
    terms['wind_mean_removal'] = tuple(-item[...,None] for item in pair('wind_mean'))
    # This is the observed final masking map, not a residual fitted after summing terms.
    terms['final_residual_mask'] = tuple(a-b for a,b in zip(pair('residual'),pair('after_wind')))
    bottom = arrays['bottom_mask']*wet
    terms['bottom_drag_compensation'] = tuple(scalars['r_bot']*fields[f'{state}_{key}']*bottom for key in ('u','v'))
    return terms, pair


def require_accepted_record(record):
    if (record['valid'] is not True or record['equivalence']['passed'] is not True
            or not record['checks'] or any(value is False for value in record['checks'].values()
                                           if isinstance(value,bool))):
        raise ValueError('original acceptance or paired equivalence gate failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('archive_dir','probe_dir','protocol','original_protocol','output'):
        parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    args=parser.parse_args()
    protocol_bytes=args.protocol.read_bytes()
    protocol=json.loads(protocol_bytes)
    original_bytes=args.original_protocol.read_bytes()
    if hashlib.sha256(original_bytes).hexdigest()!=protocol['original_protocol_sha256']:
        raise ValueError('original protocol changed')
    original=json.loads(original_bytes)
    contents={}
    for name,expected in original['base_inputs'].items():
        contents[name]=(args.archive_dir/name).read_bytes()
        if hashlib.sha256(contents[name]).hexdigest()!=expected:
            raise ValueError('original input changed: '+name)
    from io import BytesIO
    arrays=unpack(BytesIO(contents['parameter_arrays_used.npz']))
    scalars=json.loads(contents['manifest_start.json'])['scalar_params']
    resources=json.loads((args.probe_dir/'resources.json').read_text(encoding='utf-8-sig'))
    digest=hashlib.sha256(protocol_bytes).hexdigest()
    if any(item['protocol_sha256']!=digest or item['exit_code']!=0 or item['stop_reason'] for item in resources):
        raise ValueError('resource/protocol witness failed')
    record=json.loads((args.probe_dir/'instrumented/records.json').read_text())[0]
    require_accepted_record(record)
    fields=unpack(args.probe_dir/'instrumented/6h-stages-private.npz')
    if not all(np.all(np.isfinite(value)) for value in fields.values()):
        raise ValueError('nonfinite observation')
    wet=arrays['wet_mask_z']
    if not np.all((wet==0)|(wet==1)):
        raise ValueError('nonbinary wet mask')
    h0=arrays['dz_node']*wet
    height=h0.sum(axis=-1)
    normal=np.divide(h0,height[...,None],out=np.zeros_like(h0),where=height[...,None]>0)
    norm_error=float(np.max(np.abs(normal-arrays['dz_norm'])))
    height_error=float(np.max(np.abs(height-arrays['H_sw'])[arrays['wet_mask']>0]))
    if norm_error>FACTOR or height_error>FACTOR*max(1.,float(height.max())):
        raise ValueError('reference mass contract failed')
    area=arrays['dx_2d']*scalars['dy']
    mass=1025.*area[...,None]*h0
    dt=scalars['dt']
    u0,v0,un,vn=[fields[key] for key in ('state0_u','state0_v','state_out_u','state_out_v')]
    before=float(.5*np.sum(mass*(u0*u0+v0*v0)))
    after=float(.5*np.sum(mass*(un*un+vn*vn)))
    checks=[]

    def check(name,left,right):
        left,right=np.broadcast_arrays(left,right)
        error=float(np.max(np.abs(left-right)))
        scale=max(1.,float(np.max(np.abs(left))),float(np.max(np.abs(right))))
        tolerance=FACTOR*scale
        checks.append(dict(name=name,max_absolute_error=error,bound=tolerance,passed=error<=tolerance))

    tables=[]
    for label,state in [('first','state0'),('second','state_euler')]:
        terms,pair=rhs_terms(fields,label,state,arrays,scalars)
        if set(terms)!=set(protocol['terms']):
            raise ValueError('term contract changed')
        tables.append(terms)
        full_names=['advection_raw','advection_dealias_difference','pressure_full','wind_surface',
                    'horizontal_diffusion_full','vertical_diffusion_full','coriolis_full','bottom_drag_full']
        prior=pair('full')
        for axis in (0,1):
            check(label+'_full_'+str(axis),sum(terms[name][axis] for name in full_names),prior[axis])
        sequence=[('after_horizontal','horizontal_diffusion_removal'),('after_vertical','vertical_diffusion_removal'),
                  ('after_coriolis','coriolis_removal'),('after_eta','eta_mean_removal'),
                  ('after_density','density_mean_removal'),('after_wind','wind_mean_removal'),
                  ('residual','final_residual_mask')]
        for observed,term in sequence:
            current=pair(observed)
            for axis in (0,1):
                check(label+'_'+observed+'_'+str(axis),prior[axis]+terms[term][axis],current[axis])
            prior=current
        for axis in (0,1):
            actual=fields[f'{label}_rhs_{axis}']
            check(label+'_drag_compensation_'+str(axis),pair('residual')[axis]+terms['bottom_drag_compensation'][axis],actual)
            check(label+'_all_terms_rhs_'+str(axis),sum(value[axis] for value in terms.values()),actual)
    for axis,key in enumerate(('u','v')):
        check('Euler_'+key,fields['state0_'+key]+dt*fields[f'first_rhs_{axis}'],fields['state_euler_'+key])
        check('N_RK_'+key,fields['state0_'+key]+.5*dt*(fields[f'first_rhs_{axis}']+fields[f'second_rhs_{axis}']),fields['state_out_'+key])
        reconstructed=.5*dt*sum(tables[0][name][axis]+tables[1][name][axis] for name in tables[0])
        check('N_increment_'+key,reconstructed,fields['state_out_'+key]-fields['state0_'+key])
    for key in ('T','S'):
        check('second_actual_tracer_'+key,fields['state_euler_'+key],fields['tracer_'+key])
    check('N_fixed_eta',fields['state_out_eta'],fields['state0_eta'])
    works=component_work(*tables,u0,v0,un,vn,mass,dt)
    delta=after-before
    total=sum(item['work_J'] for item in works.values())
    tolerance=FACTOR*max(1.,abs(before)+abs(after)+sum(item['absolute_products_J'] for item in works.values()))
    identity=dict(delta_Kref_J=delta,component_work_sum_J=total,residual_J=delta-total,bound_J=tolerance,passed=abs(delta-total)<=tolerance)
    groups={
        'pressure_net':['pressure_full','eta_mean_removal','density_mean_removal'],
        'advection_net':['advection_raw','advection_dealias_difference'],
        'wind_net':['wind_surface','wind_mean_removal'],
        'horizontal_cancel':['horizontal_diffusion_full','horizontal_diffusion_removal'],
        'vertical_cancel':['vertical_diffusion_full','vertical_diffusion_removal'],
        'coriolis_cancel':['coriolis_full','coriolis_removal'],
        'bottom_cancel':['bottom_drag_full','bottom_drag_compensation'],
        'final_residual_mask':['final_residual_mask']}
    grouped={name:dict(work_J=sum(works[item]['work_J'] for item in names),members=names) for name,names in groups.items()}
    for name in ('horizontal_cancel','vertical_cancel','coriolis_cancel','bottom_cancel','final_residual_mask'):
        grouped[name]['zero_work_bound_J']=tolerance
        grouped[name]['zero_work_passed']=abs(grouped[name]['work_J'])<=tolerance
    geometry={}
    for label in ('state0','state_out'):
        geometry[label+'_Kmat_minus_Kref_J']=float(.5*1025.*np.sum(area*arrays['wet_mask']*fields[label+'_eta']*(fields[label+'_u'][...,0]**2+fields[label+'_v'][...,0]**2)))
    geometry['change_J']=geometry['state_out_Kmat_minus_Kref_J']-geometry['state0_Kmat_minus_Kref_J']
    output=dict(schema='ocean.n_rhs_evidence.v1',protocol_sha256=digest,record=record,resources=resources,
                reference_norm_error=norm_error,reference_height_error_m=height_error,
                second_TS_change_from_first={key:float(np.max(np.abs(fields['state_euler_'+key]-fields['state0_'+key]))) for key in ('T','S')},
                first_Kref_J=before,final_Kref_J=after,work_identity=identity,checks=checks,terms=works,groups=grouped,
                geometry=geometry,all_gates_passed=bool(identity['passed'] and all(item['passed'] for item in checks) and all(item.get('zero_work_passed',True) for item in grouped.values())),
                scope='Actual N operator accounting only; no compatible buoyancy PE or full mechanical closure; no repair.')
    with args.output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(output,indent=2,allow_nan=False)+'\n')
    if not output['all_gates_passed']:
        raise ValueError('N reconstruction gate failed; do not infer sources')
    print(json.dumps(dict(work_identity=identity,groups=grouped,geometry=geometry),allow_nan=False))


if __name__=='__main__':
    main()
