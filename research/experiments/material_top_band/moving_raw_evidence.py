"""Source-bound receipts of actual manufactured moving raw commits."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np

from .moving_raw_cases import manufactured_mean_case
from .moving_raw_characteristic import MovingRawSlice
from .moving_raw_oracle import audit_receipt

ROOT = Path(__file__).resolve().parents[3]
CHANNELS = ('water_m3','IT_temperature_m3','IS_salinity_m3','Mu_kg_m_s','Mv_kg_m_s','physical_KE_J','gravity_PE_J')


def case_evidence(parameters, durations):
    profile,spec = manufactured_mean_case(**parameters)
    integrator = MovingRawSlice(profile,authority='manufactured_half_prism_raw_means')
    rows = []
    for dt in durations:
        before = integrator.profile
        receipt = integrator.step(dt)
        independent = audit_receipt(spec,receipt)
        after = receipt.profile.state
        horizontal,vertical = receipt.faces['horizontal'],receipt.faces['vertical']
        stock_change = after.stocks-before.state.stocks
        area = spec.distance*spec.length/2.
        rows.append(dict(duration_s=dt,time_s=receipt.time_s,accepted_raw_steps=receipt.accepted_raw_steps,
                         independent_audit=independent,eta_change_m=float(after.eta[0]-before.state.eta[0]),
                         maximum_actual_h_change_m=float(np.max(abs(after.h-before.state.h))),
                         maximum_actual_deep_IS_change=float(np.max(abs(stock_change[:,3:,1]))),
                         maximum_actual_deep_Mu_change=float(np.max(abs(stock_change[:,3:,2]))),
                         actual_global_stock_change=[area*math.fsum(channel.ravel()) for channel in np.moveaxis(stock_change,-1,0)],
                         shared_horizontal_segment_count=len(horizontal),vertical_face_count=len(vertical),
                         maximum_shared_water_transport_m3=max(abs(row['transport'][1,0]) for row in horizontal),
                         maximum_internal_R_water_transport_m3=max(abs(row['transport'][0]) for row in vertical if 0 < row['interface'] < 14),
                         maximum_internal_pressure_energy_J=max(abs(row['pressure_energy']) for row in vertical if 0 < row['interface'] < 14),
                         moving_surface_pressure_energy_J=math.fsum(row['pressure_energy'] for row in vertical if row['interface'] == 0),
                         M_change_norm=float(np.linalg.norm(receipt.M_after-receipt.M_before)),R_change_norm=float(np.linalg.norm(receipt.R_after-receipt.R_before)),
                         physical_KE_change_J=receipt.physical_KE_change_J,raw_KE_change_J=receipt.raw_KE_change_J,PE_change_J=receipt.PE_change_J,
                         covariance_change_J=receipt.covariance_change_J,true_pressure_work_J=receipt.true_pressure_work_J,
                         midpoint_pressure_work_J=receipt.midpoint_pressure_work_J,midpoint_minus_true_work_J=receipt.midpoint_pressure_work_J-receipt.true_pressure_work_J,
                         pressure_work_error_bound_J=receipt.work_error_bound_J,raw_mass_chain=receipt.raw_mass_chain,physical_mass_chain=receipt.physical_mass_chain,
                         maximum_local_gcl_residual_m3=float(np.max(abs(receipt.gcl_residual))),global_gcl_residual_m3=math.fsum(receipt.gcl_residual.ravel()),
                         maximum_cumulative_gcl_residual_m3=float(np.max(abs(receipt.cumulative_gcl))),
                         maximum_face_bound_ratio=receipt.maximum_face_bound_ratio,
                         maximum_time_numerator_degree=max(int(np.max(row[key+'_degree'])) for row in horizontal for key in ('transport','pressure','pressure_energy')),
                         maximum_time_lambda_power=max(int(np.max(row[key+'_power'])) for row in horizontal for key in ('transport','pressure','pressure_energy')),
                         maximum_face_truncation_bounds={name:max(float(np.max(row['transport_truncation'][...,channel])) for row in [*horizontal,*vertical]) for channel,name in enumerate(CHANNELS)},
                         maximum_face_input_reconstruction_bounds={name:max(float(np.max(row['transport_input'][...,channel])) for row in [*horizontal,*vertical]) for channel,name in enumerate(CHANNELS)},
                         accepted_moving_geometry=receipt.accepted_moving_geometry))
    return dict(parameters=parameters,steps=rows,accepted_raw_steps=integrator.accepted_raw_steps,
                accepted_moving_steps=sum(int(row['accepted_moving_geometry']) for row in rows))


def provenance():
    from ocean_solver.provenance.archives import current_source_files
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('committed clean scientific source required')
    selected = list((ROOT/'research/experiments/material_top_band').glob('*.py'))
    selected += [ROOT/'docs/moving_raw_protocol.json',ROOT/'scripts/run_bounded_research_tests.py',ROOT/'research/experiments/material_top_band/affine_requirements.lock']
    selected += [ROOT/'tests/research/contracts'/name for name in ('test_raw_mean_geometry.py','test_rational_time_integral.py','test_moving_raw_characteristic.py')]
    files = current_source_files(ROOT,selected)
    return dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                source_sha256={label:hashlib.sha256(path.read_bytes()).hexdigest() for label,path in sorted(files.items())},
                Python=platform.python_version(),package_versions={name:importlib.metadata.version(name) for name in ('numpy','pytest','ruff')})


def build_evidence():
    protocol = json.loads((ROOT/'docs/moving_raw_protocol.json').read_text(encoding='utf-8-sig'))
    if protocol['frozen_thresholds']['roundoff_eps_multiplier'] != 512 or protocol['qualification_passed']:
        raise ValueError('frozen gate or qualification boundary changed')
    cases = dict(positive=case_evidence({},(.02,.03)),negative=case_evidence(dict(U=-.03,alpha=-.08,external=(200.,80.)),(.02,.03)),
                 zero_pressure=case_evidence(dict(external=(80.,80.)),(.02,)),fixed_eta=case_evidence(dict(alpha=0.),(.02,)))
    return dict(contract=protocol['contract'],restricted_manufactured_raw_mean_ALE_passed=True,
                accepted_raw_steps=sum(case['accepted_raw_steps'] for case in cases.values()),
                accepted_moving_steps=sum(case['accepted_moving_steps'] for case in cases.values()),cases=cases,qualification_passed=False,original_global_CV_identified=False,
                production_force_consumption_qualified=False,real_archive_steps=0,
                numerical_scope='Canonical affine characteristic FV reconstruction qualified against actual raw means/P1 endpoints. First-slot ALE only. No original CV/top-three band/predict12fastreplay, source adapters, generic order or speed qualification.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    arguments = parser.parse_args()
    source = provenance()
    result = dict(build_evidence(),provenance=source)
    text = json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if arguments.output:
        arguments.output.write_text(text,encoding='utf-8')
    else:
        print(text,end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
