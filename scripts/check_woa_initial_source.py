"""Check every original WOA support mask and reserved cells against pinned GSW.

Use the separately compiled check_teos10 --temperatures oracle directory. This
checks original-grid conversion, not native remapping or a global ocean score.
"""
import argparse
import json
import subprocess
from pathlib import Path

import netCDF4
import numpy as np

from zhenmode.provenance.sources import load_json, sha256_file

if __package__:
    from .check_teos10 import GSW_COMMIT, SOURCES, TEMPERATURE_SOURCES
else:
    from check_teos10 import GSW_COMMIT, SOURCES, TEMPERATURE_SOURCES


def checked(path, row):
    path=Path(path)
    if path.stat().st_size!=row['bytes'] or sha256_file(path)!=row['sha256']:
        raise ValueError('actual input identity mismatch: '+str(path))
    return path


def check(prepared,acquisition,pressure_reference,oracle,output):
    prepared,acquisition,pressure_reference,oracle,output=map(Path,(prepared,acquisition,pressure_reference,oracle,output))
    preparation_sha=sha256_file(prepared/'initialization.json')
    report=load_json(prepared/'initialization.json')
    if (report['status']!='original_grid_thermodynamics_prepared'
            or sha256_file(acquisition)!=report['acquisition_sha256']
            or sha256_file(pressure_reference)!=report['pressure_receipt_sha256']):
        raise ValueError('original-grid preparation receipt mismatch')
    original=load_json(acquisition)
    rows={row['role']:row for row in original['files'] if row['role'] in {'temperature','salinity'}
          and row['climatology_month']==0}
    paths={name:checked(acquisition.parent/row['path'],row) for name,row in rows.items()}
    pinfo=load_json(pressure_reference)
    ppath=checked(pressure_reference.parent/pinfo['path'],pinfo)
    transformed=checked(prepared/report['output']['path'],report['output'])
    reference=load_json(oracle/'report.json')
    if (reference.get('status')!='PASS' or reference['gsw_commit']!=GSW_COMMIT
            or reference['source_sha256']!=SOURCES|TEMPERATURE_SOURCES
            or sha256_file(oracle/'reference')!=reference['binary_sha256']
            or sha256_file(oracle/'driver.f90')!=reference['driver_sha256']):
        raise ValueError('pinned original GSW oracle identity mismatch')
    for name,expected in reference['source_sha256'].items():
        if sha256_file(oracle/Path(name).name)!=expected:
            raise ValueError('original compiled GSW source changed: '+name)
    output.mkdir(parents=True,exist_ok=False)
    try:
        inputs,actual,indices=[],[],[]
        total_masks=total_valid=0
        rng=np.random.default_rng(20261006)
        with netCDF4.Dataset(paths['temperature']) as tds, netCDF4.Dataset(paths['salinity']) as sds, \
                netCDF4.Dataset(ppath) as pds, netCDF4.Dataset(transformed) as native:
            for name in ('lon','lat','depth'):
                np.testing.assert_array_equal(tds[name][:],native[name][:])
            np.testing.assert_array_equal(tds['time'][:],native['time'][:])
            np.testing.assert_array_equal(tds['time'][:],sds['time'][:])
            for original_ds in (tds,sds):
                original_time=original_ds['time']
                if original_time.units != native['time'].units:
                    raise ValueError('converted time units disagree with original climatology')
                original_bounds=original_time.climatology
                native_bounds=native['time'].climatology
                np.testing.assert_array_equal(original_ds[original_bounds][:],native[native_bounds][:])
            pressure=np.asarray(pds[pinfo['variable']][:])
            for k in range(len(tds['depth'])):
                t=np.ma.asarray(tds['t_an'][0,k],dtype=float).filled(np.nan)
                s=np.ma.asarray(sds['s_an'][0,k],dtype=float).filled(np.nan)
                valid=np.isfinite(t)&np.isfinite(s)
                np.testing.assert_array_equal(native['paired_source_support'][0,k],valid.astype(np.int8))
                for name in ('ptemp','ct','sr'):
                    data=native[name][0,k]
                    np.testing.assert_array_equal(np.ma.getmaskarray(data),~valid)
                    if not np.isfinite(data.data[valid]).all():
                        raise ValueError('converted field has nonfinite supported values')
                total_masks+=valid.size
                total_valid+=int(valid.sum())
                supported=np.flatnonzero(valid)
                chosen=np.sort(rng.choice(supported,size=min(8,len(supported)),replace=False))
                for flat in chosen:
                    j,i=np.unravel_index(flat,valid.shape)
                    inputs.append([float(s[j,i]),float(t[j,i]),float(pressure[k,j])])
                    actual.append([float(native[name][0,k,j,i]) for name in ('sr','ct','ptemp')])
                    indices.append([k,int(j),int(i)])
        if not inputs:
            raise ValueError('no paired original WOA support to check')
        def evaluate(values,name):
            text='\n'.join(' '.join(f'{value:.17e}' for value in row) for row in values)+'\n'
            (output/(name+'-input.txt')).write_text(text)
            result=subprocess.run([str((oracle/'reference').resolve())],input=text,
                                   capture_output=True,text=True,timeout=60,cwd=output)
            (output/(name+'-output.txt')).write_text(result.stdout)
            (output/(name+'-stderr.txt')).write_text(result.stderr)
            result.check_returncode()
            data=np.array([[float(value) for value in row.split()] for row in result.stdout.splitlines()])
            if data.shape!=(len(values),15) or not np.isfinite(data).all():
                raise ValueError('original Fortran oracle returned incomplete values')
            return data
        # First route: execute original gsw_sr_from_sp on raw SP numbers.
        # Other columns of this call are irrelevant and are not used as SA.
        salt=evaluate(inputs,'practical-salinity')[:,4]
        thermo_inputs=[[float(sr),row[1],row[2]] for sr,row in zip(salt,inputs,strict=True)]
        thermo=evaluate(thermo_inputs,'reference-thermodynamics')
        expected=np.column_stack((salt,thermo[:,8],thermo[:,9]))
        actual=np.asarray(actual)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-9)
        damaged=expected.copy()
        damaged[0,1]+=.01
        try:
            np.testing.assert_allclose(actual,damaged,rtol=0,atol=1e-9)
        except AssertionError:
            pass
        else:
            raise AssertionError('conversion comparator did not reject planted 0.01C error')
        # A PASS must refer to the frozen inputs actually inspected, even if
        # a writer changes an unsampled value during the Fortran comparison.
        for role,path in paths.items():
            checked(path,rows[role])
        checked(ppath,pinfo)
        checked(transformed,report['output'])
        if (sha256_file(prepared/'initialization.json') != preparation_sha
                or sha256_file(acquisition) != report['acquisition_sha256']
                or sha256_file(pressure_reference) != report['pressure_receipt_sha256']
                or sha256_file(oracle/'reference') != reference['binary_sha256']):
            raise ValueError('checked receipt or oracle changed during comparison')
        result={'status':'PASS','scope':'original_grid_support_and_reserved_GSW_conversion_cells_not_native_ocean',
            'data_kind':report['data_kind'],'mask_cells_checked':total_masks,'paired_original_cells':total_valid,
            'reserved_conversion_cells':len(inputs),'random_selection_seed':20261006,'indices':indices,
            'maximum_absolute_errors':dict(zip(('SR_g_kg','CT_degC','pt_degC'),
                np.max(np.abs(actual-expected),axis=0).tolist(),strict=True)),
            'comparison_absolute_tolerance':1e-9,'planted_mismatch_rejected':True,
            'preparation_receipt_sha256':preparation_sha,
            'original_file_sha256':{name:sha256_file(path) for name,path in paths.items()},
            'pressure_sha256':sha256_file(ppath),'converted_file_sha256':sha256_file(transformed),
            'oracle_receipt_sha256':sha256_file(oracle/'report.json'),'oracle_binary_sha256':reference['binary_sha256'],
            'harness_sha256':sha256_file(__file__),
            'oracle_pin_definition_sha256':sha256_file(Path(__file__).with_name('check_teos10.py')),
            'independence_scope':'original Fortran evaluation checks JAX conversion/writing; pressure input and mathematical definitions are shared',
            'native_mapping_verified':False,'pressure_producer_lineage_verified':False,
            'execution_ready':False,'climate_qualification':False,'ocean_observation_score':None}
        (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return result
    except (Exception,KeyboardInterrupt) as error:
        (output/'failure.json').write_text(json.dumps({'status':'FAIL','reason_type':type(error).__name__,
                                                     'reason':str(error)},indent=2)+'\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','acquisition','pressure-reference','oracle','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    result=check(args.prepared,args.acquisition,args.pressure_reference,args.oracle,args.output)
    print(json.dumps({key:result[key] for key in ('status','mask_cells_checked','reserved_conversion_cells','maximum_absolute_errors')}))
