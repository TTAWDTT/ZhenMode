"""Compile the actual pinned FMS reader and check bounded means independently.

Manufactured NetCDF input, actual FMS IO/time modules; no ocean integration.
The caller must apply one CPU / 180s / 4GiB to the whole invocation.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import netCDF4
import numpy as np

from zhenmode.baselines.mom6.omip2_time import corrected_time_reader
from zhenmode.provenance.sources import load_json, sha256_file


def prepared_driver(original, native, output):
    """Six-hour first-window check against independent original CF records."""
    receipt, spatial = load_json(native/'time-inputs.json'), load_json(original/'preparation.json')
    if (receipt['status'] != 'prepared' or spatial['status'] != 'prepared_and_reader_verified'
            or receipt['input_preparation_sha256'] != sha256_file(original/'preparation.json')
            or receipt['dt_atmos_seconds'] != 3600 or receipt['dt_cpld_seconds'] != 3600
            or spatial['window_seconds'] != [-378691200, -378669600]):
        raise ValueError('check requires the verified 1958 first six-hour window and hourly clocks')
    from zhenmode.model.inputs.forcing.jra55 import FIELDS
    if set(receipt['files']) != set(FIELDS) or set(spatial['fields']) != set(FIELDS):
        raise ValueError('complete eleven-field input receipts required')
    for field, (variable, *_) in FIELDS.items():
        row = receipt['files'][field]
        if (row['sha256'] != sha256_file(native/row['path'])
                or row['original_sha256'] != sha256_file(original/(variable+'.nc'))
                or row['original_sha256'] != spatial['fields'][field]['prepared_sha256']):
            raise ValueError('prepared input identity mismatch: '+field)
    (output/'INPUT').symlink_to(native.resolve(), target_is_directory=True)
    return '''program real_inputs
use fms_mod, only: fms_init,fms_end
use time_manager_mod, only: set_calendar_type,GREGORIAN,set_date,set_time,operator(+)
use time_interp_external2_mod, only: init_external_field,time_interp_external,time_interp_external_init,get_external_field_size
implicit none
character(len=8) :: variables(11)=[character(len=8)::'uas','vas','tas','huss','psl','rsds','rlds','prra','prsn','friver','licalvf']
integer :: nf,k,index,siz(4),hour
real,allocatable :: data(:,:)
call fms_init()
call set_calendar_type(GREGORIAN)
call time_interp_external_init()
do nf=1,11
index=init_external_field('INPUT/'//trim(variables(nf))//'.nc',trim(variables(nf)),ongrid=.true.)
siz=get_external_field_size(index)
allocate(data(siz(1),siz(2)))
do k=0,5
hour=k+1
if(nf>=10) hour=k
call time_interp_external(index,set_date(1958,1,1,0,0,0)+set_time(hour*3600,0),data)
write(*,'(a,a,1x,i2,1x,4(es25.17,1x))') 'VALUE ',trim(variables(nf)),hour, &
data(1,1),data(siz(1)/2,siz(2)/2),data(siz(1),siz(2)),sum(data)
enddo
deallocate(data)
enddo
call fms_end()
end program
'''


def prepared_reference(original, native, output):
    """Do not use either the patched reader or its record-selection implementation."""
    from zhenmode.model.inputs.forcing.jra55 import FIELDS
    actual = {}
    for line in (output/'prepared.log').read_text().splitlines():
        if line.startswith('VALUE '):
            parts = line.split()
            key = (parts[1], int(parts[2]))
            if key in actual:
                raise ValueError('duplicate native sample')
            actual[key] = np.array([float(value) for value in parts[3:]])
    reference = {}
    for field, (variable, _, _, interpretation, _) in FIELDS.items():
        with netCDF4.Dataset(original/(variable+'.nc')) as ds:
            def seconds(values):
                dates = netCDF4.num2date(values, ds['time'].units, calendar=ds['time'].calendar)
                return np.asarray(netCDF4.date2num(dates, 'seconds since 1958-01-01', calendar='gregorian'))
            times = seconds(ds['time'][:])
            bounds = seconds(ds[ds['time'].bounds][:]) if interpretation == 'mean' else None
            for step in range(6):
                hour = step if field in {'runoff', 'calving'} else step+1
                query = hour*3600
                if interpretation == 'instant':
                    right = np.searchsorted(times, query, side='left')
                    if times[right] == query:
                        data = ds[variable][right]
                    else:
                        left = right-1
                        weight = (query-times[left])/(times[right]-times[left])
                        data = ds[variable][left]*(1-weight)+ds[variable][right]*weight
                else:
                    lower = query if field in {'runoff', 'calving'} else query-3600
                    indices = np.flatnonzero((bounds[:,0] <= lower) & (bounds[:,1] >= lower+3600))
                    if len(indices) != 1:
                        raise ValueError('independent CF reference lacks unique complete interval')
                    data = ds[variable][indices[0]]
                ny, nx = data.shape
                expected = np.array([data[0,0], data[ny//2-1,nx//2-1], data[-1,-1],
                                     np.sum(data, dtype=np.float64)])
                observed = actual[(variable,hour)]
                np.testing.assert_allclose(observed[:3], expected[:3], rtol=2e-13, atol=1e-12)
                np.testing.assert_allclose(observed[3], expected[3], rtol=2e-11, atol=1e-8)
                reference[f'{variable}:{hour}'] = {'expected':expected.tolist(), 'actual':observed.tolist()}
    if len(actual) != len(reference):
        raise ValueError('unexpected or missing native samples')
    return {'scope':'actual_FMS_prepared_11_field_6h_not_coupled_ocean',
            'data_kind':load_json(native/'time-inputs.json')['data_kind'],
            'comparisons':len(reference), 'reference':reference,
            'native_input_receipt_sha256':sha256_file(native/'time-inputs.json'),
            'original_input_receipt_sha256':sha256_file(original/'preparation.json')}


def fixture(path, phase, *, daily=False, wrong_units=False):
    with netCDF4.Dataset(path, 'w') as ds:
        ds.data_kind = 'manufactured'
        for axis, values, cart, units in [
            ('lon', [90, 270], 'X', 'degrees_east'),
            ('lat', [-45, 45], 'Y', 'degrees_north'),
        ]:
            ds.createDimension(axis, 2)
            var = ds.createVariable(axis, 'f8', (axis,))
            var[:] = values
            var.cartesian_axis, var.units = cart, units
        ds.createDimension('time', None)
        time = ds.createVariable('time', 'f8', ('time',))
        time.units = 'seconds since 1958-01-01 00:00:00'
        time.calendar, time.cartesian_axis = 'gregorian', 'T'
        time[:] = [43200] if daily else [5400, 16200]
        for name, values in [('average_T1', [0] if daily else [0, 10800]),
                             ('average_T2', [86400] if daily else [10800, 21600])]:
            bound = ds.createVariable(name, 'f8', ('time',))
            bound.units = 'hours since 1958-01-01 00:00:00' if wrong_units else time.units
            bound[:] = values
        var = ds.createVariable('rsds', 'f8', ('time', 'lat', 'lon'))
        var.units, var.cell_methods = 'W m-2', 'time: mean'
        var.time_avg_info = 'average_T1,average_T2'
        if phase:
            var.zhenmode_interval_sampling = phase
            var.zhenmode_interval_seconds = np.int32(3600)
        var[:] = np.array([7] if daily else [0, 400])[:, None, None]


def check(source, fms_build, fms_include, output, *, original=None, native=None):
    source, fms_build, fms_include, output = map(Path, (source, fms_build, fms_include, output))
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output/'time_interp_external2.F90').write_bytes(corrected_time_reader(source.read_bytes()))
        (output/'input.nml').write_text('&time_interp_external_nml\n/\n')
        for name, phase in [('right', 'right_endpoint'), ('left', 'left_endpoint'), ('linear', '')]:
            fixture(output/(name+'.nc'), phase)
        fixture(output/'daily.nc', 'left_endpoint', daily=True)
        fixture(output/'wrong_units.nc', 'right_endpoint', wrong_units=True)
        fixture(output/'legacy_epoch.nc', 'right_endpoint')
        with netCDF4.Dataset(output/'legacy_epoch.nc','a') as ds:
            for name in ('time','average_T1','average_T2'):
                ds[name][:] = ds[name][:] - 378691200
                ds[name].units = 'seconds since 1970-01-01 00:00:00'
        driver = '''program check_time
use fms_mod, only: fms_init, fms_end
use time_manager_mod, only: time_type,set_calendar_type,GREGORIAN,set_date,set_time,operator(+)
use time_interp_external2_mod, only: time_interp_external_init,init_external_field,time_interp_external
implicit none
character(len=32) :: mode
type(time_type) :: base
integer :: index,k
real :: data(2,2)
call get_command_argument(1,mode)
call fms_init()
call set_calendar_type(GREGORIAN)
call time_interp_external_init()
base=set_date(1958,1,1,0,0,0)
select case(trim(mode))
case('right','left')
index=init_external_field(trim(mode)//'.nc','rsds',ongrid=.true.)
do k=0,5
if(trim(mode)=='right') then
call time_interp_external(index,base+set_time((k+1)*3600,0),data)
else
call time_interp_external(index,base+set_time(k*3600,0),data)
endif
write(*,'(a,es25.17)') 'VALUE ',data(1,1)
enddo
case('linear')
index=init_external_field('linear.nc','rsds',ongrid=.true.)
call time_interp_external(index,base+set_time(10800,0),data)
write(*,'(a,es25.17)') 'VALUE ',data(1,1)
case('crossing','outside')
index=init_external_field('right.nc','rsds',ongrid=.true.)
k=0
if(trim(mode)=='crossing') k=12600
call time_interp_external(index,base+set_time(k,0),data)
case('daily')
index=init_external_field('daily.nc','rsds',ongrid=.true.)
call time_interp_external(index,base+set_time(82800,0),data)
write(*,'(a,es25.17)') 'VALUE ',data(1,1)
case('daily_outside')
index=init_external_field('daily.nc','rsds',ongrid=.true.)
call time_interp_external(index,base+set_time(86400,0),data)
case('wrong_units')
index=init_external_field('wrong_units.nc','rsds',ongrid=.true.)
case('legacy_epoch')
index=init_external_field('legacy_epoch.nc','rsds',ongrid=.true.)
end select
call fms_end()
end program
'''
        if (original is None) != (native is None):
            raise ValueError('original and native prepared inputs must be supplied together')
        if original is not None:
            original, native = Path(original), Path(native)
            driver = prepared_driver(original, native, output)
        (output/'driver.f90').write_text(driver)
        libraries = shlex.split(subprocess.check_output(['nf-config', '--flibs'], text=True))
        command = ['mpif90', '-cpp', '-O0', '-ffree-line-length-none', '-fdefault-real-8',
                   '-fdefault-double-8', '-I'+str(fms_build.resolve()),
                   '-I'+str(fms_include.resolve()), '-I/usr/include', 'time_interp_external2.F90',
                   'driver.f90', str((fms_build/'libFMS.a').resolve()), *libraries, '-o', 'check_time']
        compilation = subprocess.run(command, cwd=output, text=True, capture_output=True, timeout=60)
        (output/'compile.txt').write_text(compilation.stdout+compilation.stderr)
        compilation.check_returncode()
        expected = {'right': [0,0,0,400,400,400], 'left': [0,0,0,400,400,400],
                    'linear': [200], 'daily': [7]}
        controls = {'crossing': 'crosses bounds', 'outside': 'crosses bounds',
                    'daily_outside': 'crosses bounds', 'wrong_units': 'bound units disagree',
                    'legacy_epoch':'time is negative'}
        if original is not None:
            expected, controls = {}, {}
            result = subprocess.run([str((output/'check_time').resolve())], cwd=output,
                                    text=True, capture_output=True, timeout=60)
            (output/'prepared.log').write_text(result.stdout+result.stderr)
            result.check_returncode()
        cases = {}
        for mode in expected | controls:
            result = subprocess.run([str((output/'check_time').resolve()), mode], cwd=output,
                                    text=True, capture_output=True, timeout=15)
            (output/(mode+'.log')).write_text(result.stdout+result.stderr)
            values = [float(line.split()[1]) for line in result.stdout.splitlines() if line.startswith('VALUE ')]
            if mode in expected:
                result.check_returncode()
                np.testing.assert_array_equal(values, expected[mode])
            elif result.returncode == 0 or controls[mode] not in result.stdout+result.stderr:
                raise AssertionError('native negative control not independently rejected: '+mode)
            cases[mode] = {'exit_code': result.returncode, 'values': values}
        report = {'status': 'PASS', 'scope': 'manufactured_actual_FMS_IO_not_coupled_ocean',
                  'upstream_sha256': sha256_file(source),
                  'adapted_sha256': sha256_file(output/'time_interp_external2.F90'),
                  'adapter_sha256': sha256_file(__import__('zhenmode.baselines.mom6.omip2_time',fromlist=['']).__file__),
                  'harness_sha256': sha256_file(__file__), 'driver_sha256': sha256_file(output/'driver.f90'),
                  'fms_archive_sha256': sha256_file(fms_build/'libFMS.a'),
                  'header_sha256': sha256_file(fms_include/'file_version.h'),
                  'binary_sha256': sha256_file(output/'check_time'), 'command': command, 'cases': cases,
                  'expected_six_hour_energy_J_m2': 4320000,
                  'actual_six_hour_energy_J_m2': sum(cases['right']['values'])*3600 if original is None else None,
                  'execution_ready': False, 'climate_qualification': False}
        if original is not None:
            report.pop('expected_six_hour_energy_J_m2')
            report.pop('actual_six_hour_energy_J_m2')
            report.update(prepared_reference(original, native, output))
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'status':'FAIL','reason':str(error)},indent=2))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','fms-build','fms-include','output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--original-prepared')
    parser.add_argument('--native-prepared')
    args = parser.parse_args()
    result = check(args.source,args.fms_build,args.fms_include,args.output,
                   original=args.original_prepared,native=args.native_prepared)
    print(json.dumps({'status':result['status'],'scope':result['scope']}))
