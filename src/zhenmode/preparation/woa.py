"""Original-grid WOA thermodynamic preparation, separate from native interpolation.

Keep missing support and source identities. This exports potential temperature
for MOM and CT/SR for the explicit reference variant, not a complete ocean case.
Pressure is an identity-bound input with a declared definition, never a layer index.
"""
from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import netCDF4
import numpy as np

from zhenmode.model.solver.numerics.backend import jax, jnp
from zhenmode.model.solver.physics.teos10 import (
    conservative_from_in_situ,
    potential_from_in_situ,
    reference_salinity,
    validate_state,
)
from zhenmode.provenance.sources import (
    checked_file,
    load_json,
    package_source_hashes,
    sha256_file,
)


def _coordinate(ds, name, units):
    variable=ds[name]
    values=variable[:]
    if (variable.dimensions!=(name,) or variable.units!=units or np.ma.is_masked(values)
            or values.ndim!=1 or not len(values) or not np.isfinite(values).all()
            or np.any(np.diff(values)<=0)):
        raise ValueError('invalid original WOA coordinate/units: '+name)
    return np.asarray(values,dtype=float)


def _annual_time(ds):
    """Retain WOA climatology months; do not reinterpret year zero as a civil date."""
    time = ds['time']
    values = np.ma.asarray(time[:])
    bound_name = getattr(time, 'climatology', None)
    if (time.dimensions != ('time',) or values.shape != (1,)
            or np.ma.is_masked(values) or not np.isfinite(values).all()
            or getattr(time, 'units', None) != 'months since 0000-01-01 00:00:00'
            or bound_name not in ds.variables):
        raise ValueError('original WOA annual climatology requires finite time and bounds')
    bounds = ds[bound_name]
    intervals = np.ma.asarray(bounds[:])
    if (bounds.dimensions[:1] != ('time',) or intervals.shape != (1, 2)
            or np.ma.is_masked(intervals) or not np.isfinite(intervals).all()
            or not np.array_equal(intervals, [[0., 12.]])
            or getattr(bounds, 'units', time.units) != time.units
            or not 0. <= values[0] <= 12.):
        raise ValueError('original WOA annual climatology must cover months 0 to 12')
    return {'values': np.asarray(values).tolist(), 'bounds': np.asarray(intervals).tolist(),
            'units': time.units, 'calendar': getattr(time, 'calendar', None)}


def prepare_woa_thermodynamics(acquisition, pressure_reference, output):
    """Stream paired annual WOA13v2 levels, preserve holes, and export PT/CT/SR.

    Pressure JSON: path/variable/bytes/sha256/units/definition. Its NetCDF must
    contain matching depth/lat coordinates and p[depth,lat] in dbar. The output
    is original-grid source data; native wet masks, infill, remapping, vertical
    extrapolation, initial volumes and execution eligibility are separate gates.
    """
    acquisition,pressure_reference,output=map(Path,(acquisition,pressure_reference,output))
    acquisition_sha = sha256_file(acquisition)
    pressure_receipt_sha = sha256_file(pressure_reference)
    origin=load_json(acquisition)
    pressure_info=load_json(pressure_reference)
    if (sha256_file(acquisition) != acquisition_sha
            or sha256_file(pressure_reference) != pressure_receipt_sha):
        raise ValueError('initial source receipt changed during reading')
    if origin.get('kind')!='WOA13v2_originals' or origin.get('status')!='completed':
        raise ValueError('complete WOA13v2 original acquisition receipt required')
    if (set(pressure_info)!={'path','variable','bytes','sha256','units','definition'}
            or pressure_info['units']!='dbar' or not isinstance(pressure_info['definition'],str)
            or not pressure_info['definition'].strip()):
        raise ValueError('pressure reference requires exact fields, dbar units and explicit definition')
    chosen={}
    for row in origin['files']:
        if row.get('climatology_month')==0 and row.get('role') in {'temperature','salinity'}:
            if row['role'] in chosen:
                raise ValueError('ambiguous annual WOA role')
            chosen[row['role']]=row
    if set(chosen)!={'temperature','salinity'}:
        raise ValueError('annual original WOA temperature and salinity are required')
    paths={name:checked_file(acquisition.parent,row) for name,row in chosen.items()}
    pressure_path=checked_file(pressure_reference.parent,pressure_info)
    output.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'status':'running','scope':'WOA_original_grid_not_native_initialization',
            'acquisition_sha256':acquisition_sha,'pressure_receipt_sha256':pressure_receipt_sha,
            'pressure_definition':pressure_info['definition'],'source_files':chosen,
            'pressure_producer_lineage_verified':False,
            'pressure_sha256':pressure_info['sha256'],'missing_support_policy':'preserve_paired_original_mask',
            'salinity_definition':'SR_approximates_SA_no_geographic_anomaly',
            'native_initialization_ready':False,'execution_ready':False,'climate_qualification':False,
            'package_source_sha256':package_source_hashes(__file__),
            'levels':[]}
    try:
        with ExitStack() as stack:
            ds={name:stack.enter_context(netCDF4.Dataset(path)) for name,path in paths.items()}
            pressure_ds=stack.enter_context(netCDF4.Dataset(pressure_path))
            temperature,salinity=ds['temperature'],ds['salinity']
            time_definition = _annual_time(temperature)
            if _annual_time(salinity) != time_definition:
                raise ValueError('WOA temperature/salinity annual times differ')
            report['source_time_definition'] = time_definition
            coordinates={name:_coordinate(temperature,name,unit) for name,unit in (
                ('lon','degrees_east'),('lat','degrees_north'),('depth','meters'))}
            if (coordinates['depth'][0]!=0 or np.any(coordinates['depth']<0)
                    or np.any(np.abs(coordinates['lat'])>90)
                    or coordinates['lon'][-1]-coordinates['lon'][0]>=360):
                raise ValueError('WOA source depth/latitude/periodic longitude is invalid')
            for name,unit in [('lon','degrees_east'),('lat','degrees_north'),('depth','meters')]:
                if not np.array_equal(_coordinate(salinity,name,unit),coordinates[name]):
                    raise ValueError('WOA temperature/salinity coordinates differ: '+name)
            for name,unit in [('lat','degrees_north'),('depth','meters')]:
                if not np.array_equal(_coordinate(pressure_ds,name,unit),coordinates[name]):
                    raise ValueError('pressure coordinates differ from WOA: '+name)
            manufactured=any(getattr(item,'data_kind',None)=='manufactured' for item in ds.values())
            report['data_kind']='manufactured' if manufactured else 'observational_source_transform'
            for role,variable,units in [('temperature','t_an',('degrees_celsius',)),('salinity','s_an',('1','psu'))]:
                original=ds[role]
                field=original[variable]
                if (field.dimensions!=('time','depth','lat','lon') or field.shape!=(1,*[
                        len(coordinates[k]) for k in ('depth','lat','lon')]) or field.units not in units
                        or getattr(field,'standard_name',None) != ('sea_water_temperature' if role=='temperature' else 'sea_water_salinity')
                        or 'World Ocean Atlas 2013 version 2' not in getattr(original,'title','')
                        or 'Annual' not in original.title or not getattr(original,'license','').strip()):
                    raise ValueError('invalid original WOA role/version/units/license: '+role)
                if not manufactured and not chosen[role].get('source_url','').startswith(
                        'https://www.ncei.noaa.gov/data/oceans/woa/WOA13/DATAv2/'):
                    raise ValueError('original WOA source URL/release missing')
            report['source_units']={role:ds[role]['t_an' if role=='temperature' else 's_an'].units for role in ds}
            report['practical_salinity_values_rescaled']=False  # PSS-78 is dimensionless, not kg/kg.
            p=pressure_ds[pressure_info['variable']]
            if p.dimensions!=('depth','lat') or p.units!='dbar':
                raise ValueError('pressure variable must be p[depth,lat] in dbar')
            pressure=np.ma.asarray(p[:])
            if (pressure.shape!=(len(coordinates['depth']),len(coordinates['lat']))
                    or np.ma.is_masked(pressure) or not np.isfinite(pressure).all()
                    or np.any((pressure<0)|(pressure>8000)) or np.any(pressure[0]!=0)
                    or np.any(np.diff(pressure,axis=0)<0)):
                raise ValueError('pressure must be finite, ordered, surface-zero sea pressure in dbar')
            target=output/'woa13v2_thermodynamics.nc'
            native=stack.enter_context(netCDF4.Dataset(target,'w'))
            native.Conventions='CF-1.6'
            native.data_kind=report['data_kind']
            native.title='WOA13v2 annual original-grid thermodynamic conversion'
            native.pressure_definition=pressure_info['definition']
            native.salinity_approximation='Reference Salinity as approximation to Absolute Salinity'
            native.license=temperature.license
            native.createDimension('time',1)
            source_time=temperature['time']
            time=native.createVariable('time',source_time.dtype,('time',))
            time.setncatts({key:source_time.getncattr(key) for key in source_time.ncattrs() if key!='_FillValue'})
            time[:]=source_time[:]
            time.cartesian_axis='T'
            if 'climatology' in source_time.ncattrs():
                bound_name=source_time.climatology
                bounds=temperature[bound_name]
                for dimension in bounds.dimensions:
                    if dimension not in native.dimensions:
                        native.createDimension(dimension,len(temperature.dimensions[dimension]))
                cloned=native.createVariable(bound_name,bounds.dtype,bounds.dimensions)
                cloned.setncatts({key:bounds.getncattr(key) for key in bounds.ncattrs() if key!='_FillValue'})
                cloned[:]=bounds[:]
            for name,values in coordinates.items():
                native.createDimension(name,len(values))
                axis=native.createVariable(name,'f8',(name,))
                axis.units=temperature[name].units
                axis[:]=values
                axis.cartesian_axis={'lon':'X','lat':'Y','depth':'Z'}[name]
            native['depth'].positive='down'
            outputs={}
            for name,standard in [('ptemp','sea_water_potential_temperature'),
                                  ('ct','sea_water_conservative_temperature'),('sr',None)]:
                field=native.createVariable(name,'f8',('time','depth','lat','lon'),fill_value=9.96921e36,
                    zlib=True,complevel=1,chunksizes=(1,1,len(coordinates['lat']),len(coordinates['lon'])))
                field.units='g kg-1' if name=='sr' else 'degrees_celsius'
                if standard:
                    field.standard_name=standard
                if name=='ptemp':
                    field.reference_pressure_dbar=0.
                outputs[name]=field
            valid_output=native.createVariable('paired_source_support','i1',('time','depth','lat','lon'))
            pt_compute=jax.jit(potential_from_in_situ)
            ct_compute=jax.jit(conservative_from_in_situ)
            for level,depth in enumerate(coordinates['depth']):
                t=np.ma.asarray(temperature['t_an'][0,level],dtype=float).filled(np.nan)
                s=np.ma.asarray(salinity['s_an'][0,level],dtype=float).filled(np.nan)
                if np.isinf(t).any() or np.isinf(s).any():
                    raise ValueError('infinite WOA field values are not missing support')
                valid=np.isfinite(t)&np.isfinite(s)
                full_pressure=np.broadcast_to(pressure[level,:,None],t.shape)
                sr=reference_salinity(jnp.asarray(np.where(valid,s,35.),dtype=jnp.float64))
                safe_t=jnp.asarray(np.where(valid,t,0.),dtype=jnp.float64)
                validate_state(sr,safe_t,full_pressure)
                pt=np.asarray(pt_compute(sr,safe_t,full_pressure))
                ct=np.asarray(ct_compute(sr,safe_t,full_pressure))
                validate_state(sr,ct,full_pressure)
                for name,value in [('ptemp',pt),('ct',ct),('sr',np.asarray(sr))]:
                    outputs[name][0,level]=np.ma.array(value,mask=~valid)
                valid_output[0,level]=valid.astype(np.int8)
                report['levels'].append({'depth_m':float(depth),'paired_original_cells':int(valid.sum()),
                    'temperature_only_cells':int((np.isfinite(t)&~np.isfinite(s)).sum()),
                    'salinity_only_cells':int((np.isfinite(s)&~np.isfinite(t)).sum()),
                    'missing_cells_preserved':int((~valid).sum())})
            # Safe values in masked cells never become initial water. This is
            # computation-domain sanitation, not an infill/fallback product.
        for role,path in paths.items():
            if sha256_file(path)!=chosen[role]['sha256']:
                raise ValueError('original WOA changed during conversion: '+role)
        if sha256_file(pressure_path)!=pressure_info['sha256']:
            raise ValueError('pressure input changed during conversion')
        if (sha256_file(acquisition) != acquisition_sha
                or sha256_file(pressure_reference) != pressure_receipt_sha):
            raise ValueError('initial source receipt changed during conversion')
        if not any(level['paired_original_cells'] for level in report['levels']):
            raise ValueError('no paired original WOA support to convert')
        actual_sources = package_source_hashes(__file__)
        if actual_sources != report['package_source_sha256']:
            raise ValueError('executed package changed during conversion')
        report.update(status='original_grid_thermodynamics_prepared',
                      output={'path':target.name,'sha256':sha256_file(target),'bytes':target.stat().st_size},
                      mom_input={'temperature_variable':'ptemp','temperature_file':target.name,
                                 'salinity_variable':'s_an','salinity_file':str(paths['salinity']),
                                 'salinity_units':'PSS-78 Practical Salinity'},
                      native_mapping_and_infill='not_performed')
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',reason_type=type(error).__name__,reason=str(error))
        raise
    finally:
        (output/'initialization.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
