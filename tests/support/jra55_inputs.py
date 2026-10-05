"""Shared manufactured JRA inputs; no scoring or numerical reference implementation."""
import json

import netCDF4
import numpy as np

from zhenmode.model.inputs.forcing.jra55 import FIELDS, TIME_UNITS
from zhenmode.provenance.sources import sha256_file


def original_fixture(tmp_path, *, epoch=None, instant_count=3, mean_count=2):
    lon,lat=np.array([90.,270.]),np.array([-45.,45.])
    area=np.full((2,2),np.pi*6371000**2)
    bounds={'lon':np.array([[0.,180.],[180.,360.]]),'lat':np.array([[-90.,0.],[0.,90.]])}
    grid=tmp_path/'native-grid.npz'
    np.savez(grid,lon=lon,lat=lat,lon_bounds=bounds['lon'],lat_bounds=bounds['lat'],area=area,wet_mask=np.array([[1,1],[0,1]]))
    if epoch is None:
        epoch=netCDF4.date2num(netCDF4.num2date(0,'seconds since 1958-01-01',calendar='proleptic_gregorian'),TIME_UNITS,calendar='proleptic_gregorian')
    acquisition={'product':'JRA55-do','version':'1.4.0','execution_status':'completed',
                 'data_kind':'manufactured','files':{},'verified':{}}
    def axes(ds):
        ds.source_id='MRI-JRA55-do-1-4-0'
        ds.data_kind='manufactured'
        ds.license='manufactured fixture; no observational provenance'
        ds.createDimension('bounds',2)
        for axis,values,unit in [('lon',lon,'degrees_east'),('lat',lat,'degrees_north')]:
            ds.createDimension(axis,2)
            v=ds.createVariable(axis,'f8',(axis,))
            v.units=unit
            v[:]=values
            v.bounds=axis+'_bounds'
            ds.createVariable(v.bounds,'f8',(axis,'bounds'))[:]=bounds[axis]
    for _,(variable,units,cadence,interpretation,height) in FIELDS.items():
        path=tmp_path/(variable+'-original.nc')
        count=instant_count if interpretation=='instant' else 1 if cadence==86400 else mean_count
        with netCDF4.Dataset(path,'w') as ds:
            axes(ds)
            ds.createDimension('time',count)
            time=ds.createVariable('time','f8',('time',))
            time.units=TIME_UNITS
            time.calendar='proleptic_gregorian'
            time[:]=epoch+np.arange(count)*cadence+(cadence/2 if interpretation=='mean' else 0)
            if interpretation=='mean':
                time.bounds='time_bounds'
                ds.createVariable('time_bounds','f8',('time','bounds'))[:]=epoch+np.column_stack((np.arange(count)*cadence,(np.arange(count)+1)*cadence))
            if height:
                h=ds.createVariable('height','f8')
                h.units='m'
                h.assignValue(height)
            v=ds.createVariable(variable,'f8',('time','lat','lon'))
            v.units=units[0]
            v.cell_methods='area: mean time: point' if interpretation=='instant' else {
                'friver':'area: mean where sea time: mean','licalvf':'area: time: mean where ice_sheet'}.get(variable,'area: time: mean')
            value={'tas':280,'huss':.004,'psl':101325,'rlds':320,'prra':1e-5,'prsn':0,
                   'friver':1e-5,'licalvf':2e-6}.get(variable,3)
            v[:]=value
            if variable=='rsds':
                v[0]=0
                v[1]=400
        acquisition['files'][variable]={'filename':path.name,'bytes':path.stat().st_size,
                'sha256':sha256_file(path),'source_url':'https://example.invalid/manufactured/'+variable}
        acquisition['verified'][variable]=sha256_file(path)
    original_area=tmp_path/'original-area.nc'
    with netCDF4.Dataset(original_area,'w') as ds:
        axes(ds)
        v=ds.createVariable('areacello','f8',('lat','lon'))
        v.units='m2'
        v[:]=area
    area_manifest=tmp_path/'area.json'
    area_manifest.write_text(json.dumps({'path':original_area.name,'variable':'areacello',
        'bytes':original_area.stat().st_size,'sha256':sha256_file(original_area)}))
    received=tmp_path/'acquisition.json'
    received.write_text(json.dumps(acquisition))
    return received,grid,area_manifest,epoch
