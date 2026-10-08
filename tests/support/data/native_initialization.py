"""Shared manufactured WOA and native fixtures; no observational qualification."""
import json

import netCDF4
import numpy as np

from zhenmode.preparation.woa import prepare_woa_thermodynamics
from zhenmode.provenance.sources import sha256_file


def woa_source_fixture(tmp_path,*,change=None):
    coordinates={'lon':[-90.,90.],'lat':[-45.,45.],'depth':[0.,500.,4000.]}
    units={'lon':'degrees_east','lat':'degrees_north','depth':'meters'}
    receipt={'kind':'WOA13v2_originals','status':'completed','files':[]}
    for role,variable,unit in [('temperature','t_an','degrees_celsius'),('salinity','s_an','psu')]:
        path=tmp_path/(role+'.nc')
        with netCDF4.Dataset(path,'w') as ds:
            ds.title='World Ocean Atlas 2013 version 2 : Annual manufactured fixture'
            ds.license='' if change=='license' else 'manufactured test fixture'
            ds.data_kind='manufactured'
            ds.createDimension('time',1)
            time=ds.createVariable('time','f8',('time',))
            time.units='months since 0000-01-01 00:00:00'
            time[:]=6.
            time.climatology='climatology_bounds'
            ds.createDimension('nbounds',2)
            ds.createVariable('climatology_bounds','f8',('time','nbounds'))[:]=[[0.,12.]]
            for name,values in coordinates.items():
                ds.createDimension(name,len(values))
                axis=ds.createVariable(name,'f8',(name,))
                axis.units=units[name]
                axis[:]=np.asarray(values)+(1 if change=='coordinates' and role=='salinity' and name=='lat' else 0)
            field=ds.createVariable(variable,'f8',('time','depth','lat','lon'),fill_value=9.96921e36)
            field.units='K' if change=='units' and role=='temperature' else unit
            field.standard_name='sea_water_temperature' if role=='temperature' else 'sea_water_salinity'
            values=np.full((3,2,2),35.) if role=='salinity' else np.broadcast_to(np.array([10.,10.,0.])[:,None,None],(3,2,2)).copy()
            mask=np.zeros(values.shape,dtype=bool)
            mask[1,0,0]=True
            field[0]=np.ma.array(values,mask=mask)
        receipt['files'].append({'role':role,'climatology_month':0,'path':path.name,
            'bytes':path.stat().st_size,'sha256':sha256_file(path),
            'source_url':'https://example.invalid/manufactured','checksum_scope':'manufactured'})
    acquisition=tmp_path/'acquisition.json'
    acquisition.write_text(json.dumps(receipt))
    path=tmp_path/'pressure.nc'
    with netCDF4.Dataset(path,'w') as ds:
        for name in ('depth','lat'):
            ds.createDimension(name,len(coordinates[name]))
            axis=ds.createVariable(name,'f8',(name,))
            axis.units=units[name]
            axis[:]=coordinates[name]
        field=ds.createVariable('p','f8',('depth','lat'))
        field.units='Pa' if change=='pressure_units' else 'dbar'
        field[:]=np.array([0.,500.,4000.])[:,None]+np.zeros((3,2))
        if change=='pressure_surface':
            field[0]=1.
    pressure=tmp_path/'pressure.json'
    pressure.write_text(json.dumps({'path':path.name,'variable':'p','bytes':path.stat().st_size,
        'sha256':sha256_file(path),'units':'dbar','definition':'manufactured explicit reference pressure'}))
    return acquisition,pressure


def native_inputs(tmp_path, *, unanchored=False):
    acquisition, pressure = woa_source_fixture(tmp_path)
    source = tmp_path / "source"
    prepare_woa_thermodynamics(acquisition, pressure, source)
    geometry = tmp_path / "geometry"
    geometry.mkdir()
    lon = np.array([90.0, 270.0])
    lat = np.array([-45.0, 45.0])
    bed = np.array([[0.5 if unanchored else 750.0, 6500.0], [750.0, 0.0]])
    area = np.full((2, 2), np.pi * 6371000.0**2)
    np.savez(
        geometry / "grid.npz",
        lon=lon,
        lat=lat,
        wet_mask=(bed > 0).astype(np.uint8),
        area=area,
        lon_bounds=[[0.0, 180.0], [180.0, 360.0]],
        lat_bounds=[[-90.0, 0.0], [0.0, 90.0]],
    )
    np.savez(geometry / "bathymetry.npz", lon=lon, lat=lat, depth=bed)
    (geometry / "geometry.json").write_text(
        json.dumps(
            {
                "grid_sha256": sha256_file(geometry / "grid.npz"),
                "bathymetry_sha256": sha256_file(geometry / "bathymetry.npz"),
            }
        )
    )
    nodes = tmp_path / "nodes.json"
    nodes.write_text(json.dumps({"z_nodes_m": [0.0, -500.0, -4000.0, -6000.0, -7000.0]}))
    return source, geometry, nodes, bed, area
