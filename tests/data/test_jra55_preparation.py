"""Independent interpolation/routing and explicit manufactured end-to-end input."""
import json

import netCDF4
import numpy as np
import pytest

from zhenmode.execution.preparation import coastal_routing, prepare_jra_window, route_discharge
from zhenmode.model.inputs.forcing.jra55 import (
    FIELDS,
    TIME_UNITS,
    JRA55Forcing,
    bilinear_rectilinear_weights,
    remap_rectilinear_means,
)
from zhenmode.provenance.sources import sha256_file


def test_periodic_bilinear_state_and_no_polar_extrapolation():
    weights = bilinear_rectilinear_weights([0,90,180,270],[-60,60],[315,45],[0])
    field = np.array([[0,2,4,6],[10,12,14,16]])
    np.testing.assert_allclose(remap_rectilinear_means(field,weights),[[8,6]])
    with pytest.raises(ValueError,match='extrapolation'):
        bilinear_rectilinear_weights([0,180],[-60,60],[90],[89.5])
    with pytest.raises(ValueError,match='seam'):
        bilinear_rectilinear_weights([0,360],[-60,60],[90],[0])


def test_route_conserves_independent_mass_and_refuses_distant_discharge():
    wet = np.array([[0,1],[1,1]])
    routing = coastal_routing([90,270],[-45,45],[90,270],[-45,45],wet,maximum_distance_m=20e6)
    source_area=np.array([[2,3],[5,7]])
    source=np.array([[1,2],[3,4]])
    target_area=np.array([[11,13],[17,19]])
    actual=route_discharge(source,source_area,target_area,routing)
    assert actual[0,0] == 0
    assert np.sum(actual*target_area) == pytest.approx(51)  # 2 + 6 + 15 + 28 kg/s
    with pytest.raises(ValueError,match='routing limit'):
        route_discharge(source,source_area,target_area,(routing[0],routing[1],1))
    with pytest.raises(ValueError,match='recipient'):
        coastal_routing([90,270],[-45,45],[90,270],[-45,45],np.ones((2,2)),maximum_distance_m=1e6)


def test_unresolved_island_routes_locally_without_loosening_distance_limit():
    source=np.array([[1.,0.],[0.,0.]])
    routing=coastal_routing([90,270],[-45,45],[90,270],[-45,45],np.ones((2,2)),
        maximum_distance_m=1000,land_fraction=np.array([[.05,0],[0,0]]))
    actual=route_discharge(source,np.ones((2,2))*7,np.ones((2,2))*11,routing)
    np.testing.assert_allclose(actual,[[7/11,0],[0,0]])
    with pytest.raises(ValueError,match='land fraction'):
        coastal_routing([90,270],[-45,45],[90,270],[-45,45],np.ones((2,2)),
            maximum_distance_m=1000,land_fraction=np.full((2,2),1.01))


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


def test_prepared_cf_keeps_time_means_and_manufactured_identity(tmp_path):
    received,grid,area,epoch=original_fixture(tmp_path)
    output=tmp_path/'prepared'
    report=prepare_jra_window(received,grid,area,output,start=epoch,end=epoch+21600,maximum_routing_distance_m=20e6)
    assert report['status']=='prepared_and_reader_verified'
    assert not report['execution_ready'] and not report['climate_qualification']
    with np.load(output/'grid.npz') as native:
        reader=JRA55Forcing(output/'forcing.json',lon=native['lon'],lat=native['lat'],
                            wet_mask=native['wet_mask'],start_seconds=epoch,end_seconds=epoch+21600)
    assert reader.data_kind=='manufactured'
    sample=reader.sample(epoch+3600,interval_end_seconds=epoch+18000)
    np.testing.assert_allclose(sample.shortwave_down[reader.wet],200)
    assert sample.shortwave_down[~reader.wet].sum()==0
    assert report['fields']['runoff']['mass_integrals'][0]['original_kg_s'] == pytest.approx(
        4e-5*np.pi*6371000**2)
    with pytest.raises(FileExistsError):
        prepare_jra_window(received,grid,area,output,start=epoch,end=epoch+21600,maximum_routing_distance_m=20e6)


def test_wrong_original_identity_cannot_enter_native_manifest(tmp_path):
    received,grid,area,epoch=original_fixture(tmp_path)
    original=tmp_path/'uas-original.nc'
    with original.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='identity changed'):
        prepare_jra_window(received,grid,area,tmp_path/'failed',start=epoch,end=epoch+21600,maximum_routing_distance_m=20e6)
    assert json.loads((tmp_path/'failed/preparation.json').read_text())['status']=='failed'


def test_float32_discharge_accumulates_mass_in_float64():
    values=np.array([[1e-8,3e-7],[7e-6,9e-5]],dtype=np.float32)
    area=np.array([[8e7,3e8],[7e8,9e8]],dtype=np.float32)
    target=np.array([[13.,17.]])
    routing=(np.array([0,1,0,1]),np.zeros(4),1000.)
    actual=route_discharge(values,area,target,routing)
    import math
    independent=math.fsum(float(value)*float(cell) for value,cell in zip(values.ravel(),area.ravel()))
    assert np.sum(actual*target)==pytest.approx(independent,rel=1e-15)


@pytest.mark.parametrize('attribute,value',[('coordinate_units','radians'),('license','')])
def test_hash_verified_original_still_requires_geographic_units_and_license(tmp_path,attribute,value):
    received,grid,area,epoch=original_fixture(tmp_path)
    original=tmp_path/'uas-original.nc'
    with netCDF4.Dataset(original,'a') as ds:
        if attribute=='coordinate_units':
            ds['lon'].units=value
        else:
            ds.license=value
    identity=json.loads(received.read_text())
    identity['files']['uas']['bytes']=original.stat().st_size
    identity['files']['uas']['sha256']=sha256_file(original)
    identity['verified']['uas']=sha256_file(original)
    received.write_text(json.dumps(identity))
    with pytest.raises(ValueError,match='metadata mismatch'):
        prepare_jra_window(received,grid,area,tmp_path/'failed',start=epoch,end=epoch+21600,
                           maximum_routing_distance_m=20e6)


@pytest.mark.parametrize('change', ['scaled', 'permuted'])
def test_positive_native_area_must_match_independent_bounds(tmp_path, change):
    received, grid, area, epoch = original_fixture(tmp_path)
    with np.load(grid) as ds:
        values = {key:ds[key] for key in ds.files}
    if change == 'scaled':
        values['area'] *= 2
    else:
        values['area'][0,0] *= .9
        values['area'][1,0] *= 1.1  # Same global sum, wrong local freshwater flux.
    np.savez(grid, **values)
    with pytest.raises(ValueError, match='area disagrees with spherical cell bounds'):
        prepare_jra_window(received, grid, area, tmp_path/'rejected', start=epoch,
                           end=epoch+21600, maximum_routing_distance_m=20e6)
    assert not (tmp_path/'rejected').exists()


def test_interrupted_preprocessing_keeps_failed_receipt(tmp_path, monkeypatch):
    from zhenmode.execution import preparation

    received, grid, area, epoch = original_fixture(tmp_path)
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(preparation, 'remap_rectilinear_means', interrupt)
    with pytest.raises(KeyboardInterrupt):
        prepare_jra_window(received, grid, area, tmp_path/'interrupted', start=epoch,
                           end=epoch+21600, maximum_routing_distance_m=20e6)
    receipt = json.loads((tmp_path/'interrupted/preparation.json').read_text())
    assert receipt['status'] == 'failed' and receipt['reason_type'] == 'KeyboardInterrupt'
    assert receipt['reason'] == 'KeyboardInterrupt'


def annual_fixture(tmp_path, *, offset=0):
    first, second = tmp_path/'1958', tmp_path/'1959'
    first.mkdir()
    second.mkdir()
    epoch = float(netCDF4.date2num(netCDF4.num2date(0,'seconds since 1958-12-31',
                   calendar='gregorian'),TIME_UNITS,calendar='gregorian'))
    previous,grid,area,_ = original_fixture(first,epoch=epoch,instant_count=8,mean_count=8)
    following,_,_,_ = original_fixture(second,epoch=epoch+86400+offset)
    for receipt, wind, radiation in [(previous,3,100),(following,9,400)]:
        identity = json.loads(receipt.read_text())
        for variable, value in [('uas',wind),('rsds',radiation)]:
            source = receipt.parent/identity['files'][variable]['filename']
            with netCDF4.Dataset(source,'a') as ds:
                ds[variable][:] = value
            identity['files'][variable]['bytes'] = source.stat().st_size
            identity['files'][variable]['sha256'] = sha256_file(source)
            identity['verified'][variable] = sha256_file(source)
        receipt.write_text(json.dumps(identity))
    return previous,following,grid,area,epoch


def test_adjacent_annual_shards_interpolate_and_integrate_across_new_year(tmp_path):
    previous,following,grid,area,epoch = annual_fixture(tmp_path)
    output = tmp_path/'cross-year'
    report = prepare_jra_window([following,previous],grid,area,output,
        start=epoch+21*3600,end=epoch+27*3600,maximum_routing_distance_m=20e6)
    assert report['status'] == 'prepared_and_reader_verified'
    assert len(report['fields']['wind_u']['original_sources']) == 2
    with np.load(output/'grid.npz') as native:
        reader = JRA55Forcing(output/'forcing.json',lon=native['lon'],lat=native['lat'],
            wet_mask=native['wet_mask'],start_seconds=epoch+21*3600,end_seconds=epoch+27*3600)
    sample = reader.sample(epoch+23*3600,interval_end_seconds=epoch+25*3600)
    np.testing.assert_allclose(sample.wind_u[reader.wet],7)
    np.testing.assert_allclose(sample.shortwave_down[reader.wet],250)
    assert reader.data_kind == 'manufactured'
    assert not report['execution_ready'] and not report['climate_qualification']


@pytest.mark.parametrize('offset',[-10800,10800])
def test_overlapping_or_gapped_annual_shards_are_rejected(tmp_path,offset):
    previous,following,grid,area,epoch = annual_fixture(tmp_path,offset=offset)
    with pytest.raises(ValueError,match='overlap or leave a time gap'):
        prepare_jra_window([previous,following],grid,area,tmp_path/'rejected',
            start=epoch+21*3600,end=epoch+27*3600,maximum_routing_distance_m=20e6)
    assert json.loads((tmp_path/'rejected/preparation.json').read_text())['status']=='failed'


def test_missing_next_year_instant_is_not_extrapolated(tmp_path):
    previous,_,grid,area,epoch = annual_fixture(tmp_path)
    with pytest.raises(ValueError,match='instant cadence/gap/duplicate'):
        prepare_jra_window(previous,grid,area,tmp_path/'uncovered',
            start=epoch+21*3600,end=epoch+24*3600,maximum_routing_distance_m=20e6)
    assert json.loads((tmp_path/'uncovered/preparation.json').read_text())['status']=='failed'


def test_repeated_acquisition_cli_prepares_cross_year_window(tmp_path, capsys):
    from zhenmode.execution.benchmark import main

    previous,following,grid,area,_ = annual_fixture(tmp_path)
    output = tmp_path/'cli-cross-year'
    assert main(['prepare-forcing','--acquisition',str(previous),'--acquisition',str(following),
        '--grid',str(grid),'--runoff-area',str(area),'--output',str(output),
        '--start','1958-12-31T21:00:00','--end','1959-01-01T03:00:00',
        '--maximum-routing-distance-m','20000000']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'prepared_and_reader_verified'
    receipt = json.loads((output/'preparation.json').read_text())
    assert len(receipt['acquisition_sha256']) == 2
    assert len(receipt['fields']['wind_u']['selected_records']) == 3
