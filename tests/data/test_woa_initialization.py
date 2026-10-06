"""Original-grid source conversion, masks, independent GSW values and refusals."""
import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from zhenmode.execution.benchmark import main
from zhenmode.execution.initialization import prepare_woa_thermodynamics
from zhenmode.provenance.sources import sha256_file


def fixture(tmp_path,*,change=None):
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


def test_dimensionless_pss78_original_is_not_scaled_to_mass_fraction(tmp_path):
    acquisition,pressure=fixture(tmp_path)
    path=tmp_path/'salinity.nc'
    with netCDF4.Dataset(path,'a') as ds:
        ds['s_an'].units='1'
    receipt=json.loads(acquisition.read_text())
    receipt['files'][1]['sha256']=sha256_file(path)
    receipt['files'][1]['bytes']=path.stat().st_size
    acquisition.write_text(json.dumps(receipt))
    out=tmp_path/'normalized'
    report=prepare_woa_thermodynamics(acquisition,pressure,out)
    assert report['source_units']['salinity']=='1' and not report['practical_salinity_values_rescaled']
    with netCDF4.Dataset(out/report['output']['path']) as ds:
        np.testing.assert_allclose(ds['sr'][0,0],35.16504,rtol=0,atol=1e-13)


def test_original_grid_conversion_matches_gsw_and_preserves_missing_support(tmp_path):
    acquisition,pressure=fixture(tmp_path)
    out=tmp_path/'converted'
    report=prepare_woa_thermodynamics(acquisition,pressure,out)
    assert report['status']=='original_grid_thermodynamics_prepared'
    assert report['data_kind']=='manufactured' and not report['native_initialization_ready']
    assert not report['execution_ready'] and not report['climate_qualification']
    reference=np.asarray(json.loads((Path(__file__).resolve().parents[1]/'support/temperature_reference.json').read_text())['rows'])
    expected=reference[[9,7,6]]  # SA=35.16504, T=10/10/0, p=0/500/4000
    with netCDF4.Dataset(out/report['output']['path']) as ds:
        np.testing.assert_allclose(ds['ct'][0,:,1,1],expected[:,5],rtol=0,atol=1e-10)
        np.testing.assert_allclose(ds['ptemp'][0,:,1,1],expected[:,6],rtol=0,atol=1e-10)
        np.testing.assert_allclose(ds['sr'][0,:,1,1],35.16504,rtol=0,atol=1e-13)
        for name in ('ptemp','ct','sr'):
            assert ds[name][0,1].mask[0,0]
            assert np.ma.getmaskarray(ds[name][:]).sum()==1
        assert ds['paired_source_support'][0,1,0,0]==0
    assert report['levels'][1]['missing_cells_preserved']==1
    assert report['mom_input']['salinity_variable']=='s_an'
    assert sha256_file(out/report['output']['path'])==report['output']['sha256']
    assert 'zhenmode/model/config' in report['package_source_sha256']
    with pytest.raises(FileExistsError):
        prepare_woa_thermodynamics(acquisition,pressure,out)


@pytest.mark.parametrize('change,reason',[('units','role/version/units'),('coordinates','coordinates differ'),
    ('license','role/version/units/license'),('pressure_units','in dbar'),('pressure_surface','surface-zero')])
def test_initial_source_refuses_metadata_or_pressure_disagreement(tmp_path,change,reason):
    acquisition,pressure=fixture(tmp_path,change=change)
    out=tmp_path/'failed'
    with pytest.raises(ValueError,match=reason):
        prepare_woa_thermodynamics(acquisition,pressure,out)
    report=json.loads((out/'initialization.json').read_text())
    assert report['status']=='failed' and not report['execution_ready']


def test_initial_source_refuses_rewritten_original_bytes(tmp_path):
    acquisition,pressure=fixture(tmp_path)
    with netCDF4.Dataset(tmp_path/'temperature.nc','a') as ds:
        ds['t_an'][0,0,0,0]+=1
    with pytest.raises(ValueError,match='identity mismatch'):
        prepare_woa_thermodynamics(acquisition,pressure,tmp_path/'failed')
    assert not (tmp_path/'failed').exists()


def test_initial_source_cli_is_installed_entry_and_does_not_qualify_case(tmp_path,capsys):
    acquisition,pressure=fixture(tmp_path)
    assert main(['prepare-initial-source','--acquisition',str(acquisition),'--pressure-reference',str(pressure),
                 '--output',str(tmp_path/'cli')])==0
    assert json.loads(capsys.readouterr().out)['execution_ready'] is False


def rebind_source(acquisition, role):
    receipt=json.loads(acquisition.read_text())
    row=next(row for row in receipt['files'] if row['role']==role)
    path=acquisition.parent/row['path']
    row.update(sha256=sha256_file(path),bytes=path.stat().st_size)
    acquisition.write_text(json.dumps(receipt))


@pytest.mark.parametrize('change',['time','bounds','units','missing_bounds'])
def test_paired_annual_time_and_bounds_are_checked(tmp_path,change):
    acquisition,pressure=fixture(tmp_path)
    with netCDF4.Dataset(tmp_path/'salinity.nc','a') as ds:
        if change=='time':
            ds['time'][:]=7.
        elif change=='bounds':
            ds['climatology_bounds'][:]=[[0.,1.]]
        elif change=='units':
            ds['time'].units='days since 1958-01-01 00:00:00'
        else:
            ds['time'].delncattr('climatology')
    rebind_source(acquisition,'salinity')
    with pytest.raises(ValueError,match='annual times differ|annual climatology'):
        prepare_woa_thermodynamics(acquisition,pressure,tmp_path/'failed')
    assert json.loads((tmp_path/'failed/initialization.json').read_text())['status']=='failed'


def test_pair_support_is_intersection_without_infill(tmp_path):
    acquisition,pressure=fixture(tmp_path)
    with netCDF4.Dataset(tmp_path/'salinity.nc','a') as ds:
        ds['s_an'][0,2,1,1]=np.ma.masked
    rebind_source(acquisition,'salinity')
    report=prepare_woa_thermodynamics(acquisition,pressure,tmp_path/'paired')
    with netCDF4.Dataset(tmp_path/'paired'/report['output']['path']) as ds:
        expected=np.zeros((3,2,2),dtype=bool)
        expected[1,0,0]=expected[2,1,1]=True
        for name in ('ct','ptemp','sr'):
            np.testing.assert_array_equal(np.ma.getmaskarray(ds[name][0]),expected)
        np.testing.assert_array_equal(ds['paired_source_support'][0],~expected)
    assert report['levels'][2]['temperature_only_cells']==1
    assert report['source_time_definition']['bounds']==[[0.,12.]]


def test_all_missing_source_does_not_announce_success(tmp_path):
    acquisition,pressure=fixture(tmp_path)
    with netCDF4.Dataset(tmp_path/'temperature.nc','a') as ds:
        ds['t_an'][:]=np.ma.masked_all(ds['t_an'].shape)
    rebind_source(acquisition,'temperature')
    with pytest.raises(ValueError,match='no paired original WOA support'):
        prepare_woa_thermodynamics(acquisition,pressure,tmp_path/'failed')
    assert json.loads((tmp_path/'failed/initialization.json').read_text())['status']=='failed'


@pytest.mark.parametrize('document',['acquisition','pressure'])
def test_receipt_mutation_during_conversion_is_not_blessed(tmp_path,monkeypatch,document):
    import zhenmode.execution.initialization as initialization
    acquisition,pressure=fixture(tmp_path)
    original=initialization._coordinate
    changed=False
    def coordinate(*args):
        nonlocal changed
        result=original(*args)
        if not changed:
            path=acquisition if document=='acquisition' else pressure
            with path.open('a') as stream:
                stream.write('\n')
            changed=True
        return result
    monkeypatch.setattr(initialization,'_coordinate',coordinate)
    with pytest.raises(ValueError,match='receipt changed during conversion'):
        prepare_woa_thermodynamics(acquisition,pressure,tmp_path/'failed')
    assert json.loads((tmp_path/'failed/initialization.json').read_text())['status']=='failed'
