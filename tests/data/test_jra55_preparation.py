"""Independent interpolation/routing and explicit manufactured end-to-end input."""
import json

import netCDF4
import numpy as np
import pytest

from tests.support.jra55_inputs import original_fixture
from zhenmode.execution.preparation import coastal_routing, prepare_jra_window, route_discharge
from zhenmode.model.inputs.forcing.jra55 import (
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
