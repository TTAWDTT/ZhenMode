"""FMS-specific time format and identity checks; no ocean qualification."""
import json

import netCDF4
import numpy as np
import pytest

from tests.support.jra55_inputs import original_fixture
from zhenmode.baselines.mom6.omip2_time import corrected_time_reader, prepare_time_inputs
from zhenmode.execution.preparation import prepare_jra_window


@pytest.mark.parametrize('atmos,coupling', [(0,3600), (True,3600), (3600.,3600),
                                          (4000,3600), (3600,5400)])
def test_invalid_coupling_clock_rejected_before_inputs(atmos,coupling):
    with pytest.raises(ValueError,match='coupling intervals'):
        prepare_time_inputs('missing','unused',dt_atmos=atmos,dt_cpld=coupling)


def test_unpinned_native_time_source_rejected():
    with pytest.raises(ValueError,match='audited pin'):
        corrected_time_reader(b'unverified source')


def prepared_fixture(tmp_path):
    received,grid,area,epoch=original_fixture(tmp_path)
    prepared=tmp_path/'spatial'
    prepare_jra_window(received,grid,area,prepared,start=epoch,end=epoch+21600,
                       maximum_routing_distance_m=20e6)
    return prepared


def test_native_time_adapter_preserves_numbers_bounds_and_manufactured_role(tmp_path):
    prepared=prepared_fixture(tmp_path)
    output=tmp_path/'fms'
    report=prepare_time_inputs(prepared,output)
    assert report['status']=='prepared' and report['data_kind']=='manufactured'
    assert not report['execution_ready'] and not report['climate_qualification']
    assert report['files']['shortwave_down']['mean_phase']=='right_endpoint'
    assert report['files']['runoff']['mean_phase']=='left_endpoint'
    for name in ('rsds','friver','tas'):
        with netCDF4.Dataset(prepared/(name+'.nc')) as source, netCDF4.Dataset(output/(name+'.nc')) as native:
            assert native.dimensions['time'].isunlimited()
            assert native['time'].calendar=='gregorian'
            assert native['lon'].cartesian_axis=='X'
            assert native['lat'].cartesian_axis=='Y'
            for variable in source.variables:
                if variable not in {'time','time_bounds'}:
                    np.testing.assert_array_equal(native[variable][:],source[variable][:])
                else:
                    dates=netCDF4.num2date(native[variable][:],native['time'].units,calendar='gregorian')
                    np.testing.assert_array_equal(netCDF4.date2num(dates,source['time'].units,
                        calendar=source['time'].calendar),source[variable][:])
            if name!='tas':
                np.testing.assert_array_equal(native['average_T1'][:],native['time_bounds'][:,0])
                np.testing.assert_array_equal(native['average_T2'][:],native['time_bounds'][:,1])
                assert native[name].zhenmode_interval_seconds==3600
            else:
                assert 'zhenmode_interval_sampling' not in native[name].ncattrs()
    with pytest.raises(FileExistsError):
        prepare_time_inputs(prepared,output)


def test_time_adapter_refuses_spatial_receipt_disagreement(tmp_path):
    prepared=prepared_fixture(tmp_path)
    path=prepared/'preparation.json'
    report=json.loads(path.read_text())
    report['fields']['wind_u']['prepared_sha256']='f'*64
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='spatial preparation receipt'):
        prepare_time_inputs(prepared,tmp_path/'failed')
    assert not (tmp_path/'failed').exists()


def test_actual_reader_harness_rejects_rewritten_native_bytes(tmp_path):
    from scripts.check_mom6_time_inputs import prepared_driver

    prepared = prepared_fixture(tmp_path)
    native = tmp_path/'native'
    prepare_time_inputs(prepared, native)
    # The harness is intentionally for the frozen first real-data window.
    path = prepared/'preparation.json'
    receipt = json.loads(path.read_text())
    receipt['window_seconds'] = [-378691200, -378669600]
    path.write_text(json.dumps(receipt))
    from zhenmode.provenance.sources import sha256_file
    path = native/'time-inputs.json'
    receipt = json.loads(path.read_text())
    receipt['input_preparation_sha256'] = sha256_file(prepared/'preparation.json')
    path.write_text(json.dumps(receipt))
    with netCDF4.Dataset(native/'tas.nc', 'a') as ds:
        ds['tas'][0,0,0] += 1
    with pytest.raises(ValueError, match='input identity mismatch'):
        prepared_driver(prepared, native, tmp_path/'check')
