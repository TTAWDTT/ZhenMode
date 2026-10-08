"""FMS-specific time format and identity checks; no ocean qualification."""
import json
import math
import os
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from tests.support.jra55_inputs import original_fixture
from zhenmode.baselines.mom6 import omip2_time
from zhenmode.baselines.mom6.omip2_time import corrected_time_reader, prepare_time_inputs
from zhenmode.preparation.forcing import prepare_jra_window


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


@pytest.mark.parametrize('offset,accepted', [(1800, False), (3600, True)])
def test_window_phase_checked_against_actual_mean_boundaries(tmp_path, offset, accepted):
    received, grid, area, epoch = original_fixture(tmp_path)
    prepared = tmp_path/'spatial'
    # Both windows pass the CF reader and contain complete hourly requests.
    prepare_jra_window(received, grid, area, prepared, start=epoch+offset,
                       end=epoch+offset+14400, maximum_routing_distance_m=20e6)
    output = tmp_path/'native'
    if accepted:
        report = prepare_time_inputs(prepared, output)
        assert report['mean_clock_validation']['shortwave_down']['interior_boundaries'] == 1
        assert report['mean_clock_validation']['runoff']['aligned']
    else:
        with pytest.raises(ValueError, match='source boundary does not align'):
            prepare_time_inputs(prepared, output)
        assert not output.exists()


def test_discharge_daily_boundary_uses_coupling_clock():
    reader = SimpleNamespace(start=84600., end=88200., records={
        'runoff': [SimpleNamespace(bounds=(0., 86400.)), SimpleNamespace(bounds=(86400., 172800.))]})
    fields = {'runoff': ('friver', (), 86400, 'mean', None)}
    with pytest.raises(ValueError, match='runoff source boundary does not align'):
        omip2_time._mean_clock_alignment(reader, fields, 1800, 3600)
    assert omip2_time._mean_clock_alignment(reader, fields, 1800, 1800)['runoff']['aligned']


@pytest.mark.parametrize('shape', [(2921, 260, 720), (1, 2048, 2048)])
def test_large_field_blocks_cover_whole_array_below_budget(shape):
    blocks = list(omip2_time._variable_blocks(shape, 8))
    assert len(blocks) > 1
    volumes = [math.prod(part.stop-part.start for part in block) for block in blocks]
    assert max(volumes)*8 <= omip2_time._COPY_BYTES
    assert sum(volumes) == math.prod(shape)
    assert len({tuple((part.start, part.stop) for part in block) for block in blocks}) == len(blocks)
    for axis, length in enumerate(shape):
        spans = sorted({(block[axis].start, block[axis].stop) for block in blocks})
        assert spans[0][0] == 0 and spans[-1][1] == length
        assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))


def test_real_netcdf_copy_streams_instead_of_whole_arrays(tmp_path, monkeypatch):
    prepared = prepared_fixture(tmp_path)
    monkeypatch.setattr(omip2_time, '_COPY_BYTES', 32)
    copy = omip2_time._copy_variable
    chunks = []
    class GuardedVariable:
        def __init__(self, variable):
            self.variable = variable
        def __getattr__(self, name):
            return getattr(self.variable, name)
        def __getitem__(self, block):
            value = self.variable[block]
            assert value.nbytes <= 32
            chunks.append(value.nbytes)
            return value
    def guarded(source, *args):
        return copy(GuardedVariable(source), *args)
    monkeypatch.setattr(omip2_time, '_copy_variable', guarded)
    output = tmp_path/'native'
    report = prepare_time_inputs(prepared, output)
    assert report['status'] == 'prepared' and max(chunks) == 32
    for field in ('tas', 'rsds', 'friver'):
        with netCDF4.Dataset(prepared/(field+'.nc')) as source, netCDF4.Dataset(output/(field+'.nc')) as native:
            np.testing.assert_array_equal(source[field][:], native[field][:])
    assert all(row['source_revalidated_after_copy'] for row in report['files'].values())


@pytest.mark.parametrize('change', ['bytes', 'stat'])
def test_source_rechecked_after_conversion(tmp_path, monkeypatch, change):
    prepared = prepared_fixture(tmp_path)
    output = tmp_path/'native'
    original_hash = omip2_time.sha256_file
    changed = False
    def mutate_after_copy(path):
        nonlocal changed
        if path == prepared/'tas.nc' and (output/'tas.nc').exists() and not changed:
            changed = True
            if change == 'bytes':
                with path.open('ab') as stream:
                    stream.write(b'changed after copying')
            else:
                stat = path.stat()
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1000000000))
        return original_hash(path)
    monkeypatch.setattr(omip2_time, 'sha256_file', mutate_after_copy)
    with pytest.raises(ValueError, match='source bytes changed during time conversion'):
        prepare_time_inputs(prepared, output)
    receipt = json.loads((output/'time-inputs.json').read_text())
    assert changed and receipt['status'] == 'failed'
    assert not receipt['execution_ready'] and not receipt['climate_qualification']


def test_time_copy_keyboard_interrupt_records_failure(tmp_path, monkeypatch):
    prepared = prepared_fixture(tmp_path)
    def interrupted(*args):
        raise KeyboardInterrupt('manufactured Ctrl-C during copy')
    monkeypatch.setattr(omip2_time, '_copy_variable', interrupted)
    output = tmp_path/'native'
    with pytest.raises(KeyboardInterrupt):
        prepare_time_inputs(prepared, output)
    receipt = json.loads((output/'time-inputs.json').read_text())
    assert receipt['status'] == 'failed' and receipt['reason_type'] == 'KeyboardInterrupt'
