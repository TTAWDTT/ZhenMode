"""Explicit bounded means for the pinned FMS reader; ordinary inputs stay linear.

Atmospheric rain/radiation are read at the advanced atmospheric clock. River
and calving enter land-to-ice at the coupling interval's initial clock instead.
This adapter does not qualify the rest of the coupled experiment.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

from zhenmode.provenance.sources import load_json, sha256_file

_PIN = load_json(Path(__file__).with_name('omip2-pins.json'))['bounded_mean_source']
FMS_SOURCE, FMS_SHA256 = _PIN['path'], _PIN['sha256']
MOM_TIME_UNITS = 'seconds since 1958-01-01 00:00:00'
_COPY_BYTES = 8 * 1024**2


def _mean_clock_alignment(reader, fields, dt_atmos, dt_cpld):
    """Every interior source boundary must lie on its actual caller's clock.

    A window may start within a source record if no request crosses bounds.
    Duration alone does not prove this: 00:30 + hourly steps crosses 03:00.
    """
    report = {}
    for field, (_, _, _, interpretation, _) in fields.items():
        if interpretation != 'mean':
            continue
        interval = dt_cpld if field in {'runoff', 'calving'} else dt_atmos
        boundaries = {bound for record in reader.records[field] for bound in record.bounds
                      if reader.start < bound < reader.end}
        for bound in boundaries:
            phase = (bound-reader.start)/interval
            if not math.isclose(phase, round(phase), rel_tol=0, abs_tol=1e-9):
                raise ValueError(f'{field} source boundary does not align with the coupling clock')
        report[field] = {'interior_boundaries': len(boundaries), 'interval_seconds': interval,
                         'aligned': True}
    return report


def _variable_blocks(shape, itemsize):
    """Multiaxis slices bounding each numeric array, including a large single record."""
    if itemsize > _COPY_BYTES or any(length <= 0 for length in shape):
        raise ValueError('invalid numeric variable shape or block budget')
    block = list(shape)
    while math.prod(block)*itemsize > _COPY_BYTES:
        axis = max(range(len(block)), key=lambda axis: block[axis])
        block[axis] = (block[axis]+1)//2
    yield from (tuple(slice(start, min(start+width, length))
                      for start, width, length in zip(starts, block, shape, strict=True))
                for starts in itertools.product(*(range(0, length, width)
                                                  for length, width in zip(shape, block, strict=True))))


def _copy_variable(source, clone, source_time, is_time):
    """Stream actual NetCDF arrays; validate all values and normalized timestamps."""
    import netCDF4
    import numpy as np

    for block in _variable_blocks(source.shape, source.dtype.itemsize):
        values = source[block]
        if np.ma.is_masked(values) or not np.isfinite(values).all():
            raise ValueError('missing or nonfinite prepared data: '+source.name)
        if is_time:
            dates = netCDF4.num2date(values, source_time.units, calendar=source_time.calendar)
            if any(date.year < 1958 for date in np.asarray(dates).flat):
                raise ValueError('FMS native time precedes frozen 1958 reference')
            normalized = netCDF4.date2num(dates, MOM_TIME_UNITS, calendar='gregorian')
            actual_dates = netCDF4.num2date(normalized, MOM_TIME_UNITS, calendar='gregorian')
            if not np.array_equal(netCDF4.date2num(actual_dates, source_time.units,
                                                  calendar=source_time.calendar), values):
                raise ValueError('calendar/reference normalization changed actual timestamps')
            values = normalized
        clone[block] = values


def _file_identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def prepare_time_inputs(prepared, output, *, dt_atmos=3600, dt_cpld=3600):
    """Adapt normalized CF files for actual FMS record/clock conventions.

    Requires the separately identity-checked spatial preparation. Only modern
    Gregorian timestamps are supported. Flux values and actual dates are retained;
    records become unlimited and means gain explicit FMS interval metadata.
    This is a time/format adapter, not a grid or complete-case preparation.
    """
    import netCDF4
    import numpy as np

    from zhenmode.model.inputs.forcing.jra55 import FIELDS, JRA55Forcing

    if (type(dt_atmos) is not int or type(dt_cpld) is not int
            or dt_atmos <= 0 or dt_cpld <= 0 or 10800 % dt_atmos
            or 86400 % dt_cpld or dt_cpld % dt_atmos):
        raise ValueError('coupling intervals must divide the source periods and each other')
    prepared, output = Path(prepared), Path(output)
    receipt = load_json(prepared/'preparation.json')
    start, end = receipt['window_seconds']
    if receipt['status'] != 'prepared_and_reader_verified' or (end-start) % dt_cpld:
        raise ValueError('verified preparation and complete coupling intervals required')
    with np.load(prepared/'grid.npz', allow_pickle=False) as grid:
        reader = JRA55Forcing(prepared/'forcing.json', lon=grid['lon'], lat=grid['lat'],
                             wet_mask=grid['wet_mask'], start_seconds=start, end_seconds=end)
    reader.sample(start, interval_end_seconds=start+dt_atmos)
    manifest = load_json(prepared/'forcing.json')
    if set(receipt['fields']) != set(FIELDS):
        raise ValueError('incomplete original spatial preparation')
    for row in manifest['files']:
        if row['sha256'] != receipt['fields'][row['field']]['prepared_sha256']:
            raise ValueError('native bytes disagree with spatial preparation receipt')
    mean_clocks = _mean_clock_alignment(reader, FIELDS, dt_atmos, dt_cpld)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status':'running', 'scope':'FMS_time_format_only_not_complete_case',
              'data_kind':reader.data_kind, 'dt_atmos_seconds':dt_atmos, 'dt_cpld_seconds':dt_cpld,
              'input_preparation_sha256':sha256_file(prepared/'preparation.json'),
              'input_manifest_sha256':sha256_file(prepared/'forcing.json'),
              'executed_adapter_sha256':sha256_file(__file__), 'files':{},
              'mean_clock_validation':mean_clocks, 'numeric_copy_block_limit_bytes':_COPY_BYTES,
              'execution_ready':False, 'climate_qualification':False}
    try:
        for row in manifest['files']:
            field = row['field']
            variable, _, _, interpretation, _ = FIELDS[field]
            path = prepared / row['path']
            target = output / (variable+'.nc')
            phase = ('left_endpoint' if field in {'runoff','calving'} else 'right_endpoint')
            interval = dt_cpld if field in {'runoff','calving'} else dt_atmos
            before = _file_identity(path)
            original_sha = sha256_file(path)
            if before[2] != row['bytes'] or original_sha != row['sha256']:
                raise ValueError('source bytes changed before time conversion: '+field)
            with netCDF4.Dataset(path) as original, netCDF4.Dataset(target, 'w') as native:
                source_time = original['time']
                original_time_units = source_time.units
                bound_name = source_time.bounds if interpretation == 'mean' else None
                native.setncatts({name:original.getncattr(name) for name in original.ncattrs()})
                for name, dimension in original.dimensions.items():
                    native.createDimension(name, None if name == 'time' else len(dimension))
                for name, source in original.variables.items():
                    clone = native.createVariable(name, source.dtype, source.dimensions,
                        fill_value=source.getncattr('_FillValue') if '_FillValue' in source.ncattrs() else False)
                    clone.setncatts({key:source.getncattr(key) for key in source.ncattrs() if key != '_FillValue'})
                    is_time = name in {'time',bound_name}
                    _copy_variable(source, clone, source_time, is_time)
                    if name in {'lon','lat','time'}:
                        clone.cartesian_axis = {'lon':'X','lat':'Y','time':'T'}[name]
                    if is_time:
                        clone.calendar = 'gregorian'
                        clone.units = MOM_TIME_UNITS
                if interpretation == 'mean':
                    time = original['time']
                    bounds = native[time.bounds][:]
                    for name, column in [('average_T1',0),('average_T2',1)]:
                        bound = native.createVariable(name, 'f8', ('time',))
                        bound.units = MOM_TIME_UNITS
                        bound[:] = bounds[:,column]
                    native[variable].time_avg_info = 'average_T1,average_T2'
                    native[variable].zhenmode_interval_sampling = phase
                    native[variable].zhenmode_interval_seconds = np.int32(interval)
            final_sha = sha256_file(path)
            if _file_identity(path) != before or final_sha != original_sha:
                raise ValueError('source bytes changed during time conversion: '+field)
            report['files'][field] = {'path':target.name, 'original_sha256':original_sha,
                'source_revalidated_after_copy':True,
                'sha256':sha256_file(target), 'bytes':target.stat().st_size,
                'mean_phase':phase if interpretation == 'mean' else None,
                'interval_seconds':interval if interpretation == 'mean' else None,
                'flux_values_unchanged':True, 'physical_datetimes_unchanged':True,
                'original_time_units':original_time_units, 'native_time_units':MOM_TIME_UNITS,
                'calendar':'gregorian'}
        report['status'] = 'prepared'
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', reason_type=type(error).__name__, reason=str(error))
        raise
    finally:
        (output/'time-inputs.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def corrected_time_reader(source):
    """Opt-in full-interval selection, including one-record daily means.

    Native inputs must explicitly declare phase, interval seconds and FMS
    average_T1/T2. An interval crossing an input boundary fails; nothing is
    silently interpolated, extrapolated or treated as an indefinite constant.
    """
    if hashlib.sha256(source).hexdigest() != FMS_SHA256:
        raise ValueError('FMS time reader does not match the audited pin')
    text = source.decode('utf-8')

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError('ambiguous pinned FMS time adaptation: ' + old)
        text = text.replace(old, new)

    replace('get_variable_size, get_time_calendar, get_variable_missing, get_variable_units',
            'get_variable_size, get_time_calendar, get_variable_missing, get_variable_units, &\n'
            '                               get_variable_attribute')
    replace('     integer :: nbuf\n',
            '     integer :: nbuf\n'
            '     character(len=16) :: omip_phase = ""\n'
            '     integer :: omip_interval_seconds = 0\n')
    replace('      character(len=128) :: timename, timeunits',
            '      character(len=128) :: timename, timeunits, omip_methods, omip_bound_units')
    replace('      field(num_fields)%name = trim(fieldname)',
            '      field(num_fields)%omip_phase = ""\n'
            '      field(num_fields)%omip_interval_seconds = 0\n'
            "      if (variable_att_exists(fileobj, fieldname, 'zhenmode_interval_sampling')) then\n"
            "         call get_variable_attribute(fileobj, fieldname, 'zhenmode_interval_sampling', &\n"
            '                                    field(num_fields)%omip_phase)\n'
            '         if (trim(field(num_fields)%omip_phase) /= "left_endpoint" .and. &\n'
            '             trim(field(num_fields)%omip_phase) /= "right_endpoint") &\n'
            '              call mpp_error(FATAL, "OMIP requires an explicit interval phase")\n'
            "         if (.not.variable_att_exists(fileobj, fieldname, 'zhenmode_interval_seconds') .or. &\n"
            "             .not.variable_att_exists(fileobj, fieldname, 'time_avg_info') .or. &\n"
            "             .not.variable_att_exists(fileobj, fieldname, 'cell_methods') .or. &\n"
            "             .not.variable_exists(fileobj, 'average_T1') .or. &\n"
            "             .not.variable_exists(fileobj, 'average_T2')) &\n"
            '              call mpp_error(FATAL, "OMIP bounded means require explicit interval and bounds")\n'
            "         call get_variable_attribute(fileobj, fieldname, 'cell_methods', omip_methods)\n"
            '         if (trim(omip_methods) /= "time: mean") &\n'
            '              call mpp_error(FATAL, "OMIP bounded input must declare time: mean")\n'
            "         call get_variable_units(fileobj, 'average_T1', omip_bound_units)\n"
            '         if (trim(omip_bound_units) /= trim(timeunits)) &\n'
            '              call mpp_error(FATAL, "OMIP lower bound units disagree with time")\n'
            "         call get_variable_units(fileobj, 'average_T2', omip_bound_units)\n"
            '         if (trim(omip_bound_units) /= trim(timeunits)) &\n'
            '              call mpp_error(FATAL, "OMIP upper bound units disagree with time")\n'
            "         call get_variable_attribute(fileobj, fieldname, 'zhenmode_interval_seconds', &\n"
            '                                    field(num_fields)%omip_interval_seconds)\n'
            '         if (field(num_fields)%omip_interval_seconds <= 0) &\n'
            '              call mpp_error(FATAL, "OMIP interval seconds must be positive")\n'
            '      endif\n'
            '      field(num_fields)%name = trim(fieldname)')
    replace('      if (field(index)%siz(4) == 1) then\n'
            '         ! only one record in the file => time-independent field\n'
            '         call load_record(field(index),1,horz_interp, is_in, ie_in ,js_in, je_in,window_id)\n'
            '         i1 = find_buf_index(1,field(index)%ibuf)',
            '      if (field(index)%siz(4) == 1 .or. field(index)%omip_interval_seconds > 0) then\n'
            '         t1 = 1\n'
            '         if (field(index)%omip_interval_seconds > 0) t1 = omip_mean_record(field(index), time)\n'
            '         call load_record(field(index),t1,horz_interp, is_in, ie_in ,js_in, je_in,window_id)\n'
            '         i1 = find_buf_index(t1,field(index)%ibuf)')
    # The bounded API is specifically for 2D/3D weather/flux data_override.
    # Refuse a scalar call instead of silently bypassing the interval contract.
    replace('      if (field(index)%siz(4) == 1) then\n'
            '         ! only one record in the file => time-independent field\n'
            '         call load_record_0d(field(index),1)',
            '      if (field(index)%omip_interval_seconds > 0) &\n'
            '          call mpp_error(FATAL, "OMIP bounded scalar means are unsupported")\n'
            '      if (field(index)%siz(4) == 1) then\n'
            '         ! only one record in the file => time-independent field\n'
            '         call load_record_0d(field(index),1)')
    replace('end module time_interp_external2_mod', '''! Explicit input interval is checked before using a held record.
integer function omip_mean_record(input, time) result(record)
  type(ext_fieldtype), intent(in) :: input
  type(time_type), intent(in) :: time
  type(time_type) :: lower, upper, interval
  integer :: j, matches
  interval = set_time(input%omip_interval_seconds, 0)
  if (trim(input%omip_phase) == "right_endpoint") then
     lower = time - interval
     upper = time
  else
     lower = time
     upper = time + interval
  endif
  record = -1
  matches = 0
  do j = 1, size(input%time)
     if (lower >= input%start_time(j) .and. input%end_time(j) >= upper) then
        matches = matches + 1
        record = j
     endif
  enddo
  if (matches /= 1) call mpp_error(FATAL, &
       "OMIP mean interval crosses bounds, is uncovered or ambiguous: "//trim(input%name))
end function omip_mean_record

end module time_interp_external2_mod''')
    return text.encode('utf-8')
