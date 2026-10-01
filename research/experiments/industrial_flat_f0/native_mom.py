"""Read actual MOM files without evolving or interpolating model state."""
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
from prepare_inputs import native_arrays
from resolved_config import parse_resolved, verify_mom_geometry, verify_resolved_settings


def read_field(dataset, name, dimensions):
    variable = dataset[name]
    values = np.ma.asarray(variable[:])
    if (variable.dimensions != dimensions or np.any(np.ma.getmaskarray(values))
            or values.dtype.itemsize != 8 or not np.all(np.isfinite(values))):
        raise ValueError('invalid native field: ' + name)
    return np.asarray(values)


def verify_initial(path):
    """The native restart-registry IC has Time/Layer/Interface staggering."""
    inputs = native_arrays(0.)
    expected_eta = inputs['interfaces_m'].transpose(2, 1, 0).copy()
    expected_eta[0] *= np.sinc(1. / 64.)
    with netCDF4.Dataset(path) as ds:
        for name, expected in [('lonh', inputs['x_m']), ('lath', inputs['y_m'])]:
            if ds[name].units not in ('m', 'meter', 'meters'):
                raise ValueError('native IC coordinate units: ' + name)
            coordinate = read_field(ds, name, (name,))
            np.testing.assert_allclose(coordinate, expected, rtol=0., atol=1.e-9)
        time = read_field(ds, 'Time', ('Time',))
        if time.shape != (1,) or time[0] != 0. or ds['Time'].units != 'days':
            raise ValueError('native IC is not t0')
        eta = read_field(ds, 'eta', ('Time', 'Interface', 'lath', 'lonh'))
        np.testing.assert_allclose(eta[0], expected_eta, rtol=0., atol=1.e-10)
        for name, expected in [('Temp', 15.), ('Salt', 35.)]:
            value = read_field(ds, name, ('Time', 'Layer', 'lath', 'lonh'))
            if value.shape != (1, 4, 8, 64) or not np.allclose(value, expected, rtol=0., atol=1.e-12):
                raise ValueError('native IC tracer mismatch: ' + name)
        h = read_field(ds, 'h', ('Time', 'Layer', 'lath', 'lonh'))
        np.testing.assert_allclose(h[0], -np.diff(expected_eta, axis=0), rtol=0., atol=1.e-10)
        for name, dims, shape in [('u', ('Time', 'Layer', 'lath', 'lonq'), (1, 4, 8, 65)),
                                  ('v', ('Time', 'Layer', 'latq', 'lonh'), (1, 4, 9, 64))]:
            value = read_field(ds, name, dims)
            if value.shape != shape or np.any(value != 0.):
                raise ValueError('native IC velocity mismatch: ' + name)
    return {'native_t0_verified': True, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def inspect_run(directory):
    """Fail closed on unresolved requested options; retain original documents."""
    directory = Path(directory)
    request = json.loads((directory / 'requested_config.json').read_text())
    guard = json.loads((directory / 'guard_result.json').read_text())
    if (guard['exit_code'] != 0 or guard['stopped_reason'] is not None
            or guard['descendant_cleanup_confirmed'] is not True):
        raise ValueError('native run did not exit successfully within the guard')
    for name, identity in request['input_identities'].items():
        path = directory / name
        if (path.stat().st_size != identity['bytes']
                or hashlib.sha256(path.read_bytes()).hexdigest() != identity['sha256']):
            raise ValueError('run input identity changed: ' + name)
    documents = [directory / ('MOM_parameter_doc.' + suffix)
                 for suffix in ('all', 'debugging', 'layout')]
    text = '\n'.join(path.read_text() for path in documents)
    resolved = parse_resolved(text)
    receipt = {'requested_options': request['settings'], 'resolved_options': resolved,
               'document_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in documents},
               'qualification_passed': False}
    # Save evidence before validation; failed receipts never acquire a pass marker.
    (directory / 'native_inspection.json').write_text(json.dumps(receipt, indent=2) + '\n')
    receipt['settings_check'] = verify_resolved_settings(text, request['settings'])
    receipt['geometry_check'] = verify_mom_geometry(directory / 'ocean_geometry.nc')
    receipt['initial_check'] = verify_initial(directory / 'native_initial.nc')
    with netCDF4.Dataset(directory / 'native.nc') as ds:
        timestamps = read_field(ds, 'time', ('time',))
        if not ds['time'].units.lower().startswith('days since'):
            raise ValueError('unknown native diagnostic time units')
        epoch = netCDF4.num2date(0., ds['time'].units,
                                calendar=getattr(ds['time'], 'calendar', 'standard'))
        if (epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second) != (1, 1, 1, 0, 0, 0):
            raise ValueError('native epoch differs from diag_table t0')
        times_s = timestamps * 86400.
        expected = np.arange(0., request['simulated_seconds'] + 1.,
                             100. if request['preflight'] else 1000.)
        # Some native writers begin at the first interval; t0 then comes only
        # from the separately verified native initial file, never the oracle.
        if timestamps.shape == expected[1:].shape:
            expected = expected[1:]
        if times_s.shape != expected.shape or not np.allclose(times_s, expected, rtol=0., atol=1.e-8):
            raise ValueError('incomplete instantaneous native time axis')
        layouts = {
            'u': (('time', 'zl', 'yh', 'xq'), (len(expected), 4, 8, 65)),
            'v': (('time', 'zl', 'yq', 'xh'), (len(expected), 4, 9, 64)),
            'h': (('time', 'zl', 'yh', 'xh'), (len(expected), 4, 8, 64)),
            'e': (('time', 'zi', 'yh', 'xh'), (len(expected), 5, 8, 64)),
            'temp': (('time', 'zl', 'yh', 'xh'), (len(expected), 4, 8, 64)),
            'salt': (('time', 'zl', 'yh', 'xh'), (len(expected), 4, 8, 64)),
            'SSH': (('time', 'yh', 'xh'), (len(expected), 8, 64)),
        }
        for name, (dimensions, shape) in layouts.items():
            variable = ds[name]
            value = read_field(ds, name, dimensions)
            if value.shape != shape:
                raise ValueError('incomplete native snapshot layout: ' + name)
            if 'time: point' not in getattr(variable, 'cell_methods', ''):
                raise ValueError('native snapshot is not explicitly instantaneous: ' + name)
    receipt['runtime_contract_passed'] = True
    (directory / 'native_inspection.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    inspect_run(args.directory)
