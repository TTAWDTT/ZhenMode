"""Read actual MOM files without evolving or interpolating model state."""
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
from prepare_inputs import native_arrays
from resolved_config import parse_resolved, verify_mom_geometry, verify_resolved_settings


def active_expected(requested, resolved):
    """Recognize only six source-proven dormant options, never assume defaults."""
    dormant = {
        'ISOTROPIC': (False, 'GRID_CONFIG', 'cartesian', 'MOM_grid_initialize: mercator-only reader'),
        'KH': (0., 'LAPLACIAN', False, 'MOM_hor_visc:2468-2471 do_not_log when not Laplacian'),
        'KH_VEL_SCALE': (0., 'LAPLACIAN', False, 'MOM_hor_visc:2476-2481 Laplacian-only'),
        'SMAGORINSKY_KH': (False, 'LAPLACIAN', False, 'MOM_hor_visc:2493-2496 explicitly disabled'),
        'SMAG_BI_CONST': (0., 'SMAGORINSKY_AH', False, 'MOM_hor_visc:2614-2617 inactive Smagorinsky'),
        'KD': (0., 'ADIABATIC', True, 'MOM:3752-3756 uses adiabatic_driver_init, not diabatic_driver_init'),
    }
    expected = dict(requested)
    evidence = {}
    for name, (value, control, setting, proof) in dormant.items():
        if name not in resolved:
            if (name not in expected or type(expected[name]) is not type(value)
                    or expected[name] != value or resolved.get(control) != setting):
                raise ValueError('unproven dormant requested option: ' + name)
            expected.pop(name)
            expected[control] = setting
            evidence[name] = {'requested': value, 'resolved_control': {control: setting},
                              'status': 'inactive; numeric value not runtime logged', 'pinned_source': proof}
    return expected, evidence


def read_field(dataset, name, dimensions):
    variable = dataset[name]
    values = np.ma.asarray(variable[:])
    if (variable.dimensions != dimensions or np.any(np.ma.getmaskarray(values))
            or values.dtype.itemsize != 8 or not np.all(np.isfinite(values))):
        raise ValueError('invalid native field: ' + name)
    return np.asarray(values)


def read_native_states(directory):
    """Full unmasked native IC/restart states, indexed by actual native Time."""
    directory = Path(directory)
    states = {}
    paths = [directory / 'native_initial.nc', *sorted((directory / 'RESTART').glob('*.nc'))]
    for path in paths:
        with netCDF4.Dataset(path) as ds:
            inputs = native_arrays(0.)
            for name, expected in [('lonh', inputs['x_m']), ('lath', inputs['y_m']),
                                   ('lonq', np.arange(65) * 1002269.4248554128 / 64.),
                                   ('latq', np.arange(9) * 100000. / 8.)]:
                coordinate = read_field(ds, name, (name,))
                if ds[name].units not in ('m', 'meter', 'meters'):
                    raise ValueError('native full-state coordinate units: ' + name)
                np.testing.assert_allclose(coordinate, expected, rtol=0., atol=1.e-9)
            time = read_field(ds, 'Time', ('Time',))
            if time.shape != (1,) or ds['Time'].units != 'days':
                raise ValueError('unknown native full-state time identity')
            seconds = float(time[0]) * 86400.
            rounded = round(seconds)
            if abs(seconds - rounded) > 1.e-8 or rounded in states:
                raise ValueError('duplicate or fractional native state time')
            state = {}
            for name, actual, dims in [
                    ('T', 'Temp', ('Time', 'Layer', 'lath', 'lonh')),
                    ('S', 'Salt', ('Time', 'Layer', 'lath', 'lonh')),
                    ('h', 'h', ('Time', 'Layer', 'lath', 'lonh')),
                    ('u', 'u', ('Time', 'Layer', 'lath', 'lonq')),
                    ('v', 'v', ('Time', 'Layer', 'latq', 'lonh'))]:
                state[name] = read_field(ds, actual, dims)
                shape = (1, 4, 9, 64) if name == 'v' else (1, 4, 8, 65) if name == 'u' else (1, 4, 8, 64)
                units = {'T': 'degC', 'S': 'PPT', 'h': 'm', 'u': 'm s-1', 'v': 'm s-1'}
                if state[name].shape != shape or ds[actual].units != units[name]:
                    raise ValueError('native full-state shape or units mismatch: ' + name)
            state['filename'] = str(path.relative_to(directory))
            state['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            states[rounded] = state
    return states


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
    expected, inactive = active_expected(request['settings'], resolved)
    receipt['inactive_requested_options'] = inactive
    receipt['settings_check'] = verify_resolved_settings(text, expected)
    receipt['geometry_check'] = verify_mom_geometry(directory / 'ocean_geometry.nc')
    receipt['initial_check'] = verify_initial(directory / 'native_initial.nc')
    full_states = read_native_states(directory)
    interval = 100 if request['preflight'] else 1000
    if set(full_states) != set(range(0, request['simulated_seconds'] + 1, interval)):
        raise ValueError('missing complete native restart snapshot')
    with netCDF4.Dataset(directory / 'native.nc') as ds:
        timestamps = read_field(ds, 'time', ('time',))
        unit = ds['time'].units.lower().split(' since ')[0]
        if unit not in ('days', 'seconds'):
            raise ValueError('unknown native diagnostic time units')
        epoch = netCDF4.num2date(0., ds['time'].units,
                                calendar=getattr(ds['time'], 'calendar', 'standard'))
        if (epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second) != (1, 1, 1, 0, 0, 0):
            raise ValueError('native epoch differs from diag_table t0')
        times_s = timestamps * (86400. if unit == 'days' else 1.)
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
            if name == 'v':
                if variable.dimensions != dimensions or variable.dtype != np.dtype('float64'):
                    raise ValueError('invalid native diagnostic v layout')
                masked = np.ma.asarray(variable[:])
                mask = np.ma.getmaskarray(masked)
                walls = np.zeros(shape, dtype=bool)
                walls[:, :, 0, :] = walls[:, :, -1, :] = True
                if mask.shape != shape or not np.array_equal(mask, walls):
                    raise ValueError('diagnostic v missing outside known closed-wall faces')
                # No masked diagnostic value is filled. The scorer consumes
                # full unmasked native restart values at each actual timestamp.
                value = np.stack([full_states[int(round(t))]['v'][0] for t in times_s])
                if (np.any(value[:, :, (0, -1), :] != 0.)
                        or not np.array_equal(value[:, :, 1:-1, :], masked.data[:, :, 1:-1, :])):
                    raise ValueError('native complete v differs from diagnostic interior or closed walls')
            else:
                value = read_field(ds, name, dimensions)
            if value.shape != shape:
                raise ValueError('incomplete native snapshot layout: ' + name)
            if 'time: point' not in getattr(variable, 'cell_methods', ''):
                raise ValueError('native snapshot is not explicitly instantaneous: ' + name)
            if name in ('u', 'h', 'temp', 'salt'):
                key = {'temp': 'T', 'salt': 'S'}.get(name, name)
                native = np.stack([full_states[int(round(t))][key][0] for t in times_s])
                if not np.array_equal(value, native):
                    raise ValueError('diagnostic differs from full native state: ' + name)
    receipt['full_state_identities'] = {str(t): {'filename': s['filename'], 'sha256': s['sha256']}
                                        for t, s in full_states.items()}
    receipt['runtime_contract_passed'] = True
    (directory / 'native_inspection.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    inspect_run(args.directory)
