"""Serialize only complete native trajectories into the independent scorer schema."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'standing_wave_v0'))
from native_mom import inspect_run, read_field, read_native_states  # noqa: E402
from score import MOM6, SCHEMA, digest, validate  # noqa: E402


def identity(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(frozen, model, guard, initialization, integration, identities, numerics):
    return dict(schema=SCHEMA, contract_sha256=digest(frozen), model=model,
                **identities, coordinate_system='cartesian_m',
                units=dict(time='s', x='m', eta='m', h='m', u='m/s', volume='m3',
                           area='m2', T='degC', S='psu'), snapshot_kind='instantaneous',
                sampling_eta='node' if model == 'ocean-solver' else 'cell_mean',
                sampling_u='node', velocity_layout='collocated' if model == 'ocean-solver' else 'cgrid',
                dt_s=100., steps=320, full_dynamics=True,
                zero_processes=frozen['explicitly_zero'], recorded_numerics=numerics,
                initialization_s=initialization, integration_s=integration,
                total_wall_s=guard['wall_s'], aggregate_peak_bytes=guard['sampled_aggregate_peak_rss_bytes'],
                cpu=1, ranks=1, trajectory=dict(kind='continuous', processes=1),
                g=frozen['gravity'], rho0=frozen['rho0'], H=frozen['H_m'],
                Lx=frozen['Lx_m'], Ly=frozen['Ly_m'], f=frozen['f'],
                eos=frozen['eos'], boundary=frozen['boundary'])


def native_quadrature(h, x, y, area, layout, faces=None):
    nt = h.shape[0]
    if layout == 'collocated':
        return dict(x_u=x.copy(), y_u=y.copy(), x_v=x.copy(), y_v=y.copy(),
                    width_u=np.zeros_like(x), volume_u=h * area[None, :, None],
                    volume_v=h * area[None, :, None])
    hh = h.reshape(nt, 64, 8, 4)
    if faces is None or not np.all(area == area[0]):
        raise ValueError('actual face coordinates and verified uniform native area required')
    ux, uy, vx, vy = (faces[name] for name in ('x_u', 'y_u', 'x_v', 'y_v'))
    vu = .5 * (hh + np.roll(hh, 1, axis=1)) * area[0]
    vv = np.concatenate((.5 * hh[:, :, :1], .5 * (hh[:, :, :-1] + hh[:, :, 1:]),
                         .5 * hh[:, :, -1:]), axis=2) * area[0]
    return dict(x_u=ux.ravel(), y_u=uy.ravel(), x_v=vx.ravel(), y_v=vy.ravel(),
                width_u=np.zeros(512), volume_u=vu.reshape(nt, 512, 4),
                volume_v=vv.reshape(nt, 576, 4))


def ocean_arrays(directory, frozen, guard, source_sha):
    native = directory / 'native'
    receipt = json.loads((directory / 'trajectory_receipt.json').read_text())
    if receipt['completed_steps'] != 320:
        raise ValueError('incomplete production trajectory')
    index = [json.loads(line) for line in (native / 'snapshot_index.jsonl').read_text().splitlines()]
    times = np.arange(0., 32001., 1000.)
    if [entry['time_s'] for entry in index] != times.tolist():
        raise ValueError('incomplete native snapshot index')
    snapshots = []
    for entry in index:
        path = native / entry['filename']
        if identity(path) != entry['sha256'] or entry['gate'].get('passed') is not True:
            raise ValueError('native snapshot changed or gate failed')
        with np.load(path, allow_pickle=False) as archive:
            snapshots.append({name: archive[name] for name in ('eta', 'u', 'v', 'T', 'S', 'h_geometric_m')})
    with np.load(native / 'native_metrics.npz', allow_pickle=False) as metrics:
        xx, yy = np.meshgrid(metrics['x_m'], metrics['y_m'], indexing='ij')
        area = metrics['area_m2'].ravel()
    values = dict(time=times, x_eta=xx.ravel(), y_eta=yy.ravel(), area=area,
                  eta=np.stack([s['eta'] for s in snapshots]).reshape(33, 512))
    for name, field in [('h', 'h_geometric_m'), ('u', 'u'), ('v', 'v'), ('T', 'T'), ('S', 'S')]:
        values[name] = np.stack([s[field] for s in snapshots]).reshape(33, 512, 4)
    values.update(native_quadrature(values['h'], values['x_eta'], values['y_eta'], area, 'collocated'))
    configuration = receipt['native_configuration']
    values['metadata'] = metadata(frozen, 'ocean-solver', guard, receipt['initialization_s'],
                                  receipt['integration_s'], dict(source_sha=source_sha,
                                  executable_sha256=configuration['source_module_sha256'],
                                  input_sha256=configuration['input_sha256'],
                                  config_sha256=identity(native / 'actual_config.json')),
                                  dict(time_scheme=configuration['numerical_scheme'], transport='native full FD tracer/momentum',
                                       filters=configuration['filters'], vertical_coordinate=configuration['thickness'],
                                       substeps='mode_split=False; native legacy process subcycles',
                                       resolved_options=configuration['options']))
    return values


def mom_arrays(directory, frozen, guard, executable):
    inspection = inspect_run(directory)
    states = read_native_states(directory)
    times = np.arange(0., 32001., 1000.)
    values = dict(time=times)
    with netCDF4.Dataset(directory / 'native.nc') as ds:
        axes = {}
        for name, count, offset, spacing in [('xh', 64, .5, frozen['Lx_m']/64),
                                             ('yh', 8, .5, frozen['Ly_m']/8),
                                             ('xq', 65, 0., frozen['Lx_m']/64),
                                             ('yq', 9, 0., frozen['Ly_m']/8)]:
            axes[name] = read_field(ds, name, (name,))
            if ds[name].units not in ('m', 'meter', 'meters') or axes[name].shape != (count,):
                raise ValueError('native diagnostic face coordinate units or shape')
            np.testing.assert_allclose(axes[name], (np.arange(count)+offset)*spacing, rtol=0., atol=1.e-9)
        xx, yy = np.meshgrid(axes['xh'], axes['yh'], indexing='ij')
        ux, uy = np.meshgrid(axes['xq'][:-1], axes['yh'], indexing='ij')
        vx, vy = np.meshgrid(axes['xh'], axes['yq'], indexing='ij')
        faces = dict(x_u=ux.ravel(), y_u=uy.ravel(), x_v=vx.ravel(), y_v=vy.ravel())
        values.update(x_eta=xx.ravel(), y_eta=yy.ravel())
        seconds = np.asarray(ds['time'][:]) * (86400. if ds['time'].units.lower().startswith('days') else 1.)
        # MOM.F90:1098-1100 sends cycle-averaged ssh to SSH. The requested
        # native e field is the instantaneous interface geometry, not that
        # intrinsically averaged surface diagnostic (even with time: point).
        ssh = read_field(ds, 'e', ('time', 'zi', 'yh', 'xh'))[:, 0]
        if np.array_equal(seconds, times[1:]):
            with netCDF4.Dataset(directory / 'native_initial.nc') as initial:
                ssh = np.concatenate((np.asarray(initial['eta'][:])[:, 0], ssh), axis=0)
        elif not np.array_equal(seconds, times):
            raise ValueError('native MOM output timestamps differ')
        values['eta'] = ssh.transpose(0, 2, 1).reshape(33, 512)
    with netCDF4.Dataset(directory / 'ocean_geometry.nc') as geometry:
        values['area'] = read_field(geometry, 'Ah', ('lath', 'lonh')).T.ravel()
        for name, dims, shape, spacing in [('dxCu', ('lath', 'lonq'), (8, 65), frozen['Lx_m']/64),
                                           ('dyCu', ('lath', 'lonq'), (8, 65), frozen['Ly_m']/8),
                                           ('dxCv', ('latq', 'lonh'), (9, 64), frozen['Lx_m']/64),
                                           ('dyCv', ('latq', 'lonh'), (9, 64), frozen['Ly_m']/8)]:
            metric = read_field(geometry, name, dims)
            if geometry[name].units != 'm' or metric.shape != shape:
                raise ValueError('native face metric units or shape: ' + name)
            np.testing.assert_allclose(metric, spacing, rtol=1.e-12, atol=0.)
    for name in ('h', 'u', 'v', 'T', 'S'):
        field = np.concatenate([states[int(t)][name] for t in times], axis=0)
        if name == 'u':
            if not np.array_equal(field[:, :, :, 0], field[:, :, :, -1]):
                raise ValueError('native periodic u endpoints are not exact aliases')
            field = field[:, :, :, :-1]
        values[name] = field.transpose(0, 3, 2, 1).reshape(33, -1, 4)
    values.update(native_quadrature(values['h'], values['x_eta'], values['y_eta'], values['area'], 'cgrid', faces))
    log = (directory / 'guard_stdout.log').read_text()
    def native_clock(name):
        matches = re.findall(r'^' + re.escape(name) + r'\s+1\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+', log, re.M)
        if len(matches) != 1 or len(set(matches[0])) != 1:
            raise ValueError('missing one-rank native elapsed clock: ' + name)
        return float(matches[0][0])
    options = inspection['resolved_options']
    values['metadata'] = metadata(frozen, 'MOM6', guard, native_clock('Initialization'),
                                  native_clock('Main loop'), dict(source_sha=MOM6,
                                  executable_sha256=identity(executable),
                                  input_sha256=identity(directory / 'INPUT/initial.nc'),
                                  config_sha256=identity(directory / 'native_inspection.json')),
                                  dict(time_scheme='native split RK2; FMS elapsed Main loop includes diagnostics/checkpoint I/O',
                                       transport='native PPM continuity and tracer transport',
                                       filters='native BEBT=' + str(options['BEBT']) + '; all defaults in retained parameter documents',
                                       vertical_coordinate='four homogeneous fixed control volumes; eta is native e top interface, not cycle-mean SSH',
                                       substeps='actual DTBT/DT=100s; native complete-state output',
                                       resolved_options={key: options[key] for key in frozen['mom_time_options']}))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('ocean-solver', 'MOM6'), required=True)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sha')
    parser.add_argument('--executable', type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.contract.read_text())
    guard = json.loads((args.run_directory / 'guard_result.json').read_text())
    if guard['exit_code'] != 0 or guard['stopped_reason'] is not None or not guard['descendant_cleanup_confirmed']:
        raise ValueError('incomplete or unbounded native run')
    values = (ocean_arrays(args.run_directory, frozen, guard, args.source_sha)
              if args.model == 'ocean-solver' else mom_arrays(args.run_directory, frozen, guard, args.executable))
    validate(frozen, values)
    values['metadata'] = np.array(json.dumps(values['metadata'], sort_keys=True, allow_nan=False))
    with args.output.open('xb') as stream:
        np.savez(stream, **values)


if __name__ == '__main__':
    main()
