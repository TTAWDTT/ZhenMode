"""Read-only four-snapshot redistribution and historical pressure diagnostics."""
# ruff: noqa: E402
import os

os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import argparse
import hashlib
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from scipy.ndimage import label


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_verified(path, expected):
    content = path.read_bytes()
    if digest(content) != expected:
        raise ValueError(f"identity mismatch: {path.name}")
    return content


def unpack(content):
    with np.load(BytesIO(content), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def connected_basins(wet, area):
    labels, count = label(wet)
    parent = list(range(count + 1))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in zip(labels[0], labels[-1], strict=True):
        if left and right:
            parent[root(int(right))] = root(int(left))
    merged = np.array([root(index) for index in range(count + 1)])[labels]
    identifiers = sorted(set(merged[wet]), key=lambda i: (-float(area[merged == i].sum()), i))
    return {f"basin_{n + 1}": merged == identifier for n, identifier in enumerate(identifiers)}


def gradient(field, arrays, scalars):
    wet = arrays['wet_mask_z']
    cosine = arrays['cos_lat']
    xp = wet * np.roll(wet, -1, axis=0)
    xm = wet * np.roll(wet, 1, axis=0)
    yp = wet * np.roll(wet, -1, axis=1)
    ym = wet * np.roll(wet, 1, axis=1)
    yp[:, -1] = 0
    ym[:, 0] = 0
    gx = arrays['inv_dx'][..., :1] * .5 * (
        (np.roll(field, -1, axis=0) - field) * xp +
        (field - np.roll(field, 1, axis=0)) * xm)
    gy = scalars['inv_dy'] / cosine[None, :, None] * .5 * (
        (np.roll(field, -1, axis=1) - field) * yp *
        (.5 * (cosine + np.roll(cosine, -1)))[None, :, None] +
        (field - np.roll(field, 1, axis=1)) * ym *
        (.5 * (cosine + np.roll(cosine, 1)))[None, :, None])
    return gx, gy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run_dir', 'archive_dir', 'source_dir', 'original_protocol',
                 'protocol', 'artifact_hashes', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    protocol_content = args.protocol.read_bytes()
    protocol = json.loads(protocol_content)
    original_content = args.original_protocol.read_bytes()
    resources = json.loads((args.run_dir / 'resources.json').read_bytes())
    if (digest(original_content) != '9d9e756162c8f19ac1d3e32bbfb178d3c44d13c4f11ccf83e697b7e9c716f148'
            or resources['protocol_sha256'] != digest(original_content)):
        raise ValueError('original run protocol identity mismatch')
    original = json.loads(original_content)
    evidence_hashes = json.loads(args.artifact_hashes.read_bytes())
    identities = {}
    for name, expected in original['historical_source_hashes'].items():
        if name == 'src/material_top.py':
            expected = original['instrumented_material_sha256']
        read_verified(args.source_dir / Path(name).name, expected)
    base = {}
    for name, expected in original['base_inputs'].items():
        base[name] = read_verified(args.archive_dir / name, expected)
        identities[name] = expected
    arrays, grid = unpack(base['parameter_arrays_used.npz']), unpack(base['grid_used.npz'])
    scalars = json.loads(base['manifest_start.json'])['scalar_params']
    sys.path.insert(0, str(args.source_dir))
    import jax.numpy as jnp

    import jax_solver_global as solver
    from config import ALPHA_T, BETA_S, G_EARTH, RHO_0

    params = solver.FDPhysParams(**{
        key: jnp.asarray(arrays[key]) if key in arrays else scalars[key]
        for key in solver.FDPhysParams._fields})
    wet3 = arrays['wet_mask_z'].astype(bool)
    wet = arrays['wet_mask'].astype(bool)
    if not np.array_equal(wet, wet3.any(axis=-1)) or not np.array_equal(wet, wet3[..., 0]):
        raise ValueError('wet surface connectivity contract mismatch')
    area = grid['dx_2d'] * scalars['dy']
    masks = {'whole_wet_domain': wet, **connected_basins(wet, area)}
    for low, high in [(-66, -58), (-58, -30), (-30, 30), (30, 66)]:
        masks[f'lat_{low}_{high}'] = wet & ((grid['lat'] >= low) & (grid['lat'] < high))[None, :]
    for low in range(0, 360, 90):
        masks[f'lon_{low}_{low + 90}'] = wet & ((grid['lon'] >= low) & (grid['lon'] < low + 90))[:, None]
    masks['south_8_rows'] = wet & (np.arange(wet.shape[1])[None, :] < 8)
    # Pressure uses original reference nodal weights, not moving eta thickness.
    weights = arrays['dz_norm']
    eps_factor = 4096 * np.finfo(np.float64).eps
    metric = max(float(arrays['inv_dx'].max()), scalars['inv_dy'] / float(arrays['cos_lat'].min()))

    def location(index):
        value = dict(index=[int(i) for i in index], lon_deg=float(grid['lon'][index[0]]),
                     lat_deg=float(grid['lat'][index[1]]))
        if len(index) == 3:
            value['z_m'] = float(grid['z'][index[2]])
        return value

    def stats(field, mask):
        values = field[mask]
        if not values.size:
            return None
        index = np.unravel_index(np.argmax(np.where(mask, np.abs(field), -np.inf)), field.shape)
        return dict(mean=float(np.sum(area[mask] * values) / area[mask].sum()),
                    rms=float(np.sqrt(np.sum(area[mask] * values**2) / area[mask].sum())),
                    min=float(values.min()), max=float(values.max()), abs_max_location=location(index))

    def vector_stats(vector, mask):
        return stats(np.hypot(*vector), mask)

    def depth_average(vector):
        return tuple(np.sum(component * weights, axis=-1) for component in vector)

    results = []
    eta0 = unpack(base['initial_state_used.npz'])['eta']
    previous_velocity = None
    for number in protocol['rounds']:
        relative = f'round_{number:03d}/' + ('I0B0.npz' if number == 0 else 'I0B0-attempted.npz')
        if number:
            marker_relative = f'round_{number:03d}/COMMITTED.json'
            marker_expected = evidence_hashes.get(marker_relative, evidence_hashes.get(marker_relative.replace('/', '\\')))
            marker = json.loads(read_verified(args.run_dir / marker_relative, marker_expected))
            if marker['round'] != number or marker['model_seconds'] != number * 300 or not marker['cells']['I0B0']['valid']:
                raise ValueError('checkpoint authority mismatch')
            expected = marker['checkpoint_sha256']['I0B0']
        else:
            expected = protocol['t0_checkpoint_sha256']
        content = read_verified(args.run_dir / relative, expected)
        identities[relative] = expected
        state_arrays = unpack(content)
        if number == 0:
            initial = unpack(base['initial_state_used.npz'])
            if any(state_arrays[key].tobytes() != initial[key].tobytes() for key in initial):
                raise ValueError('t0 differs from original initial state')
        state = solver.JaxStateG(**{key: jnp.asarray(value) for key, value in state_arrays.items()})
        delta = state_arrays['eta'] - eta0
        rho = RHO_0 * (-ALPHA_T * (state_arrays['T'] - scalars['T_ref']) +
                       BETA_S * (state_arrays['S'] - scalars['S_ref'])) * wet3
        pressure_density = np.zeros_like(rho)
        pressure_density[..., 1:] = np.cumsum(
            G_EARTH * .5 * (rho[..., :-1] + rho[..., 1:]) * arrays['dz_3d'], axis=-1)
        pressure_surface = RHO_0 * G_EARTH * state_arrays['eta'][..., None]
        pressure = pressure_density + pressure_surface
        reference_pressure = np.asarray(solver._compute_hydrostatic_pressure(state, params))
        density_force = tuple(-component / RHO_0 for component in gradient(pressure_density, arrays, scalars))
        surface_force = tuple(-component / RHO_0 for component in gradient(pressure_surface, arrays, scalars))
        total_force = tuple(a + b for a, b in zip(density_force, surface_force, strict=True))
        reference_force = tuple(np.asarray(a) for a in solver._compute_pressure_gradient(state, params))
        shifted_force = tuple(-a / RHO_0 for a in gradient(pressure + RHO_0 * G_EARTH, arrays, scalars))
        pressure_error = float(np.max(np.abs(reference_pressure - pressure)[wet3]))
        force_error = max(float(np.max(np.abs(a - b)[wet3])) for a, b in zip(total_force, reference_force, strict=True))
        gauge_error = max(float(np.max(np.abs(a - b)[wet3])) for a, b in zip(total_force, shifted_force, strict=True))
        pressure_bound = eps_factor * max(1., float(np.max(np.abs(pressure))))
        force_bound = eps_factor * max(1., float(np.max(np.abs(pressure))), RHO_0 * G_EARTH) * metric / RHO_0
        forces = {'density': depth_average(density_force), 'surface': depth_average(surface_force),
                  'pressure_sum': depth_average(total_force)}
        velocity = depth_average((state_arrays['u'], state_arrays['v']))
        coriolis = (arrays['f'] * velocity[1], -arrays['f'] * velocity[0])
        # f may be stored as a broadcast-ready 3D array; use the column field.
        if coriolis[0].ndim != 2:
            raise ValueError('unexpected Coriolis shape')
        forces['coriolis'] = coriolis
        forces['pressure_plus_coriolis'] = tuple(a + b for a, b in zip(forces['pressure_sum'], coriolis, strict=True))
        if previous_velocity is not None:
            forces['coarse_interval_eulerian_acceleration'] = tuple(
                (a - b) / 7200 for a, b in zip(velocity, previous_velocity, strict=True))
        previous_velocity = velocity
        volumes = {}
        regions = {}
        for name, mask in masks.items():
            volume = float(np.sum(area[mask] * delta[mask]))
            bound = eps_factor * max(1., float(np.sum(np.abs(area[mask] * delta[mask]))))
            volumes[name] = dict(wet_columns=int(mask.sum()), area_m2=float(area[mask].sum()),
                                 eta=stats(delta, mask), delta_volume_m3=volume,
                                 positive_volume_m3=float(np.sum(area[mask] * np.maximum(delta[mask], 0))),
                                 negative_volume_m3=float(np.sum(area[mask] * np.minimum(delta[mask], 0))),
                                 arithmetic_bound_m3=bound,
                                 closed_component_volume_check=bool(abs(volume) <= bound) if name.startswith('basin_') or name == 'whole_wet_domain' else None)
            regions[name] = dict(
                density_pressure_mean_Pa=stats(np.sum(pressure_density * weights, axis=-1), mask),
                surface_pressure_Pa=stats(pressure_surface[..., 0], mask),
                column_speed_m_s=vector_stats(velocity, mask),
                column_acceleration_m_s2={key: vector_stats(value, mask) for key, value in forces.items()})
        speed = np.hypot(state_arrays['u'], state_arrays['v'])
        maximum_index = np.unravel_index(np.argmax(np.where(wet3, speed, -np.inf)), speed.shape)
        target = (302, 2)
        result = dict(round=number, hours=number / 12, volumes=volumes, regions=regions,
                      max_wet_vector_speed_m_s=float(speed[maximum_index]), max_speed_location=location(maximum_index),
                      target=dict(eta_m=float(state_arrays['eta'][target]),
                                  column_acceleration_m_s2={key: [float(a[target]) for a in value] for key, value in forces.items()}),
                      arithmetic=dict(pressure_error_Pa=pressure_error, pressure_bound_Pa=pressure_bound,
                                      force_error_m_s2=force_error, gauge_error_m_s2=gauge_error, force_bound_m_s2=force_bound,
                                      pressure_check=bool(pressure_error <= pressure_bound),
                                      force_check=bool(force_error <= force_bound), gauge_check=bool(gauge_error <= force_bound)))
        results.append(result)
        print(json.dumps({'hours': result['hours'], 'volume': volumes['whole_wet_domain']['delta_volume_m3'],
                          'arithmetic': result['arithmetic']}), flush=True)
    report = dict(protocol_sha256=digest(protocol_content), original_protocol_sha256=digest(args.original_protocol.read_bytes()),
                  source_commit=original['source_commit'], identities=identities,
                  connectivity_masks_sha256={key: digest(mask.tobytes()) for key, mask in masks.items() if key.startswith('basin_')},
                  results=results, wall_seconds=time.perf_counter() - started,
                  limitations=protocol['dynamics'], correction=protocol['correction'])
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
