"""Prepare synthetic native inputs; never launches either ocean model.

The Coriolis value must come from the frozen scoring protocol. Array labels
x_m/y_m are Cartesian metres, never geographic longitude/latitude.
"""
import argparse
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np


def native_arrays(f0):
    if not np.isfinite(f0):
        raise ValueError('f0 must be finite and explicitly supplied')
    if f0 != 0.:
        raise ValueError('this frozen standing-wave protocol requires f0=0')
    nx, ny = 64, 8
    length_x = 32000.0 * np.sqrt(9.81 * 100.0)
    dx, dy = length_x / nx, 100000.0 / ny
    x = (np.arange(nx) + .5) * dx
    y = (np.arange(ny) + .5) * dy
    z = np.array([0., -100. / 3., -200. / 3., -100.])
    edges = np.concatenate(([0.], .5 * (z[:-1] + z[1:]), [-100.]))
    h0 = -np.diff(edges)
    eta = np.broadcast_to(.01 * np.cos(2. * np.pi * x / length_x)[:, None],
                          (nx, ny)).copy()
    interfaces = np.broadcast_to(edges, (nx, ny, 5)).copy()
    interfaces[..., 0] = eta
    zeros = np.zeros((nx, ny, 4))
    return dict(x_m=x, y_m=y, z_m=z, dz_m=-np.diff(z), h0_m=h0,
                eta_m=eta, interfaces_m=interfaces, u_m_s=zeros.copy(),
                v_m_s=zeros.copy(), T_degC=np.full_like(zeros, 15.),
                S_psu=np.full_like(zeros, 35.), dx_2d_m=np.full((nx, ny), dx),
                dy_m=np.array(dy), cos_metric=np.ones(ny),
                f_s_inverse=np.full((nx, ny), f0),
                area_m2=np.full((nx, ny), dx * dy),
                depth_m=np.full((nx, ny), 100.), wet=np.ones((nx, ny, 4)))


def write_mom_initial(path, arrays):
    """MOM native file contract: (Interface/Layer, y, x), positive-up eta."""
    mom_interfaces = arrays['interfaces_m'].copy()
    mom_interfaces[..., 0] *= np.sinc(1. / 64.)
    with netCDF4.Dataset(path, 'w', format='NETCDF3_64BIT_OFFSET') as ds:
        for label, size in [('x', 64), ('y', 8), ('Layer', 4), ('Interface', 5)]:
            ds.createDimension(label, size)
        for label, key in [('x', 'x_m'), ('y', 'y_m')]:
            var = ds.createVariable(label, 'f8', (label,))
            var.units = 'm'
            var.cartesian_axis = label.upper()
            var[:] = arrays[key]
        for label, coordinates in [('Interface', np.array([0., -100/6, -50., -500/6, -100.])),
                                   ('Layer', np.array([-100/12, -100/3, -200/3, -1100/12]))]:
            var = ds.createVariable(label, 'f8', (label,))
            var.units = 'm'
            var.cartesian_axis = 'Z'
            var[:] = coordinates
        for label, key, dim, units in [
                ('eta', 'interfaces_m', 'Interface', 'm'),
                ('PTEMP', 'T_degC', 'Layer', 'degC'),
                ('SALT', 'S_psu', 'Layer', 'psu')]:
            var = ds.createVariable(label, 'f8', (dim, 'y', 'x'))
            var.units = units
            value = mom_interfaces if label == 'eta' else arrays[key]
            var[:] = value.transpose(2, 1, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--f0', type=float, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    arrays = native_arrays(args.f0)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    np.savez(args.output_directory / 'ocean_native_inputs.npz', **arrays)
    write_mom_initial(args.output_directory / 'mom_native_initial.nc', arrays)
    identities = {}
    for path in args.output_directory.iterdir():
        identities[path.name] = dict(bytes=path.stat().st_size,
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    contract = dict(schema='ocean.flat_f0_preparation.v1',
                    status='inputs_only_not_runtime_qualified', f0_s_inverse=args.f0,
                    nx=64, ny=8, nz=4, dt_s=100, steps=320,
                    sample_interval_s=1000, H_m=100.,
                    Lx_m=float(arrays['dx_2d_m'][0, 0] * 64), Ly_m=100000.,
                    g_m_s2=9.81, rho0_kg_m3=1025.,
                    drho_dT=-.205, drho_dS=.779,
                    ocean_geometry='nodal_dual_v1',
                    ocean_eta_sampling='node', mom_eta_sampling='cell_mean',
                    mom_eta_sinc_factor=float(np.sinc(1. / 64.)),
                    z_m=arrays['z_m'].tolist(), h0_m=arrays['h0_m'].tolist(),
                    input_identities=identities,
                    unresolved=['frozen scorer and acceptance gates',
                                'MOM resolved configuration and instantaneous diagnostics',
                                'complete ocean solver process time options and gates',
                                'native y-wall placement comparison',
                                'descendant RSS/output budget guard'],
                    qualification_passed=False)
    (args.output_directory / 'contract.json').write_text(
        json.dumps(contract, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
