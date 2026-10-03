"""Export shared model forcing to MOM6 A-grid NetCDF inputs."""
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from ocean_solver.config.definitions import GlobalGridConfig
from ocean_solver.forcing.air import load_monthly_mean_air_temp
from ocean_solver.forcing.wind import real_wind_forcing
from ocean_solver.io.climatology import get_initial_fields
from ocean_solver.io.grid import make_global_grid


def add_arguments(parser):
    parser.add_argument('--kind', choices=('wind', 'air-temperature', 'sensible-heat'), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--bathy', required=True)
    parser.add_argument('--resolution', type=float, default=0.5)
    parser.add_argument('--lat-max', type=float, default=65.0)
    parser.add_argument('--year', type=int, default=2023)
    parser.add_argument('--smooth-passes', type=int, default=80)
    parser.add_argument('--min-depth', type=float, default=500.0)
    parser.add_argument('--taper-cells', type=int, default=8)
    parser.add_argument('--lambda-bulk', type=float, default=80.0)


def export_forcing(args):
    """Keep existing monthly values, units and 30-day mapping explicit.

    The sensible-heat input is a fixed-SST proxy, not live air-sea coupling.
    Native comparison contracts still have to check physics and input identity.
    """
    if args.kind not in ('wind', 'air-temperature', 'sensible-heat'):
        raise ValueError('unknown MOM6 forcing kind')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    config = replace(GlobalGridConfig(), lat_max=args.lat_max, resolution=args.resolution,
                     nx=int(round(360 / args.resolution)),
                     ny=int(round(2 * args.lat_max / args.resolution)))
    grid = make_global_grid(config, args.bathy, smooth_passes=args.smooth_passes,
                            min_depth=args.min_depth, remap='area')
    air = sst = None
    if args.kind != 'wind':
        air = load_monthly_mean_air_temp(grid, year=args.year)
    if args.kind == 'sensible-heat':
        temperature, _ = get_initial_fields(grid)
        sst = np.asarray(temperature[:, :, 0], dtype=float)
    fields = {
        'wind': (('STRESS_X', 'N m-2', 'zonal wind stress on A-grid'),
                 ('STRESS_Y', 'N m-2', 'meridional wind stress on A-grid')),
        'air-temperature': (('AIR_TEMP', 'degC', 'NCEP R1 2m air temperature on A-grid'),),
        'sensible-heat': (('sensible', 'W m-2',
                           'lambda*(2m air - WOA SST) bulk-like sensible-heat proxy'),),
    }[args.kind]
    out.parent.mkdir(parents=True, exist_ok=True)
    month0 = (args.year - 1948) * 12
    with Dataset(out, 'w') as dataset:
        dataset.createDimension('time', None)
        dataset.createDimension('y', grid.ny)
        dataset.createDimension('x', grid.nx)
        time = dataset.createVariable('time', 'f8', ('time',))
        time.units = 'days since 0001-01-01 00:00:00'
        time.calendar = 'julian'
        for name, values, units in (('x', grid.lon, 'degrees_east'),
                                    ('y', grid.lat, 'degrees_north')):
            coordinate = dataset.createVariable(name, 'f8', (name,))
            coordinate.units = units
            coordinate[:] = values
        variables = []
        for name, units, description in fields:
            variable = dataset.createVariable(name, 'f4', ('time', 'y', 'x'))
            variable.units = units
            variable.long_name = description
            variables.append(variable)
        for month in range(12):
            if args.kind == 'wind':
                values = real_wind_forcing(grid, month_idx=month0 + month,
                                           taper_cells=args.taper_cells)
            elif args.kind == 'air-temperature':
                values = (air[month],)
            else:
                values = (args.lambda_bulk * (air[month] - sst),)
            for variable, value in zip(variables, values, strict=True):
                variable[month, :, :] = np.ascontiguousarray(value.T, dtype=np.float32)
            time[month] = 1.0 + 30.0 * month + 15.0
        if args.kind == 'wind':
            dataset.title = 'NCEP R1 10m wind converted to bulk wind stress'
            dataset.ocean_solver_month_mapping = f'{month0}..{month0 + 11}'
            dataset.taper_cells = args.taper_cells
        elif args.kind == 'air-temperature':
            dataset.title = 'NCEP R1 monthly 2m air temperature'
        else:
            dataset.title = 'Ocean_solver-like bulk sensible heat proxy'
            dataset.lambda_bulk_w_m2_k = args.lambda_bulk
    return {'path': str(out), 'kind': args.kind, 'records': 12,
            'nx': grid.nx, 'ny': grid.ny,
            'time_mapping': '16 + 30 * month days, julian calendar',
            'physics': 'fixed_SST_heat_proxy' if args.kind == 'sensible-heat' else args.kind}


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    add_arguments(parser)
    print(export_forcing(parser.parse_args(argv)))


if __name__ == '__main__':
    main()
