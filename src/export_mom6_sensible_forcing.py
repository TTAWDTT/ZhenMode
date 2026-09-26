"""Export an approximate MOM6 sensible-heat-flux proxy from NCEP air."""
import os
from argparse import ArgumentParser
from dataclasses import replace

import numpy as np
from netCDF4 import Dataset

from air_reanalysis import load_monthly_mean_air_temp
from config import GlobalGridConfig
from grid import make_global_grid
from woa_data import get_initial_fields


def main():
    p = ArgumentParser()
    p.add_argument("--out", default="INPUT/ncep_2023_sens_050.nc")
    p.add_argument("--bathy", default=r"C:\Users\zhen.luo\Desktop\ETOPO_2022_v1_r3600x1800_surface.nc")
    p.add_argument("--woa-dir", default=r"C:\Users\zhen.luo\ocean_solver\data\woa")
    p.add_argument("--resolution", type=float, default=0.5)
    p.add_argument("--lat-max", type=float, default=65.0)
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--lambda-bulk", type=float, default=80.0)
    p.add_argument("--smooth-passes", type=int, default=80)
    p.add_argument("--min-depth", type=float, default=500.0)
    args = p.parse_args()

    gcfg = replace(GlobalGridConfig(), lat_max=args.lat_max,
                   resolution=args.resolution,
                   nx=int(round(360 / args.resolution)),
                   ny=int(round(2 * args.lat_max / args.resolution)))
    grid = make_global_grid(gcfg, args.bathy,
                            smooth_passes=args.smooth_passes,
                            min_depth=args.min_depth, remap="area")
    T_init, _ = get_initial_fields(grid)
    sst = np.asarray(T_init[:, :, 0], dtype=float)
    air = load_monthly_mean_air_temp(grid, year=args.year)
    nx, ny = grid.nx, grid.ny
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with Dataset(args.out, "w") as ds:
        ds.createDimension("time", None)
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "days since 0001-01-01 00:00:00"
        t.calendar = "julian"
        x = ds.createVariable("x", "f8", ("x",))
        x.units = "degrees_east"
        y = ds.createVariable("y", "f8", ("y",))
        y.units = "degrees_north"
        sens = ds.createVariable("sensible", "f4", ("time", "y", "x"))
        sens.units = "W m-2"
        sens.long_name = "lambda*(2m air - WOA SST) bulk-like sensible-heat proxy"
        x[:] = grid.lon
        y[:] = grid.lat
        for m in range(12):
            q = args.lambda_bulk * (air[m] - sst)
            sens[m, :, :] = np.ascontiguousarray(q.T, dtype=np.float32)
            t[m] = 1.0 + 30.0 * m + 15.0
        ds.title = "Ocean_solver-like bulk sensible heat proxy"
        ds.lambda_bulk_w_m2_k = args.lambda_bulk
    print(f"wrote {args.out}: 12 x {ny} x {nx}")


if __name__ == "__main__":
    main()

