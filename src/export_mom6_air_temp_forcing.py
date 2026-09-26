"""Export ocean_solver monthly NCEP 2m air temperature to an A-grid MOM6 file."""
import os
from argparse import ArgumentParser
from dataclasses import replace

import numpy as np
from netCDF4 import Dataset

from air_reanalysis import load_monthly_mean_air_temp
from config import GlobalGridConfig
from grid import make_global_grid


def main():
    p = ArgumentParser()
    p.add_argument("--out", default="INPUT/ncep_2023_air_050.nc")
    p.add_argument("--bathy", default=r"C:\Users\zhen.luo\ocean_solver\data\ETOPO_2022_v1_r3600x1800_surface.nc.npz")
    p.add_argument("--resolution", type=float, default=0.5)
    p.add_argument("--lat-max", type=float, default=65.0)
    p.add_argument("--year", type=int, default=2023)
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
    nx, ny = grid.nx, grid.ny
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    air = load_monthly_mean_air_temp(grid, year=args.year)
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
        air_var = ds.createVariable("AIR_TEMP", "f4", ("time", "y", "x"))
        air_var.units = "degC"
        air_var.long_name = "NCEP R1 2m air temperature on A-grid"
        x[:] = grid.lon
        y[:] = grid.lat
        for m in range(12):
            air_var[m, :, :] = np.ascontiguousarray(air[m].T, dtype=np.float32)
            t[m] = 1.0 + 30.0 * m + 15.0
        ds.title = "NCEP R1 monthly 2m air temperature"
    print(f"wrote {args.out}: 12 x {ny} x {nx}")


if __name__ == "__main__":
    main()
