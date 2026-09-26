"""Export ocean_solver seasonal wind stress to an A-grid MOM6 wind file."""
import os
from argparse import ArgumentParser
from dataclasses import replace

import numpy as np
from netCDF4 import Dataset

from config import GlobalGridConfig
from grid import make_global_grid
from wind_reanalysis import real_wind_forcing


def main():
    p = ArgumentParser()
    p.add_argument("--out", default="INPUT/ncep_2023_tau_050.nc")
    p.add_argument("--bathy", default=r"C:\Users\zhen.luo\ocean_solver\data\ETOPO_2022_v1_r3600x1800_surface.nc.npz")
    p.add_argument("--resolution", type=float, default=0.5)
    p.add_argument("--lat-max", type=float, default=65.0)
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--smooth-passes", type=int, default=80)
    p.add_argument("--min-depth", type=float, default=500.0)
    p.add_argument("--taper-cells", type=int, default=8)
    args = p.parse_args()

    res = args.resolution
    gcfg = replace(GlobalGridConfig(), lat_max=args.lat_max,
                   resolution=res, nx=int(round(360 / res)),
                   ny=int(round(2 * args.lat_max / res)))
    grid = make_global_grid(gcfg, args.bathy,
                            smooth_passes=args.smooth_passes,
                            min_depth=args.min_depth, remap="area")
    nx, ny = grid.nx, grid.ny
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    month0 = (args.year - 1948) * 12
    with Dataset(args.out, "w") as ds:
        ds.createDimension("time", None)
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "days since 0001-01-01 00:00:00"; t.calendar = "julian"
        x = ds.createVariable("x", "f8", ("x",)); x.units = "degrees_east"
        y = ds.createVariable("y", "f8", ("y",)); y.units = "degrees_north"
        vx = ds.createVariable("STRESS_X", "f4", ("time", "y", "x"))
        vy = ds.createVariable("STRESS_Y", "f4", ("time", "y", "x"))
        vx.units = "N m-2"; vy.units = "N m-2"
        vx.long_name = "zonal wind stress on A-grid"; vy.long_name = "meridional wind stress on A-grid"
        x[:] = grid.lon; y[:] = grid.lat
        for m in range(12):
            taux, tauy = real_wind_forcing(grid, month_idx=month0 + m,
                                           taper_cells=args.taper_cells)
            vx[m, :, :] = np.ascontiguousarray(taux.T, dtype=np.float32)
            vy[m, :, :] = np.ascontiguousarray(tauy.T, dtype=np.float32)
            t[m] = 1.0 + 30.0 * m + 15.0
        ds.title = "NCEP R1 10m wind converted to bulk wind stress"
        ds.ocean_solver_month_mapping = f"{month0}..{month0 + 11}"
        ds.taper_cells = args.taper_cells
    print(f"wrote {args.out}: 12 x {ny} x {nx}")


if __name__ == "__main__":
    main()

