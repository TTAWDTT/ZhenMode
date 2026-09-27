#!/usr/bin/env python3
"""Build the Stage-G NCEP forcing stack on the shared solver grid.

This pre-registers only the forcing artifact; it does not run the model.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import DEFAULT_CONFIG, GlobalGridConfig
from air_reanalysis import load_monthly_mean_air_temp
from grid import global_grid_dims, make_global_grid
from wind_reanalysis import load_monthly_wind_speed
from humidity_reanalysis import load_monthly_mean_specific_humidity
from surface_reanalysis import (
    load_monthly_downward_longwave,
    load_monthly_downward_shortwave,
    load_monthly_precipitation,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolution", type=float, default=0.5)
    ap.add_argument("--lat-max", type=float, default=65.0)
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--smooth-passes", type=int, default=80)
    ap.add_argument("--min-depth", type=float, default=500.0)
    ap.add_argument("--remap", choices=("legacy", "area"), default="area")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "stage_g" / "stage_g_forcing.npz"))
    args = ap.parse_args()

    nx, ny = global_grid_dims(args.resolution, args.lat_max, remap=args.remap)
    if nx < 4 or ny < 4:
        raise SystemExit(f"grid too coarse: {nx}x{ny}")
    gcfg = replace(
        GlobalGridConfig(),
        nx=nx, ny=ny, resolution=args.resolution, lat_max=args.lat_max,
    )
    print(f"building {nx}x{ny} grid ...")
    grid = make_global_grid(
        gcfg, DEFAULT_CONFIG.bathymetry_file,
        smooth_passes=args.smooth_passes, min_depth=args.min_depth,
        remap=args.remap,
    )
    print("loading Stage-G NCEP forcing ...")
    humidity = load_monthly_mean_specific_humidity(
        grid, year=args.year, cache_dir=args.cache_dir)
    longwave = load_monthly_downward_longwave(
        grid, year=args.year, cache_dir=args.cache_dir)
    shortwave = load_monthly_downward_shortwave(
        grid, year=args.year, cache_dir=args.cache_dir)
    precipitation = load_monthly_precipitation(
        grid, year=args.year, cache_dir=args.cache_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        lat=np.asarray(grid.lat),
        lon=np.asarray(grid.lon),
        wet_mask=np.asarray(grid.wet_mask),
        specific_humidity_kg_kg=np.asarray(humidity),
        downward_longwave_w_m2=np.asarray(longwave),
        downward_shortwave_w_m2=np.asarray(shortwave),
        precipitation_rate_kg_m2_s=np.asarray(precipitation),

        air_temperature_c=np.asarray(

            load_monthly_mean_air_temp(grid, year=args.year)),

        wind_speed_m_s=np.asarray(

            load_monthly_wind_speed(grid, year=args.year)),

        year=np.array(args.year),
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
