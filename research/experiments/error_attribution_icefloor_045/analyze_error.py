#!/usr/bin/env python3
"""Re-baseline error attribution for the 0.45-degree ice-floor candidate."""

from pathlib import Path
import argparse
import json
import sys
from collections import deque

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from bench_climatology_global import smooth_2d_global


def distance_to_land(ocean):
    dist = np.full(ocean.shape, np.inf)
    dist[~ocean] = 0.0
    q = deque((i, j) for i in range(ocean.shape[0])
              for j in range(ocean.shape[1]) if not ocean[i, j])
    while q:
        i, j = q.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = (i + di) % ocean.shape[0], j + dj
            if 0 <= nj < ocean.shape[1] and ocean[ni, nj] and dist[i, j] + 1 < dist[ni, nj]:
                dist[ni, nj] = dist[i, j] + 1.0
                q.append((ni, nj))
    return dist


def score(run):
    sst, woa, ocean = run["sst"], run["woa"], run["ocean"]
    lat, lon, area = run["lat"], run["lon"], run["area"]
    sst_sm = smooth_2d_global(sst, lat, deg=2.0)
    woa_sm = smooth_2d_global(woa, lat, deg=2.0)
    raw = sst - woa
    a2 = sst_sm - woa_sm
    lon2, lat2 = np.meshgrid(lon, lat, indexing="ij")

    def metrics(mask):
        aa = area[mask]
        return {
            "cells": int(mask.sum()),
            "raw_bias_c": float(np.sum(raw[mask] * aa) / np.sum(aa)),
            "raw_rmse_c": float(np.sqrt(np.sum(raw[mask] ** 2 * aa) / np.sum(aa))),
            "a2_rmse_c": float(np.sqrt(np.mean(a2[mask] ** 2))),
            "a2_sse_share": float(np.sum(a2[mask] ** 2) / np.sum(a2[ocean] ** 2)),
        }

    bands = {
        "land_0_3": ocean & (run["land_dist"] <= 3.0),
        "land_4_7": ocean & (run["land_dist"] > 3.0) & (run["land_dist"] <= 7.0),
        "land_8_14": ocean & (run["land_dist"] > 7.0) & (run["land_dist"] <= 14.0),
        "interior_ge15": ocean & (run["land_dist"] > 14.0),
        "lat_40_50": ocean & (lat2 >= 40.0) & (lat2 < 50.0),
        "lat_50_60": ocean & (lat2 >= 50.0) & (lat2 < 60.0),
        "lat_60_65": ocean & (lat2 >= 60.0),
        "lat_m40_0": ocean & (lat2 >= -40.0) & (lat2 < 0.0),
        "lat_0_40": ocean & (lat2 >= 0.0) & (lat2 < 40.0),
        "south_of_40": ocean & (lat2 < -40.0),
    }
    regions = {
        "global_ocean": ocean,
        "north_atlantic_40_60": ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 40.0) & (lat2 <= 60.0),
        "near_wall_55_60": ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 55.0) & (lat2 <= 60.0),
    }
    out = {
        "global": metrics(ocean),
        "north_atlantic_40_60": metrics(regions["north_atlantic_40_60"]),
        "near_wall_55_60": metrics(regions["near_wall_55_60"]),
        "bands": {name: metrics(mask) for name, mask in bands.items() if mask.any()},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/ice_proxy_045/global_res045_icefloor_365d_repeat.npz")
    ap.add_argument("--out", default="research/experiments/error_attribution_icefloor_045/metrics.json")
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    days = np.asarray(z["days"])
    steady = days >= days[-1] - 90.0
    ocean = np.asarray(z["wet_mask"]) > 0.5
    lat, lon = np.asarray(z["lat"]), np.asarray(z["lon"])
    area = np.broadcast_to((np.asarray(z["T_init"][0, :, 0]) * 0 + 1), ocean.shape)
    # Build physical area independently below.
    import os
    from dataclasses import replace
    from config import GlobalGridConfig
    from grid import make_global_grid
    bathy = os.environ.get("OCEAN_SOLVER_BATHYMETRY", str(Path.home() / "Desktop" / "ETOPO_2022_v1_r3600x1800_surface.nc"))
    grid = make_global_grid(replace(GlobalGridConfig(), lat_max=65.0, ny=288, nx=800, resolution=0.45),
                            bathy, smooth_passes=80, min_depth=500.0, remap="area")
    area = np.asarray(grid.dx_2d) * float(grid.dy)
    run = {
        "sst": np.mean(np.asarray(z["T_top"])[steady], axis=0),
        "woa": np.asarray(z["T_init"])[:, :, 0],
        "ocean": ocean,
        "lat": lat,
        "lon": lon,
        "area": area,
        "land_dist": distance_to_land(ocean),
    }
    result = {"npz": str(args.npz), "score": score(run)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
