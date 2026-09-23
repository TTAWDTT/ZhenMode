#!/usr/bin/env python3
"""Surface heat-budget bands for the locked 0.45-degree candidate."""

from pathlib import Path
import argparse
import json
from collections import deque
from dataclasses import replace

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys_path = str(REPO / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from air_reanalysis import load_annual_mean_air_temp
from config import GlobalGridConfig, RHO_0, C_P
from grid import make_global_grid


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


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights)
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def weighted_rms(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights)
    return float(np.sqrt(np.sum(values[valid] ** 2 * weights[valid]) / np.sum(weights[valid])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/diag_res045_terms_30d/global_res045_candidate_terms_30d.npz")
    ap.add_argument("--lambda-bulk", type=float, default=80.0)
    ap.add_argument("--surface-depth", type=float, default=5.0)
    ap.add_argument("--out", default="research/experiments/res045_surface_budget_30d")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    stem = Path(args.npz).with_suffix("")
    days = np.asarray(z["days"])
    idx = np.where(days >= days[-1] - 10.0 + 1e-9)[0]
    states = [np.load(stem.parent / (stem.name + "_3d") / f"snap_{i:05d}.npy") for i in idx]
    frames = [np.load(stem.parent / (stem.name + "_terms") / f"terms_{i:05d}.npy") for i in idx]
    terms = np.mean(np.stack(frames), axis=0) * 86400.0
    sst = np.mean(np.stack([s[0, :, :, 0] for s in states]), axis=0)

    bathy = __import__("os").environ.get(
        "OCEAN_SOLVER_BATHYMETRY", str(Path.home() / "Desktop" / "ETOPO_2022_v1_r3600x1800_surface.nc"))
    grid = make_global_grid(
        replace(GlobalGridConfig(), lat_max=65.0, ny=288, nx=800, resolution=0.45),
        bathy, smooth_passes=80, min_depth=500.0, remap="area",
    )
    T_atm = np.asarray(load_annual_mean_air_temp(grid, 2023), dtype=np.float64)
    area = np.asarray(grid.dx_2d) * float(grid.dy)
    ocean = np.asarray(z["wet_mask"]) > 0.5
    lat, lon = np.asarray(z["lat"]), np.asarray(z["lon"])
    lon2, lat2 = np.meshgrid(lon, lat, indexing="ij")
    woa = np.asarray(z["T_init"])[:, :, 0]

    surface = {
        "advection": terms[0, :, :, 0],
        "horizontal_diffusion": terms[1, :, :, 0],
        "vertical_diffusion": terms[2, :, :, 0],
        "convection": terms[3, :, :, 0],
        "gm": terms[4, :, :, 0],
        "redi": terms[5, :, :, 0],
        "bulk": args.lambda_bulk * (T_atm - sst) / (RHO_0 * C_P * args.surface_depth),
    }

    land_dist = distance_to_land(ocean)
    bands = {
        "land_0_3": ocean & (land_dist <= 3.0),
        "land_4_7": ocean & (land_dist > 3.0) & (land_dist <= 7.0),
        "land_8_14": ocean & (land_dist > 7.0) & (land_dist <= 14.0),
        "interior_ge15": ocean & (land_dist > 14.0),
    }
    regions = {
        "global_ocean": ocean,
        "north_atlantic_40_60": ocean & (lon2 >= 300) & (lon2 < 360) & (lat2 >= 40) & (lat2 <= 60),
        "near_wall_55_60": ocean & (lon2 >= 300) & (lon2 < 360) & (lat2 >= 55) & (lat2 <= 60),
    }
    result = {"npz": str(args.npz), "snapshots": [int(i) for i in idx], "groups": {}}
    for region_name, region_mask in regions.items():
        result["groups"][region_name] = {}
        for band_name, band_mask in bands.items():
            mask = region_mask & band_mask
            if not np.any(mask):
                continue
            aa = area[mask]
            row = {"cells": int(mask.sum())}
            row["sst_bias_c"] = weighted_mean(sst[mask] - woa[mask], aa)
            row["sst_rmse_c"] = weighted_rms(sst[mask] - woa[mask], aa)
            for term, field in surface.items():
                row[f"{term}_k_per_day"] = weighted_mean(field[mask], aa)
            row["net_core_k_per_day"] = weighted_mean(
                sum(surface[k][mask] for k in ("advection", "horizontal_diffusion",
                                               "vertical_diffusion", "convection", "bulk")), aa)
            result["groups"][region_name][band_name] = row

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "surface_budget.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
