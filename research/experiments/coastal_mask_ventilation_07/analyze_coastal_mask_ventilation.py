#!/usr/bin/env python3
"""Diagnose how much of the 0.7-degree cold bias is coastal/shallow.

Uses the matched GM0/GM500 final-90d states. This is analysis-only.
"""

from pathlib import Path
import argparse
import json
import os
import sys
from collections import deque

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from ocean_solver.config.definitions import GlobalGridConfig
from dataclasses import replace
from ocean_solver.io.grid import make_global_grid


def find_bathymetry():
    env = os.environ.get("OCEAN_SOLVER_BATHYMETRY")
    if env:
        return env
    candidate = Path.home() / "Desktop" / "ETOPO_2022_v1_r3600x1800_surface.nc"
    return str(candidate)


def distance_to_mask(target_mask, periodic_x=True):
    target = target_mask.astype(bool)
    nx, ny = target.shape
    dist = np.full((nx, ny), np.inf, dtype=np.float64)
    dist[target] = 0.0
    q = deque((i, j) for i in range(nx) for j in range(ny) if target[i, j])
    while q:
        i, j = q.popleft()
        d = dist[i, j]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = (i + di) % nx if periodic_x else i + di
                nj = j + dj
                if 0 <= ni < nx and 0 <= nj < ny and d + 1.0 < dist[ni, nj]:
                    dist[ni, nj] = d + 1.0
                    q.append((ni, nj))
    return dist


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def weighted_rms(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.sum(values[valid] ** 2 * weights[valid]) /
                         np.sum(weights[valid])))


def group_stats(mask, error, area, region_mask, speed, land_dist, depth, deep_dist):
    """Area-weighted error and geometry stats for one 2D group inside a region."""
    mask = mask & region_mask
    if not np.any(mask):
        return None
    area_sel = area[mask]
    total_area = np.sum(area[region_mask])
    sse = np.sum(error[mask] ** 2 * area_sel)
    total_sse = np.sum(error[region_mask] ** 2 * area[region_mask])
    return {
        "cells": int(mask.sum()),
        "area_share": float(np.sum(area_sel) / total_area),
        "sst_bias_c": weighted_mean(error[mask], area_sel),
        "sst_rmse_c": weighted_rms(error[mask], area_sel),
        "sse_share": float(sse / total_sse),
        "cold_lt_minus1_share": float(np.sum(error[mask] <= -1.0) / mask.sum()),
        "mean_land_distance_cells": weighted_mean(land_dist[mask], area_sel),
        "mean_depth_m": weighted_mean(depth[mask], area_sel),
        "mean_speed_m_s": weighted_mean(speed[mask], area_sel),
        "mean_distance_to_deep_cells": weighted_mean(deep_dist[mask], area_sel),
        "counterfactual_rmse_if_group_zero_c": float(
            np.sqrt(max(total_sse - sse, 0.0) / np.sum(area[region_mask]))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gm0", default="results/gm0_attribution_07/global_gm0_3dterms.npz")
    parser.add_argument("--gm500", default="results/gm0_attribution_07/global_gm500_3dterms.npz")
    parser.add_argument("--out", default="research/experiments/coastal_mask_ventilation_07")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    grid = make_global_grid(
        replace(GlobalGridConfig(), lat_max=65.0, ny=185, nx=514, resolution=0.7),
        find_bathymetry(), smooth_passes=30, min_depth=100.0,
    )
    area = np.asarray(grid.dx_2d) * float(grid.dy)
    depth = np.asarray(grid.depth)
    per_run = {}
    rows = []
    for label, path in [("gm0", args.gm0), ("gm500", args.gm500)]:
        z = np.load(path, allow_pickle=True)
        days = np.asarray(z["days"])
        idx = np.where(days >= days[-1] - 90.0 + 1e-9)[0]
        base = Path(path).with_suffix("")
        states = [np.load(base.parent / (base.name + "_3d") / f"snap_{i:05d}.npy",
                          allow_pickle=True) for i in idx]
        sst = np.mean(np.stack([s[0, :, :, 0] for s in states]), axis=0)
        u = np.mean(np.stack([s[1, :, :, 0] for s in states]), axis=0)
        v = np.mean(np.stack([s[2, :, :, 0] for s in states]), axis=0)
        speed = np.hypot(u, v)
        ocean = np.asarray(z["wet_mask"]) > 0.5
        lat = np.asarray(z["lat"])
        lon = np.asarray(z["lon"])
        lon2, lat2 = np.meshgrid(lon, lat, indexing="ij")
        woa = np.asarray(z["T_init"])[:, :, 0]
        error = sst - woa
        land_dist = distance_to_mask(~ocean)
        deep_dist = distance_to_mask(ocean & (depth >= 1000.0))
        regions = {
            "global_ocean": ocean,
            "north_atlantic_40_60": ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 40.0) & (lat2 <= 60.0),
            "near_wall_55_60": ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 55.0) & (lat2 <= 60.0),
        }
        # Exclusive land-distance groups.
        land_groups = {
            "land_dist_0_3": ocean & (land_dist <= 3.0),
            "land_dist_4_7": ocean & (land_dist > 3.0) & (land_dist <= 7.0),
            "land_dist_ge8": ocean & (land_dist > 7.0),
        }
        # Exclusive depth groups.
        depth_groups = {
            "depth_100_200": ocean & (depth >= 100.0) & (depth < 200.0),
            "depth_200_500": ocean & (depth >= 200.0) & (depth < 500.0),
            "depth_500_1000": ocean & (depth >= 500.0) & (depth < 1000.0),
            "depth_ge1000": ocean & (depth >= 1000.0),
        }
        # Cross groups: coastal geometry vs interior geometry.
        cross_groups = {
            "coastal_shallow": ocean & (land_dist <= 3.0) & (depth < 200.0),
            "coastal_deep": ocean & (land_dist <= 3.0) & (depth >= 200.0),
            "interior_shallow": ocean & (land_dist > 3.0) & (depth < 200.0),
            "interior_deep": ocean & (land_dist > 3.0) & (depth >= 200.0),
        }
        all_groups = {**{f"land_{k}": v for k, v in land_groups.items()},
                      **{f"depth_{k}": v for k, v in depth_groups.items()},
                      **cross_groups}
        per_run[label] = {}
        for region_name, region_mask in regions.items():
            per_run[label][region_name] = {}
            for group_name, group_mask in all_groups.items():
                stats = group_stats(group_mask, error, area, region_mask,
                                    speed, land_dist, depth, deep_dist)
                if stats and stats["cells"]:
                    per_run[label][region_name][group_name] = stats
                    rows.append({"run": label, "region": region_name,
                                 "group": group_name, **stats})

    fields = ["run", "region", "group", "cells", "area_share", "sst_bias_c",
              "sst_rmse_c", "sse_share", "cold_lt_minus1_share",
              "mean_land_distance_cells", "mean_depth_m", "mean_speed_m_s",
              "mean_distance_to_deep_cells", "counterfactual_rmse_if_group_zero_c"]
    with (out / "group_summary.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(fields) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in fields) + "\n")
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(per_run, f, indent=2)
    print(f"Wrote {out / 'group_summary.csv'}")
    print(f"Wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
