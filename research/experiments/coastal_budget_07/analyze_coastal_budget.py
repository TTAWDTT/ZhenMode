#!/usr/bin/env python3
"""Compare surface heat tendencies in the baseline and hard coastal-restore runs."""

from pathlib import Path
import argparse, json, sys
from collections import deque
from dataclasses import replace

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from air_reanalysis import load_annual_mean_air_temp
from config import GlobalGridConfig, RHO_0, C_P
from grid import make_global_grid


def find_bathymetry():
    import os
    env = os.environ.get("OCEAN_SOLVER_BATHYMETRY")
    if env:
        return env
    candidate = Path.home() / "Desktop" / "ETOPO_2022_v1_r3600x1800_surface.nc"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("Set OCEAN_SOLVER_BATHYMETRY.")


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


def load_budget_run(path, last_days=10):
    z = np.load(path, allow_pickle=True)
    days = np.asarray(z["days"])
    idx = np.where(days >= days[-1] - last_days + 1e-9)[0]
    base = Path(path).with_suffix("")
    states = [np.load(base.parent / (base.name + "_3d") / f"snap_{i:05d}.npy")
              for i in idx]
    terms = [np.load(base.parent / (base.name + "_terms") / f"terms_{i:05d}.npy")
             for i in idx]
    # terms_fn returns K/s; convert to K/day for readability.
    terms = np.mean(np.stack(terms), axis=0) * 86400.0
    sst = np.mean(np.stack([s[0, :, :, 0] for s in states]), axis=0)
    return z, terms, sst


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/coastal_budget_07/global_coastal_budget_baseline_30d.npz")
    ap.add_argument("--restore", default="results/coastal_budget_07/global_coastal_budget_restore_30d.npz")
    ap.add_argument("--restore-days", type=float, default=0.5)
    ap.add_argument("--out", default="research/experiments/coastal_budget_07")
    args = ap.parse_args()

    grid = make_global_grid(
        replace(GlobalGridConfig(), lat_max=65.0, ny=185, nx=514, resolution=0.7),
        find_bathymetry(), smooth_passes=80, min_depth=500.0,
    )
    area2d = np.asarray(grid.dx_2d) * float(grid.dy)
    T_atm = np.asarray(load_annual_mean_air_temp(grid, 2023), dtype=np.float64)
    heat_factor = 1.0 / (RHO_0 * C_P * 5.0)

    base_z, base_terms, base_sst = load_run = None, None, None
    for label, path in (("baseline", args.baseline), ("restore", args.restore)):
        z = np.load(path, allow_pickle=True)
        stem = Path(path).with_suffix("")
        days = np.asarray(z["days"])
        idx = np.where(days >= days[-1] - 10.0 + 1e-9)[0]
        states = [np.load(stem.parent / (stem.name + "_3d") / f"snap_{i:05d}.npy") for i in idx]
        frames = [np.load(stem.parent / (stem.name + "_terms") / f"terms_{i:05d}.npy") for i in idx]
        terms = np.mean(np.stack(frames), axis=0) * 86400.0
        sst = np.mean(np.stack([s[0, :, :, 0] for s in states]), axis=0)
        if label == "baseline":
            base_z, base_terms, base_sst = z, terms, sst
        else:
            rest_z, rest_terms, rest_sst = z, terms, sst

    ocean = np.asarray(base_z["wet_mask"]).astype(bool)
    lat = np.asarray(base_z["lat"])
    lon = np.asarray(base_z["lon"])
    lon2, lat2 = np.meshgrid(lon, lat, indexing="ij")
    woa = np.asarray(base_z["T_init"])[:, :, 0]
    sst_bias = base_sst - woa
    restoring_rate = 1.0 / (args.restore_days * 86400.0) * 86400.0  # K/day

    # Terms from terms_fn: adv, horizontal diffusion, vertical diffusion,
    # convection, GM, Redi. Surface terms are the top layer.
    base_surface = {
        "advection": base_terms[0, :, :, 0],
        "horizontal_diffusion": base_terms[1, :, :, 0],
        "vertical_diffusion": base_terms[2, :, :, 0],
        "convection": base_terms[3, :, :, 0],
    }
    rest_surface = {
        "advection": rest_terms[0, :, :, 0],
        "horizontal_diffusion": rest_terms[1, :, :, 0],
        "vertical_diffusion": rest_terms[2, :, :, 0],
        "convection": rest_terms[3, :, :, 0],
    }
    base_surface["bulk"] = 160.0 * (T_atm - base_sst) * heat_factor
    rest_surface["bulk"] = 160.0 * (T_atm - rest_sst) * heat_factor
    coastal_restore_target = (woa - rest_sst) * restoring_rate
    land_dist = np.full(ocean.shape, np.inf)
    dist = np.full(ocean.shape, np.inf)
    dist[~ocean] = 0.0
    q = deque((i, j) for i in range(ocean.shape[0])
              for j in range(ocean.shape[1]) if not ocean[i, j])
    while q:
        i, j = q.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = (i + di) % ocean.shape[0], j + dj
            if 0 <= nj < ocean.shape[1] and ocean[ni, nj] and dist[i, j] + 1.0 < dist[ni, nj]:
                dist[ni, nj] = dist[i, j] + 1.0
                q.append((ni, nj))
    land_dist = dist
    rest_surface["sst_restore"] = coastal_restore_target * ((land_dist <= 7.0) & ocean)
    base_surface["sst_restore"] = np.zeros_like(coastal_restore_target)

    land_dist = np.full(ocean.shape, np.inf)
    dist = np.full(ocean.shape, np.inf)
    dist[~ocean] = 0.0
    q = deque((i, j) for i in range(ocean.shape[0])
              for j in range(ocean.shape[1]) if not ocean[i, j])
    while q:
        i, j = q.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = (i + di) % ocean.shape[0], j + dj
            if 0 <= nj < ocean.shape[1] and ocean[ni, nj] and dist[i, j] + 1.0 < dist[ni, nj]:
                dist[ni, nj] = dist[i, j] + 1.0
                q.append((ni, nj))
    land_dist = dist
    groups = {
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

    result = {"restore_days": args.restore_days, "groups": {}}
    rows = []
    for group_name, group_mask in groups.items():
        for region_name, region_mask in regions.items():
            mask = group_mask & region_mask
            if not np.any(mask):
                continue
            aa = area2d[mask]
            row = {"group": group_name, "region": region_name, "cells": int(mask.sum())}
            for label, sst in (("baseline", base_sst), ("restore", rest_sst)):
                row[f"{label}_sst_bias_c"] = weighted_mean(sst[mask] - woa[mask], aa)
                row[f"{label}_sst_rmse_c"] = weighted_rms(sst[mask] - woa[mask], aa)
                terms = base_surface if label == "baseline" else rest_surface
                for term_name, field in terms.items():
                    row[f"{label}_{term_name}_k_per_day"] = weighted_mean(field[mask], aa)
                row[f"{label}_net_k_per_day"] = weighted_mean(sum(terms[k][mask] for k in terms), aa)
            for term_name in base_surface:
                row[f"delta_{term_name}_k_per_day"] = row[f"restore_{term_name}_k_per_day"] - row[f"baseline_{term_name}_k_per_day"]
            rows.append(row)
            result["groups"].setdefault(region_name, {})[group_name] = row

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "surface_budget.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    fields = list(rows[0].keys())
    with (out / "surface_budget.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(fields) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fields) + "\n")
    print(f"Wrote {out / 'surface_budget.json'}")
    print(f"Wrote {out / 'surface_budget.csv'}")


if __name__ == "__main__":
    main()