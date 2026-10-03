#!/usr/bin/env python3
"""Separate coastal geometry from weak high-latitude ventilation.

Uses the same matched GM0/GM500 0.7-degree final-90d states as the near-wall
gradient and vertical-attribution diagnostics. This is analysis-only; it does
not rerun the model.
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

from ocean_solver.forcing.air import load_annual_mean_air_temp
from ocean_solver.config.definitions import DEFAULT_CONFIG, GlobalGridConfig, RHO_0, C_P
from dataclasses import replace
from ocean_solver.io.grid import make_global_grid


def find_bathymetry():
    env = os.environ.get("OCEAN_SOLVER_BATHYMETRY")
    if env:
        return env
    candidate = Path.home() / "Desktop" / \
        "ETOPO_2022_v1_r3600x1800_surface.nc"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("Set OCEAN_SOLVER_BATHYMETRY to the ETOPO2022 file.")


def nearest_land_distance(ocean):
    land = ~ocean
    nx, ny = land.shape
    dist = np.full((nx, ny), np.inf, dtype=np.float64)
    dist[land] = 0.0
    q = deque((i, j) for i in range(nx) for j in range(ny) if land[i, j])
    while q:
        i, j = q.popleft()
        d = dist[i, j]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = (i + di) % nx
                nj = j + dj
                if 0 <= nj < ny and not land[ni, nj] and d + 1.0 < dist[ni, nj]:
                    dist[ni, nj] = d + 1.0
                    q.append((ni, nj))
    return dist


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(values[valid] * weights[valid]) /
                 np.sum(weights[valid]))


def weighted_rms(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.sum(values[valid] ** 2 * weights[valid]) /
                         np.sum(weights[valid])))


def load_run(path):
    z = np.load(path, allow_pickle=True)
    days = np.asarray(z["days"])
    idx = np.where(days >= days[-1] - 90.0 + 1e-9)[0]
    base = Path(path).with_suffix("")
    states = [np.load(base.parent / (base.name + "_3d") /
                      f"snap_{i:05d}.npy", allow_pickle=True) for i in idx]
    term_frames = [np.load(base.parent / (base.name + "_terms") /
                           f"terms_{i:05d}.npy", allow_pickle=True) for i in idx]
    # K/s -> K/day
    terms = np.mean(np.stack(term_frames), axis=0) * 86400.0
    sst = np.mean(np.stack([s[0, :, :, 0] for s in states]), axis=0)
    u = np.mean(np.stack([s[1, :, :, 0] for s in states]), axis=0)
    v = np.mean(np.stack([s[2, :, :, 0] for s in states]), axis=0)
    speed = np.hypot(u, v)
    return z, terms, sst, speed


def summarize_group(mask, error, speed, land_dist, terms, bulk, depth_node, area2d):
    """Return surface and subsurface statistics for a 2D cell group."""
    if not np.any(mask):
        return {"cells": 0}
    area = area2d[mask]
    surface_adv = terms[0, :, :, 0][mask]
    surface_conv = terms[3, :, :, 0][mask]
    surface_gm = terms[4, :, :, 0][mask]
    surface_bulk = bulk[mask]
    upper_mask = (depth_node < -50.0) & (depth_node >= -200.0)
    upper_adv = terms[0, mask, :][:, upper_mask]
    upper_conv = terms[3, mask, :][:, upper_mask]
    upper_area = np.broadcast_to(area[:, None], upper_adv_shape := (len(area), upper_mask.sum()))
    return {
        "cells": int(mask.sum()),
        "area_m2": float(np.sum(area)),
        "sst_bias_c": weighted_mean(error[mask], area),
        "sst_rmse_c": weighted_rms(error[mask], area),
        "cold_lt_minus1_share": float(np.sum(error[mask] <= -1.0) / mask.sum()),
        "mean_land_distance_cells": weighted_mean(land_dist[mask], area),
        "mean_speed_m_s": weighted_mean(speed[mask], area),
        "surface_advection_k_per_day": weighted_mean(surface_adv, area),
        "surface_convection_k_per_day": weighted_mean(surface_conv, area),
        "surface_bulk_flux_k_per_day": weighted_mean(surface_bulk, area),
        "surface_gm_k_per_day": weighted_mean(surface_gm, area),
        "surface_net_k_per_day": weighted_mean(
            surface_adv + surface_conv + surface_gm + surface_bulk, area),
        "upper_50_200m_advection_k_per_day": weighted_mean(
            terms[0, mask, :][:, upper_mask],
            np.broadcast_to(area[:, None], (len(area), upper_mask.sum()))),
        "upper_50_200m_convection_k_per_day": weighted_mean(
            terms[3, mask, :][:, upper_mask],
            np.broadcast_to(area[:, None], (len(area), upper_mask.sum()))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gm0", default="results/gm0_attribution_07/global_gm0_3dterms.npz")
    parser.add_argument("--gm500", default="results/gm0_attribution_07/global_gm500_3dterms.npz")
    parser.add_argument("--out", default="research/experiments/coastal_vs_ventilation_07")
    parser.add_argument("--n-worst", type=int, default=100)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OCEAN_SOLVER_BATHYMETRY", find_bathymetry())
    grid = make_global_grid(
        replace(GlobalGridConfig(), lat_max=65.0, ny=185, nx=514, resolution=0.7),
        find_bathymetry(), smooth_passes=30, min_depth=100.0,
    )
    T_atm = np.asarray(load_annual_mean_air_temp(grid, 2023), dtype=np.float64)
    heat_factor = 1.0 / (RHO_0 * 3992.0 * 5.0)

    per_run = {}
    group_rows = []
    cell_rows = []
    for label, path in [("gm0", args.gm0), ("gm500", args.gm500)]:
        z, terms, sst, speed = load_run(path)
        ocean = np.asarray(z["wet_mask"]) > 0.5
        lat = np.asarray(z["lat"])
        lon = np.asarray(z["lon"])
        lon2, lat2 = np.meshgrid(lon, lat, indexing="ij")
        woa = np.asarray(z["T_init"])[:, :, 0]
        error = sst - woa
        area = np.asarray(grid.dx_2d) * float(grid.dy)
        land_dist = nearest_land_distance(ocean)
        bulk = 80.0 * (T_atm - sst) * heat_factor * ocean * 86400.0
        near_wall = ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 55.0) & (lat2 <= 60.0)

        # Groups for the preregistered comparison.
        groups = {
            "near_wall_all": near_wall,
            "coastal_dist_le3": near_wall & (land_dist <= 3.0),
            "interior_dist_gt3": near_wall & (land_dist > 3.0),
            "highlat_interior_59_60": near_wall & (land_dist > 3.0) & (lat2 >= 59.0),
        }

        # Rank the interior near-wall cells by SST error and take the coldest N.
        interior_cells = np.where(near_wall & (land_dist > 3.0))
        order = np.argsort(error[interior_cells])
        worst_interior = [(int(i), int(j), int(rank), float(error[i, j]))
                          for rank, flat in enumerate(order[:args.n_worst], start=1)
                          for i, j in [(int(interior_cells[0][flat]), int(interior_cells[1][flat]))]]
        if worst_interior:
            mask = np.zeros_like(near_wall)
            for i, j, _, _ in worst_interior:
                mask[i, j] = True
            groups["interior_coldest_100"] = mask

        for name, mask in groups.items():
            row = {"run": label, "group": name,
                   **summarize_group(mask, error, speed, land_dist, terms, bulk,
                                     np.asarray(z["z"]), area)}
            group_rows.append(row)
            per_run.setdefault(label, {})[name] = row

        for i, j, rank, err in worst_interior:
            cell_rows.append({
                "run": label, "rank": rank,
                "lon_e": float(lon[i]), "lat_n": float(lat[j]),
                "raw_error_c": float(err), "sst_c": float(sst[i, j]),
                "woa_c": float(woa[i, j]), "mean_speed_m_s": float(speed[i, j]),
                "land_distance_cells": float(land_dist[i, j]),
                "surface_advection_k_per_day": float(terms[0, i, j, 0]),
                "surface_convection_k_per_day": float(terms[3, i, j, 0]),
                "surface_bulk_k_per_day": float(bulk[i, j]),
                "surface_total_k_per_day": float(
                    terms[0, i, j, 0] + terms[3, i, j, 0] +
                    terms[4, i, j, 0] + bulk[i, j]),
                "upper_50_200m_convection_k_per_day": float(np.mean(
                    terms[3, i, j, (np.asarray(z["z"]) < -50.0) & (np.asarray(z["z"]) >= -200.0)])),
            })

    fields = ["run", "group", "cells", "area_m2", "sst_bias_c", "sst_rmse_c",
              "cold_lt_minus1_share", "mean_land_distance_cells", "mean_speed_m_s",
              "surface_advection_k_per_day", "surface_convection_k_per_day",
              "surface_bulk_flux_k_per_day", "surface_gm_k_per_day",
              "surface_net_k_per_day", "upper_50_200m_advection_k_per_day",
              "upper_50_200m_convection_k_per_day"]
    with (out / "group_summary.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(fields) + "\n")
        for row in group_rows:
            f.write(",".join(str(row.get(k, "")) for k in fields) + "\n")

    cell_fields = ["run", "rank", "lon_e", "lat_n", "raw_error_c", "sst_c", "woa_c",
                   "mean_speed_m_s", "land_distance_cells",
                   "surface_advection_k_per_day", "surface_convection_k_per_day",
                   "surface_bulk_k_per_day", "surface_total_k_per_day",
                   "upper_50_200m_convection_k_per_day"]
    with (out / "interior_coldest_cells.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(cell_fields) + "\n")
        for row in cell_rows:
            f.write(",".join(str(row.get(k, "")) for k in cell_fields) + "\n")

    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(per_run, f, indent=2)

    print(f"Wrote {out / 'group_summary.csv'}")
    print(f"Wrote {out / 'interior_coldest_cells.csv'}")
    print(f"Wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
