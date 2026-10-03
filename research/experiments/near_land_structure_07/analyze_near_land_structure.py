#!/usr/bin/env python3
"""Compare the lambda80 and lambda160 candidates in the near-land band."""

from pathlib import Path
import argparse, json, math, os, sys
from collections import deque
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from ocean_solver.forcing.air import load_annual_mean_air_temp
from ocean_solver.validation.benchmarks.climatology import smooth_2d_global
from ocean_solver.config.definitions import DEFAULT_CONFIG, GlobalGridConfig, RHO_0, C_P
from dataclasses import replace
from ocean_solver.io.grid import make_global_grid


def find_bathymetry():
    env = os.environ.get("OCEAN_SOLVER_BATHYMETRY")
    if env:
        return env
    candidate = Path.home() / "Desktop" / "ETOPO_2022_v1_r3600x1800_surface.nc"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("Set OCEAN_SOLVER_BATHYMETRY to the ETOPO2022 file.")


def distance_to_mask(target_mask):
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
                ni = (i + di) % nx
                nj = j + dj
                if 0 <= nj < ny and d + 1.0 < dist[ni, nj]:
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


def weighted_corr(a, b, weights):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ma = weighted_mean(a, w)
    mb = weighted_mean(b, w)
    cov = weighted_mean((a - ma) * (b - mb), w)
    va = weighted_mean((a - ma) ** 2, w)
    vb = weighted_mean((b - mb) ** 2, w)
    return float(cov / np.sqrt(max(va * vb, 1e-30)))


def load_run(path):
    z = np.load(path, allow_pickle=True)
    days = np.asarray(z["days"])
    idx = np.where(days >= days[-1] - 90.0 + 1e-9)[0]
    base = Path(path).with_suffix("")
    states = [np.load(base.parent / (base.name + "_3d") / f"snap_{i:05d}.npy",
                      allow_pickle=True) for i in idx]
    term_frames = [np.load(base.parent / (base.name + "_terms") /
                           f"terms_{i:05d}.npy", allow_pickle=True) for i in idx]
    # K/s -> K/day
    terms = np.mean(np.stack(term_frames), axis=0) * 86400.0
    sst = np.mean(np.stack([s[0, :, :, 0] for s in states]), axis=0)
    u = np.mean(np.stack([s[1, :, :, 0] for s in states]), axis=0)
    v = np.mean(np.stack([s[2, :, :, 0] for s in states]), axis=0)
    speed = np.hypot(u, v)
    return z, terms, sst, u, v, speed


def gradient_ocean(field, ocean, dx2d, dy):
    """Ocean-aware horizontal gradient, K/m."""
    nx, ny = field.shape
    tx = np.zeros_like(field, dtype=np.float64)
    ty = np.zeros_like(field, dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            if not ocean[i, j]:
                continue
            right = ocean[(i + 1) % nx, j]
            left = ocean[(i - 1) % nx, j]
            up = j + 1 < ny and ocean[i, j + 1]
            down = j - 1 >= 0 and ocean[i, j - 1]
            if right and left:
                tx[i, j] = (field[(i + 1) % nx, j] - field[(i - 1) % nx, j]) / (
                    2.0 * dx2d[i, j])
            elif right:
                tx[i, j] = (field[(i + 1) % nx, j] - field[i, j]) / dx2d[i, j]
            elif left:
                tx[i, j] = (field[i, j] - field[(i - 1) % nx, j]) / dx2d[i, j]
            if up and down:
                ty[i, j] = (field[i, j + 1] - field[i, j - 1]) / (2.0 * dy)
            elif up:
                ty[i, j] = (field[i, j + 1] - field[i, j]) / dy
            elif down:
                ty[i, j] = (field[i, j] - field[i, j - 1]) / dy
    return tx, ty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda80", default="results/near_land_3dterms_07/global_lambda80_min500_3dterms.npz")
    parser.add_argument("--lambda160", default="results/near_land_3dterms_07/global_lambda160_min500_3dterms.npz")
    parser.add_argument("--out", default="research/experiments/near_land_structure_07")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Rebuild the grid once to get exact area and depth weights.
    grid = make_global_grid(
        replace(GlobalGridConfig(), lat_max=65.0, ny=185, nx=514, resolution=0.7),
        find_bathymetry(), smooth_passes=30, min_depth=500.0,
    )
    area2d = np.asarray(grid.dx_2d) * float(grid.dy)
    depth = np.asarray(grid.depth)
    T_atm = np.asarray(load_annual_mean_air_temp(grid, 2023), dtype=np.float64)
    heat_factor = 1.0 / (RHO_0 * C_P * 5.0)

    runs = {}
    group_rows = []
    per_run = {}
    for label, path in [("lambda80", args.lambda80), ("lambda160", args.lambda160)]:
        z, terms, sst, u, v, speed = load_run(path)
        ocean = np.asarray(z["wet_mask"]) > 0.5
        lat = np.asarray(z["lat"])
        lon = np.asarray(z["lon"])
        lon2, lat2 = np.meshgrid(lon, lat, indexing="ij")
        woa = np.asarray(z["T_init"])[:, :, 0]
        error = sst - woa
        dx2d = np.asarray(grid.dx_2d)
        dy = float(grid.dy)
        tx, ty = gradient_ocean(sst, ocean, dx2d, dy)
        grad_mag = np.hypot(tx, ty)
        alignment = (u * tx + v * ty) / np.maximum(speed * grad_mag, 1e-30)
        alignment = np.clip(alignment, -1.0, 1.0)
        adv_proxy = -(u * tx + v * ty) * 86400.0
        bulk = 80.0 * (T_atm - sst) * heat_factor * ocean * 86400.0
        land_dist = np.full(ocean.shape, np.inf)
        # BFS distance from land, periodic in longitude.
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
        land_dist = dist
        groups = {
            "coastal_0_3": ocean & (land_dist <= 3.0),
            "coastal_4_7": ocean & (land_dist > 3.0) & (land_dist <= 7.0),
            "interior_ge8": ocean & (land_dist > 7.0),
        }
        # Optional depth bins within coastal 0-7.
        depth_bins = {
            "coastal_0_7_shallow": ocean & (land_dist <= 7.0) & (depth < 1000.0),
            "coastal_0_7_deep": ocean & (land_dist <= 7.0) & (depth >= 1000.0),
        }
        all_groups = {**groups, **depth_bins}

        result = {
            "global_a2_rmse_c": float(np.sqrt(np.mean(
                (smooth_2d_global(sst, lat, deg=2.0) -
                 smooth_2d_global(woa, lat, deg=2.0))[ocean] ** 2))),
            "regions": {},
        }
        per_run[label] = result
        for group_name, mask in all_groups.items():
            if not np.any(mask):
                continue
            a = area2d[mask]
            depth_node = np.asarray(z["z"])
            upper_mask = (depth_node >= -200.0) & (depth_node < -50.0)
            upper_adv = terms[0][mask][:, upper_mask]
            upper_conv = terms[3][mask][:, upper_mask]
            weights3d = np.broadcast_to(a[:, None], (np.count_nonzero(mask), upper_mask.sum()))
            region_masks = {
                "global_ocean": ocean,
                "north_atlantic_40_60": ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 40.0) & (lat2 <= 60.0),
                "near_wall_55_60": ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 55.0) & (lat2 <= 60.0),
            }
            for region_name, region_mask in region_masks.items():
                m = mask & region_mask
                if not np.any(m):
                    continue
                aa = area2d[m]
                surface_adv = terms[0, :, :, 0][m]
                surface_conv = terms[3, :, :, 0][m]
                surface_gm = terms[4, :, :, 0][m]
                surface_bulk = bulk[m]
                upper_adv = terms[0][m][:, upper_mask]
                upper_conv = terms[3][m][:, upper_mask]
                weights3d = np.broadcast_to(aa[:, None], (np.count_nonzero(m), upper_mask.sum()))
                # Regional metrics
                row = {
                    "run": label, "group": group_name, "region": region_name,
                    "cells": int(m.sum()),
                    "area_share": float(np.sum(aa) / np.sum(area2d[region_mask])),
                    "sst_bias_c": weighted_mean(error[m], aa),
                    "sst_rmse_c": weighted_rms(error[m], aa),
                    "speed_mean_m_s": weighted_mean(speed[m], aa),
                    "speed_rms_m_s": weighted_rms(speed[m], aa),
                    "alignment_mean": weighted_mean(alignment[m], aa),
                    "adv_proxy_mean_k_per_day": weighted_mean(adv_proxy[m], aa),
                    "surface_adv_mean_k_per_day": weighted_mean(surface_adv, aa),
                    "surface_conv_mean_k_per_day": weighted_mean(surface_conv, aa),
                    "surface_gm_mean_k_per_day": weighted_mean(surface_gm, aa),
                    "surface_bulk_mean_k_per_day": weighted_mean(surface_bulk, aa),
                    "surface_net_k_per_day": weighted_mean(surface_adv + surface_conv + surface_gm + surface_bulk, aa),
                    "upper_50_200m_adv_mean_k_per_day": weighted_mean(upper_adv, weights3d),
                    "upper_50_200m_conv_mean_k_per_day": weighted_mean(upper_conv, weights3d),
                    "mean_land_distance_cells": weighted_mean(land_dist[m], aa),
                    "mean_depth_m": weighted_mean(depth[m], aa),
                }
                group_rows.append(row)
                result["regions"].setdefault(region_name, {})[group_name] = row

    fields = ["run", "region", "group", "cells", "area_share", "sst_bias_c",
              "sst_rmse_c", "speed_mean_m_s", "speed_rms_m_s", "alignment_mean",
              "adv_proxy_mean_k_per_day", "surface_adv_mean_k_per_day",
              "surface_conv_mean_k_per_day", "surface_gm_mean_k_per_day",
              "surface_bulk_mean_k_per_day", "surface_net_k_per_day",
              "upper_50_200m_adv_mean_k_per_day", "upper_50_200m_conv_mean_k_per_day",
              "mean_land_distance_cells", "mean_depth_m"]
    with (out / "group_summary.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(fields) + "\n")
        for row in group_rows:
            f.write(",".join(str(row.get(k, "")) for k in fields) + "\n")
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(per_run, f, indent=2)
    print(f"Wrote {out / 'group_summary.csv'}")
    print(f"Wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
