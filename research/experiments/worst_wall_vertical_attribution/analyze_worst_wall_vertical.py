#!/usr/bin/env python3
"""Vertical heat-tendency attribution for the coldest near-wall cells.

This follows the near-wall gradient protocol. It ranks the coldest near-wall
SST cells in the matched GM0 and GM500 0.7-degree runs and compares their
saved surface-to-subsurface heat tendencies.
"""

from pathlib import Path
import argparse
import json
import os
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from ocean_solver.forcing.air import load_annual_mean_air_temp
from ocean_solver.config.definitions import DEFAULT_CONFIG, GlobalGridConfig, RHO_0, C_P
from dataclasses import replace
from ocean_solver.io.grid import make_global_grid

TERM_NAMES = ["advection", "horizontal_diffusion", "vertical_diffusion",
              "convection", "gm_bolus", "redi"]


def find_bathymetry():
    env = os.environ.get("OCEAN_SOLVER_BATHYMETRY")
    if env:
        return env
    candidate = Path.home() / "Desktop" / \
        "ETOPO_2022_v1_r3600x1800_surface.nc"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("Set OCEAN_SOLVER_BATHYMETRY to the ETOPO2022 file.")


def nearest_land_distance(land):
    """BFS distance in grid cells to nearest land, periodic in longitude."""
    land = land.astype(bool)
    nx, ny = land.shape
    dist = np.full((nx, ny), np.inf, dtype=np.float64)
    dist[land] = 0.0
    from collections import deque
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
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def weighted_rms(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.sum(values[valid] ** 2 * weights[valid]) /
                         np.sum(weights[valid])))


def load_run(path, n_worst=100):
    z = np.load(path, allow_pickle=True)
    days = np.asarray(z["days"])
    idx = np.where(days >= days[-1] - 90.0 + 1e-9)[0]
    run_base = Path(path).with_suffix("")
    states = [np.load(run_base.parent / (run_base.name + "_3d") /
                      f"snap_{i:05d}.npy", allow_pickle=True) for i in idx]
    term_frames = [np.load(run_base.parent / (run_base.name + "_terms") /
                           f"terms_{i:05d}.npy", allow_pickle=True) for i in idx]
    terms = np.mean(np.stack(term_frames), axis=0) * 86400.0
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
    near_wall = ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 55.0) & (lat2 <= 60.0)
    cells = np.where(near_wall)
    order = np.argsort(error[cells])
    worst = []
    for rank, flat in enumerate(order[:n_worst], start=1):
        i, j = cells[0][flat], cells[1][flat]
        worst.append((i, j, rank, error[i, j]))
    return z, idx, terms, sst, speed, ocean, lat, lon, lon2, lat2, woa, error, near_wall, worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gm0", default="results/gm0_attribution_07/global_gm0_3dterms.npz")
    parser.add_argument("--gm500", default="results/gm0_attribution_07/global_gm500_3dterms.npz")
    parser.add_argument("--out", default="research/experiments/worst_wall_vertical_attribution")
    parser.add_argument("--n-worst", type=int, default=100)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Rebuild the exact annual 2m air field for the surface bulk term.
    os.environ.setdefault("OCEAN_SOLVER_BATHYMETRY", find_bathymetry())
    grid = make_global_grid(
        replace(GlobalGridConfig(), lat_max=65.0, ny=185, nx=514, resolution=0.7),
        find_bathymetry(), smooth_passes=30, min_depth=100.0,
    )
    T_atm = np.asarray(load_annual_mean_air_temp(grid, 2023), dtype=np.float64)
    dz_surface = 5.0
    heat_factor = 1.0 / (RHO_0 * C_P * dz_surface)

    depth_bins = {
        "surface_node": lambda z: z == 0.0,
        "surface_0_50m": lambda z: z >= -50.0,
        "upper_50_200m": lambda z: (z < -50.0) & (z >= -200.0),
        "intermediate_200_1000m": lambda z: (z < -200.0) & (z >= -1000.0),
    }

    per_run = {}
    term_rows = []
    geometry_rows = []
    for label, path in [("gm0", args.gm0), ("gm500", args.gm500)]:
        z, idx, terms, sst, speed, ocean, lat, lon, lon2, lat2, woa, error, near_wall, worst = load_run(
            path, n_worst=args.n_worst)
        area2d = np.asarray(grid.dx_2d) * float(grid.dy)
        nx, ny, nz = terms.shape[1:]
        depth_node = np.asarray(z["z"])
        land = ~ocean
        land_dist = nearest_land_distance(land)
        # Region groups.
        groups = {
            "near_wall_all": near_wall,
            "cold_lt_minus1": near_wall & (error <= -1.0),
            "coldest_cells": np.zeros_like(near_wall),
        }
        worst_idx = [(i, j) for i, j, _, _ in worst]
        for i, j in worst_idx:
            groups["coldest_cells"][i, j] = True

        # Surface bulk heat tendency, positive into the ocean.
        bulk = 80.0 * (T_atm - sst) * heat_factor * ocean * 86400.0
        per_run[label] = {"worst_cells": []}
        for i, j, rank, err in worst:
            cell_terms = terms[:, i, j, :]
            surface = depth_bins["surface_0_50m"](depth_node)
            conv_surface = weighted_mean(cell_terms[3, surface], area2d[i, j] * np.ones_like(cell_terms[3, surface]))
            conv_subsurf_mask = depth_bins["upper_50_200m"](depth_node)
            conv_subsurf = weighted_mean(cell_terms[3, conv_subsurf_mask], area2d[i, j] * np.ones_like(cell_terms[3, conv_subsurf_mask]))
            per_run[label]["worst_cells"].append({
                "rank": rank,
                "lon_e": float(lon[i]),
                "lat_n": float(lat[j]),
                "raw_error_c": float(err),
                "sst_c": float(sst[i, j]),
                "woa_c": float(woa[i, j]),
                "mean_speed_m_s": float(speed[i, j]),
                "land_distance_cells": float(land_dist[i, j]),
                "surface_convection_k_per_day": float(conv_surface),
                "upper_50_200m_convection_k_per_day": float(conv_subsurf),
                "surface_bulk_flux_k_per_day": float(bulk[i, j]),
            })
        # Aggregate mean terms by depth bin.
        depth_bin_masks = {name: mask(depth_node) for name, mask in depth_bins.items()}
        for group, mask2d in groups.items():
            area = area2d[mask2d]
            if not np.any(area):
                continue
            for bin_name, bin_mask in depth_bin_masks.items():
                selected_terms = terms[:, mask2d, :][:, :, bin_mask]
                weights = np.broadcast_to(area[None, :, None], selected_terms.shape)
                for term_idx, term_name in enumerate(TERM_NAMES):
                    term_weights = weights[term_idx]
                    term_rows.append({
                        "run": label, "group": group, "depth_bin": bin_name,
                        "term": term_name,
                        "mean_k_per_day": weighted_mean(
                            selected_terms[term_idx], term_weights),
                        "rms_k_per_day": weighted_rms(
                            selected_terms[term_idx], term_weights),
                    })
                # Bulk only exists in the surface node.
                if bin_name == "surface_node":
                    bulk_selected = bulk[mask2d]
                    term_rows.append({
                        "run": label, "group": group, "depth_bin": bin_name,
                        "term": "surface_bulk_flux",
                        "mean_k_per_day": weighted_mean(bulk_selected, area),
                        "rms_k_per_day": weighted_rms(bulk_selected, area),
                    })
        for (i, j), cell in zip(worst_idx, per_run[label]["worst_cells"]):
            geometry_rows.append({
                "run": label, **cell,
                "zonal_index": int(i), "meridional_index": int(j),
            })

    # Write outputs.
    with (out / "vertical_terms.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["run", "group", "depth_bin", "term", "mean_k_per_day", "rms_k_per_day"]
        f.write(",".join(fields) + "\n")
        for row in term_rows:
            f.write(",".join(str(row.get(k, "")) for k in fields) + "\n")
    with (out / "worst_cells_geometry.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["run", "rank", "lon_e", "lat_n", "raw_error_c", "sst_c", "woa_c",
                  "mean_speed_m_s", "land_distance_cells",
                  "surface_convection_k_per_day",
                  "upper_50_200m_convection_k_per_day",
                  "surface_bulk_flux_k_per_day", "zonal_index", "meridional_index"]
        f.write(",".join(fields) + "\n")
        for row in geometry_rows:
            f.write(",".join(str(row.get(k, "")) for k in fields) + "\n")
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(per_run, f, indent=2)
    print(f"Wrote {out / 'vertical_terms.csv'}")
    print(f"Wrote {out / 'worst_cells_geometry.csv'}")
    print(f"Wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
