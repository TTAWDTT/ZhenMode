#!/usr/bin/env python3
"""Decompose 3D heat tendencies in the preferred 65N candidate.

The saved 3D terms are dT/dt in K/s for:
[advection, horizontal diffusion, vertical diffusion, convection, GM, Redi].
The surface bulk heat-flux tendency is diagnosed separately because the saved
stack intentionally contains interior processes only.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from air_reanalysis import load_annual_mean_air_temp
from bench_climatology_global import smooth_2d_global
from config import DEFAULT_CONFIG, C_P, GlobalGridConfig, RHO_0
from dataclasses import replace
from grid import make_global_grid

TERM_NAMES = ["advection", "horizontal_diffusion", "vertical_diffusion",
              "convection", "gm_bolus", "redi"]


def weighted_mean(values, weights):
    return float(np.sum(values * weights) / np.sum(weights))


def term_stats(stack: np.ndarray, mask3d: np.ndarray, weight3d: np.ndarray) -> dict:
    selected = stack[:, mask3d]
    weights = np.broadcast_to(weight3d[mask3d][None, :], selected.shape)
    signed = weighted_mean(selected, weights)
    rms = float(np.sqrt(np.sum(selected**2 * weights) / np.sum(weights)))
    return {
        "mean_k_per_day": signed * 86400.0,
        "rms_k_per_day": rms * 86400.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        default="results/heat_tendency_decomposition/global_real_air_lambda80_gm500_localconv_3dterms.npz",
    )
    parser.add_argument(
        "--lambda-bulk", type=float, default=80.0,
    )
    parser.add_argument(
        "--out", default="research/experiments/heat_tendency_decomposition/metrics.json",
    )
    args = parser.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    days = np.asarray(z["days"])
    ocean = np.asarray(z["wet_mask"]) > 0.5
    lat = np.asarray(z["lat"])
    lon = np.asarray(z["lon"])
    depth_node = np.asarray(z["z"])
    lon2, lat2 = np.meshgrid(lon, lat, indexing="ij")

    # Rebuild the run grid for real horizontal area weights and air temperature.
    config = str(z["config"])
    lat_max = 65.0
    ny = 130
    grid = make_global_grid(
        replace(GlobalGridConfig(), lat_max=lat_max, ny=ny),
        DEFAULT_CONFIG.bathymetry_file,
        smooth_passes=30,
        min_depth=100.0,
    )
    area2d = np.asarray(grid.dx_2d) * float(grid.dy)
    nz = len(depth_node); nx, ny = ocean.shape; area3d = np.broadcast_to(area2d[:, :, None], (nx, ny, nz))
    T_atm = np.asarray(load_annual_mean_air_temp(grid, 2023), dtype=np.float64)

    # Final 90-day snapshots.
    idx = np.where(days >= days[-1] - 90.0)[0]
    run_stem = Path(args.npz).with_suffix(""); run_base = run_stem.parent / run_stem.name

    regions = {
        "north_atlantic_40_60": ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 40.0) & (lat2 <= 60.0),
        "near_wall_55_60": ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 55.0) & (lat2 <= 60.0),
    }

    depth_bins = {
        "surface_0_50m": (depth_node >= -50.0),
        "upper_50_200m": (depth_node < -50.0) & (depth_node >= -200.0),
        "intermediate_200_1000m": (depth_node < -200.0) & (depth_node >= -1000.0),
        "deep_below_1000m": (depth_node < -1000.0),
    }

    # Time-mean SST and official A2 metric for context.
    steady = days >= days[-1] - 90.0
    sst = np.mean(np.asarray(z["T_top"])[steady], axis=0)
    woa = np.asarray(z["T_init"])[:, :, 0]
    sst_sm = smooth_2d_global(sst, lat, deg=2.0)
    woa_sm = smooth_2d_global(woa, lat, deg=2.0)
    a2_rmse = float(np.sqrt(np.mean((sst_sm[ocean] - woa_sm[ocean]) ** 2)))
    raw_error = sst - woa

    out = {
        "run": args.npz,
        "verdict": str(z["verdict"]),
        "snapshot_days_used": days[idx].tolist(),
        "global_a2_rmse": a2_rmse,
        "regions": {},
    }

    for region_name, region2d in regions.items():
        mask3d = np.broadcast_to(region2d[:, :, None], (nx, ny, nz))
        weight3d = area3d * mask3d
        weights_surface = area2d[region2d]
        surface_error = raw_error[region2d]
        surface_weights = area2d[region2d]

        # Surface bulk heat-flux tendency from lambda*(T_atm-SST)/(rho cp dz).
        bulk_stack = []
        surface_term_means = {name: [] for name in TERM_NAMES}
        surface_term_rms = {name: [] for name in TERM_NAMES}
        for i in idx:
            state = np.load(Path(str(run_base) + "_3d") / f"snap_{i:05d}.npy")
            terms = np.load(Path(str(run_base) + "_terms") / f"terms_{i:05d}.npy")
            sst_surface = state[0, :, :, 0]
            bulk = (
                args.lambda_bulk * (T_atm - sst_surface)
                / (RHO_0 * C_P * 5.0)
            )
            bulk_stack.append(bulk)
            for k, name in enumerate(TERM_NAMES):
                surface_term_means[name].append(terms[k, :, :, 0])
                surface_term_rms[name].append(terms[k, :, :, 0])

        bulk_stack = np.asarray(bulk_stack)

        surface_stats = {}
        for k, name in enumerate(TERM_NAMES):
            arr = np.asarray(surface_term_means[name])[:, region2d]
            surface_stats[name] = {
                "mean_k_per_day": weighted_mean(arr, np.broadcast_to(weights_surface, arr.shape)) * 86400.0,
                "rms_k_per_day": float(np.sqrt(np.sum(arr**2 * weights_surface) / np.sum(weights_surface))) * 86400.0,
            }
        bulk_selected = bulk_stack[:, region2d]
        surface_stats["surface_bulk_flux"] = {
            "mean_k_per_day": weighted_mean(
                bulk_selected, np.broadcast_to(weights_surface, bulk_selected.shape)
            ) * 86400.0,
            "rms_k_per_day": float(np.sqrt(np.sum(bulk_selected**2 * weights_surface) / np.sum(weights_surface))) * 86400.0,
        }
        all_surface_means = {**surface_stats}
        denom = sum(abs(v["mean_k_per_day"]) for v in all_surface_means.values())
        for v in all_surface_means.values():
            v["mean_abs_share"] = abs(v["mean_k_per_day"]) / denom if denom else 0.0
        denom_rms = sum(v["rms_k_per_day"] for v in all_surface_means.values())
        for v in all_surface_means.values():
            v["rms_share"] = v["rms_k_per_day"] / denom_rms if denom_rms else 0.0

        # Correlation between local SST error and time-mean local tendencies.
        correlations = {}
        mean_terms_surface = {}
        for k, name in enumerate(TERM_NAMES):
            arr = np.asarray(surface_term_means[name])[:, region2d]
            mean_terms_surface[name] = np.mean(arr, axis=0)
        mean_bulk_surface = np.mean(bulk_selected, axis=0)
        for name in TERM_NAMES:
            x = mean_terms_surface[name]
            if np.std(x) > 0 and np.std(surface_error) > 0:
                correlations[name] = float(np.corrcoef(x, surface_error)[0, 1])
            else:
                correlations[name] = 0.0
        correlations["surface_bulk_flux"] = (
            float(np.corrcoef(mean_bulk_surface, surface_error)[0, 1])
            if np.std(mean_bulk_surface) > 0 and np.std(surface_error) > 0 else 0.0
        )

        depth_profiles = {}
        for bin_name, bin_mask_1d in depth_bins.items():
            bin_mask3d = mask3d & np.broadcast_to(bin_mask_1d[None, None, :], mask3d.shape)
            weight3d_bin = area3d * bin_mask3d
            profile = {}
            for k, name in enumerate(TERM_NAMES):
                arr = np.asarray([
                    np.load(Path(str(run_base) + "_terms") / f"terms_{i:05d}.npy")[k]
                    for i in idx
                ])
                profile[name] = term_stats(arr, bin_mask3d, weight3d_bin)
            depth_profiles[bin_name] = profile

        region_out = {
            "surface_cells": int(region2d.sum()),
            "sst_mean_model_c": weighted_mean(sst[region2d], weights_surface),
            "sst_mean_woa_c": weighted_mean(woa[region2d], weights_surface),
            "sst_bias_c": weighted_mean(raw_error[region2d], weights_surface),
            "sst_rmse_c": float(np.sqrt(np.sum(raw_error[region2d]**2 * weights_surface) / np.sum(weights_surface))),
            "surface_tendencies": surface_stats,
            "surface_error_tendency_correlations": correlations,
            "depth_profiles": depth_profiles,
        }
        out["regions"][region_name] = region_out

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()





