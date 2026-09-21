#!/usr/bin/env python3
"""Compare boundary/polar-cap sensitivity runs against WOA SST.

The reference is the WOA surface temperature already stored in each run's
T_init array.  Region masks use the run lat/lon coordinates, so 40--60N is
the common overlap region even when a run extends to 65N.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from bench_climatology_global import smooth_2d_global


def sst_climatology(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    days = np.asarray(z["days"])
    steady = days >= days[-1] - 90.0
    sst = np.mean(np.asarray(z["T_top"])[steady], axis=0)
    return {
        "sst": sst,
        "woa_sst": np.asarray(z["T_init"])[:, :, 0],
        "ocean": np.asarray(z["wet_mask"]) > 0.5,
        "lat": np.asarray(z["lat"]),
        "lon": np.asarray(z["lon"]),
        "heat": np.asarray(z["heat_content_J"]),
        "salt": np.asarray(z["salt_content_kg"]),
        "verdict": str(z["verdict"]),
    }


def score(run: dict) -> dict:
    sst = run["sst"]
    woa = run["woa_sst"]
    ocean = run["ocean"]
    lat = run["lat"]
    lon = run["lon"]

    zonal_m = np.array([sst[ocean[:, j], j].mean() for j in range(len(lat))])
    zonal_w = np.array([woa[ocean[:, j], j].mean() for j in range(len(lat))])
    a1_corr = float(np.corrcoef(zonal_m, zonal_w)[0, 1])
    a1_rmse = float(np.sqrt(np.mean((zonal_m - zonal_w) ** 2)))

    sst_sm = smooth_2d_global(sst, lat, deg=2.0)
    woa_sm = smooth_2d_global(woa, lat, deg=2.0)
    global_rmse = float(np.sqrt(np.mean((sst_sm[ocean] - woa_sm[ocean]) ** 2)))

    lon2, lat2 = np.meshgrid(lon, lat, indexing="ij")
    north_atlantic = ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 40.0) & (lat2 <= 60.0)
    common_near_wall = ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 55.0) & (lat2 <= 60.0)
    native_near_wall = ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (
        lat2 >= float(lat[-1]) - 5.0
    )
    near_wall_region = north_atlantic & (lat2 >= 55.0)

    raw_error = sst - woa
    a2_error = sst_sm - woa_sm

    def metrics(mask):
        e = raw_error[mask]
        ea = a2_error[mask]
        return {
            "n": int(mask.sum()),
            "raw_bias": float(np.mean(e)),
            "raw_rmse": float(np.sqrt(np.mean(e**2))),
            "a2_bias": float(np.mean(ea)),
            "a2_rmse": float(np.sqrt(np.mean(ea**2))),
        }

    global_sse = float(np.sum(a2_error[ocean] ** 2))
    regional_sse = float(np.sum(a2_error[north_atlantic] ** 2))
    return {
        "verdict": run["verdict"],
        "a1_corr": a1_corr,
        "a1_rmse": a1_rmse,
        "global_a2_rmse": global_rmse,
        "north_atlantic_40_60": metrics(north_atlantic),
        "overlap_near_wall_55_60": metrics(common_near_wall),
        "native_near_wall_top_5deg": metrics(native_near_wall),
        "near_wall_share_of_regional_a2_sse": float(
            np.sum(a2_error[near_wall_region] ** 2) / regional_sse
        ),
        "regional_share_of_global_a2_sse": float(regional_sse / global_sse),
        "heat_drift_percent": float(
            100.0 * (run["heat"][-1] - run["heat"][0]) / abs(run["heat"][0])
        ),
        "salt_drift_percent": float(
            100.0 * (run["salt"][-1] - run["salt"][0]) / abs(run["salt"][0])
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="results/vertical_mixing_sensitivity/global_real_air_kv1e-6_kconv001_repeat.npz",
    )
    parser.add_argument(
        "--lat65",
        default="results/north_boundary_ice_proxy/global_real_air_kv1e-6_kconv001_lat65.npz",
    )
    parser.add_argument(
        "--lat65-repeat",
        default="results/north_boundary_ice_proxy/global_real_air_kv1e-6_kconv001_lat65_repeat.npz",
    )
    parser.add_argument(
        "--polarcap",
        default="results/north_boundary_ice_proxy/global_real_air_kv1e-6_kconv001_polarcap4_6.npz",
    )
    parser.add_argument(
        "--out",
        default="research/experiments/north_boundary_ice_proxy/metrics.json",
    )
    args = parser.parse_args()
    labels = {
        "candidate_baseline": args.baseline,
        "lat65": args.lat65,
        "lat65_repeat": args.lat65_repeat,
        "polarcap4_6": args.polarcap,
    }
    result = {name: score(sst_climatology(Path(path))) for name, path in labels.items()}

    base_a2 = result["candidate_baseline"]["global_a2_rmse"]
    for name in ("lat65", "lat65_repeat", "polarcap4_6"):
        result[name]["global_a2_change_percent"] = 100.0 * (
            result[name]["global_a2_rmse"] / base_a2 - 1.0
        )
        result[name]["north_atlantic_a2_change_percent"] = 100.0 * (
            result[name]["north_atlantic_40_60"]["a2_rmse"]
            / result["candidate_baseline"]["north_atlantic_40_60"]["a2_rmse"]
            - 1.0
        )
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
