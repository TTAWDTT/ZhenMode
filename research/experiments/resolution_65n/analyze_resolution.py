#!/usr/bin/env python3
"""Compare 1-degree and 0.8-degree candidate runs against WOA SST."""

from pathlib import Path
import argparse
import json
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from bench_climatology_global import smooth_2d_global


def load_run(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    days = np.asarray(z["days"])
    steady = days >= days[-1] - 90.0
    return {
        "sst": np.mean(np.asarray(z["T_top"])[steady], axis=0),
        "woa_sst": np.asarray(z["T_init"])[:, :, 0],
        "ocean": np.asarray(z["wet_mask"]) > 0.5,
        "lat": np.asarray(z["lat"]),
        "lon": np.asarray(z["lon"]),
        "heat": np.asarray(z["heat_content_J"]),
        "salt": np.asarray(z["salt_content_kg"]),
        "ke": np.asarray(z["ke"]),
        "max_u": np.asarray(z["max_u"]),
        "max_eta": np.asarray(z["max_eta"]),
        "verdict": str(z["verdict"]),
    }


def score(run: dict) -> dict:
    sst, woa, ocean = run["sst"], run["woa_sst"], run["ocean"]
    lat, lon = run["lat"], run["lon"]
    zonal_m = np.array([sst[ocean[:, j], j].mean() for j in range(len(lat))])
    zonal_w = np.array([woa[ocean[:, j], j].mean() for j in range(len(lat))])
    a1_rmse = float(np.sqrt(np.mean((zonal_m - zonal_w) ** 2)))
    a1_corr = float(np.corrcoef(zonal_m, zonal_w)[0, 1])
    sst_sm = smooth_2d_global(sst, lat, deg=2.0)
    woa_sm = smooth_2d_global(woa, lat, deg=2.0)
    a2_rmse = float(np.sqrt(np.mean((sst_sm[ocean] - woa_sm[ocean]) ** 2)))
    a2_corr = float(np.corrcoef(sst_sm[ocean] - sst_sm[ocean].mean(),
                                woa_sm[ocean] - woa_sm[ocean].mean())[0, 1])
    lon2, lat2 = np.meshgrid(lon, lat, indexing="ij")
    north = ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 40.0) & (lat2 <= 60.0)
    near_wall = ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 55.0) & (lat2 <= 60.0)
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

    return {
        "verdict": run["verdict"],
        "a1_corr": a1_corr,
        "a1_rmse": a1_rmse,
        "a2_corr": a2_corr,
        "global_a2_rmse": a2_rmse,
        "north_atlantic_40_60": metrics(north),
        "overlap_near_wall_55_60": metrics(near_wall),
        "ke_last": float(run["ke"][-1]),
        "max_u_peak": float(np.max(run["max_u"])),
        "max_eta_last": float(run["max_eta"][-1]),
        "heat_drift_percent": float(100.0 * (run["heat"][-1] - run["heat"][0]) / abs(run["heat"][0])),
        "salt_drift_percent": float(100.0 * (run["salt"][-1] - run["salt"][0]) / abs(run["salt"][0])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="results/heat_tendency_decomposition/global_real_air_lambda80_gm500_localconv_3dterms.npz")
    p.add_argument("--res08", default="results/resolution_65n/global_real_air_lambda80_gm500_localconv_res08_365d.npz")
    p.add_argument("--res07", default="results/resolution_65n/global_real_air_lambda80_gm500_localconv_res07_365d.npz")
    p.add_argument("--out", default="research/experiments/resolution_65n/metrics.json")
    args = p.parse_args()
    labels = {
        "res1": args.baseline,
        "res08": args.res08,
        "res07": args.res07,
    }
    result = {name: score(load_run(Path(path))) for name, path in labels.items()}
    base = result["res1"]
    for name in ("res08", "res07"):
        cur = result[name]
        cur["global_a2_change_percent"] = 100.0 * (
            cur["global_a2_rmse"] / base["global_a2_rmse"] - 1.0
        )
        cur["north_atlantic_a2_change_percent"] = 100.0 * (
            cur["north_atlantic_40_60"]["a2_rmse"]
            / base["north_atlantic_40_60"]["a2_rmse"] - 1.0
        )
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
