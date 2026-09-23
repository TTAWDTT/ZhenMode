#!/usr/bin/env python3
"""Score the 0.45-degree GM/boundary-transport A/B ladder."""

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
    steady = days >= days[-1] - min(90.0, max(days[-1] - 1.0, 1.0))
    return {
        "path": str(path),
        "days_end": float(days[-1]),
        "sst": np.mean(np.asarray(z["T_top"])[steady], axis=0),
        "woa_sst": np.asarray(z["T_init"])[:, :, 0],
        "ocean": np.asarray(z["wet_mask"]) > 0.5,
        "lat": np.asarray(z["lat"]),
        "lon": np.asarray(z["lon"]),
        "ke": np.asarray(z["ke"]),
        "max_u_peak": float(z["max_u_peak"]),
        "max_eta_last": float(z["max_eta"][-1]),
        "verdict": str(z["verdict"]),
    }


def score(run: dict) -> dict:
    sst, woa, ocean = run["sst"], run["woa_sst"], run["ocean"]
    lat, lon = run["lat"], run["lon"]
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
        e, ea = raw_error[mask], a2_error[mask]
        return {
            "n": int(mask.sum()),
            "raw_bias": float(np.mean(e)),
            "raw_rmse": float(np.sqrt(np.mean(e ** 2))),
            "a2_rmse": float(np.sqrt(np.mean(ea ** 2))),
        }

    return {
        "path": run["path"],
        "verdict": run["verdict"],
        "days_end": run["days_end"],
        "global_a2_rmse": a2_rmse,
        "north_atlantic_40_60": metrics(north),
        "near_wall_55_60": metrics(near_wall),
        "max_u_peak": run["max_u_peak"],
        "max_eta_last": run["max_eta_last"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", required=True)
    p.add_argument("--gm500", default="results/boundary_transport_closure_045/global_res045_gm500_30d.npz")
    p.add_argument("--gm1000", default="results/boundary_transport_closure_045/global_res045_gm1000_30d.npz")
    p.add_argument("--out", default="research/experiments/boundary_transport_closure_045/metrics_30d.json")
    args = p.parse_args()
    labels = {"control": args.control, "gm500": args.gm500, "gm1000": args.gm1000}
    result = {}
    for label, path in labels.items():
        if Path(path).exists():
            result[label] = score(load_run(Path(path)))
        else:
            print(f"skip missing {label}: {path}")
    if "control" not in result:
        raise SystemExit("control run not found")
    if "gm500" in result:
        result["gm500"]["global_a2_change_percent"] = 100.0 * (
            result["gm500"]["global_a2_rmse"] / result["control"]["global_a2_rmse"] - 1.0)
    if "gm1000" in result:
        result["gm1000"]["global_a2_change_percent"] = 100.0 * (
            result["gm1000"]["global_a2_rmse"] / result["control"]["global_a2_rmse"] - 1.0)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
