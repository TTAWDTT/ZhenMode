"""Summarize surface-forcing sensitivity runs against the 365-day FCT baseline."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(REPO / "src"))

from bench_climatology_global import smooth_2d_global  # noqa: E402
from config import GlobalGridConfig  # noqa: E402
from grid import make_global_grid  # noqa: E402
from scipy import ndimage  # noqa: E402


def regional_masks(wet, lat, depth):
    pad = 30
    land = ~wet
    dist = ndimage.distance_transform_edt(
        ~np.pad(land, ((pad, pad), (0, 0)), mode="wrap")
    )[pad:-pad, :]
    rows = np.arange(wet.shape[1], dtype=float)[None, :]
    dbnd = np.minimum(rows, wet.shape[1] - 1 - rows)
    return {
        "near_wall": wet & (dbnd < 3),
        "coast": wet & (dbnd >= 3) & (dist < 3),
        "shallow_interior": wet & (dbnd >= 3) & (dist >= 3) & (depth < 1000),
        "deep_interior": wet & (dbnd >= 3) & (dist >= 3) & (depth >= 1000),
    }


def metric_summary(days, T_top, wet, lat, woa_sst):
    steady = days >= days.max() - 90.0
    if not np.any(steady):
        steady = np.ones_like(days, dtype=bool)
    model = np.mean(T_top[steady], axis=0)
    model_sm = smooth_2d_global(model, lat, deg=2.0)
    woa_sm = smooth_2d_global(woa_sst, lat, deg=2.0)
    error = model_sm - woa_sm

    zonal_model = np.array([
        np.mean(model[wet[:, j], j]) if wet[:, j].any() else np.nan
        for j in range(wet.shape[1])
    ])
    zonal_woa = np.array([
        np.mean(woa_sst[wet[:, j], j]) if wet[:, j].any() else np.nan
        for j in range(wet.shape[1])
    ])
    good = np.isfinite(zonal_model) & np.isfinite(zonal_woa)
    a1_corr = float(np.corrcoef(zonal_model[good], zonal_woa[good])[0, 1])
    a1_rmse = float(np.sqrt(np.mean((zonal_model[good] - zonal_woa[good]) ** 2)))
    a_model = model_sm[wet] - np.mean(model_sm[wet])
    a_woa = woa_sm[wet] - np.mean(woa_sm[wet])
    a2_corr = float(np.corrcoef(a_model, a_woa)[0, 1])
    a2_rmse = float(np.sqrt(np.mean(error[wet] ** 2)))
    return {
        "days_used": int(steady.sum()),
        "a1_corr": a1_corr,
        "a1_rmse": a1_rmse,
        "a2_corr": a2_corr,
        "a2_rmse": a2_rmse,
        "mean_error": float(np.mean(error[wet])),
        "rmse": a2_rmse,
        "max_abs": float(np.max(np.abs(error[wet]))),
        "error": error,
    }


def fmt(v, digits=4):
    return "NA" if not np.isfinite(v) else f"{v:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/fct_transport/global_fct_tvd_1deg_365d.npz")
    ap.add_argument("--sensitivity", nargs="+", default=[
        "results/forcing_sensitivity/global_lambda_0.25_1deg_365d.npz",
        "results/forcing_sensitivity/global_lambda_0.5_1deg_365d.npz",
        "results/forcing_sensitivity/global_lambda_2.0_1deg_365d.npz",
    ])
    ap.add_argument("--out-dir", default="research/experiments/forcing_sensitivity")
    ap.add_argument("--bathymetry", default=os.environ.get("OCEAN_SOLVER_BATHYMETRY", ""))
    args = ap.parse_args()
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if not args.bathymetry:
        raise RuntimeError("set OCEAN_SOLVER_BATHYMETRY or pass --bathymetry")

    runs = [("lambda1_baseline", args.baseline)]
    for p in args.sensitivity:
        mult = Path(p).stem.replace("global_lambda_", "").replace("_1deg_365d", "")
        runs.append((f"lambda{mult}", p))

    rows = []
    region_rows = []
    base_budget = np.load(REPO / "results/diagnostics_budget/global_fct_1deg_365d_budget.npz")
    gc = GlobalGridConfig(nx=360, ny=120, lat_max=60.0)
    grid = make_global_grid(gc, args.bathymetry, smooth_passes=30, min_depth=100.0)
    depth = np.asarray(grid.depth)
    for name, rel in runs:
        d = np.load(REPO / rel)
        wet = np.asarray(d["wet_mask"] > 0.5)
        lat = np.asarray(d["lat"], dtype=float)
        woa_sst = np.asarray(d["T_init"][:, :, 0], dtype=float)
        bd = d if "heat_content_J" in d.files else base_budget
        summary = metric_summary(np.asarray(d["days"]), np.asarray(d["T_top"]),
                                 wet, lat, woa_sst)
        heat_drift = float(
            (bd["heat_content_J"][-1] - bd["heat_content_J"][0])
            / abs(bd["heat_content_J"][0])
        )
        salt_drift = float(
            (bd["salt_content_kg"][-1] - bd["salt_content_kg"][0])
            / abs(bd["salt_content_kg"][0])
        )
        rows.append([
            name, summary["days_used"], fmt(summary["a1_corr"], 3),
            fmt(summary["a1_rmse"], 3), fmt(summary["a2_corr"], 3),
            fmt(summary["a2_rmse"], 3), fmt(summary["mean_error"], 3),
            fmt(float(d["max_u_peak"])), fmt(float(d["max_eta"][-1])),
            fmt(heat_drift, 6), fmt(salt_drift, 8),
        ])
        # Regional shares are relative to this run's own total SSE.
        regions = regional_masks(wet, lat, depth)
        total_sse = float(np.sum(summary["error"][wet] ** 2))
        for region, mask in regions.items():
            share = float(np.sum(summary["error"][mask] ** 2) / total_sse)
            region_rows.append([
                name, region, int(mask.sum()),
                fmt(float(np.mean(summary["error"][mask])), 3),
                fmt(float(np.sqrt(np.mean(summary["error"][mask] ** 2))), 3),
                fmt(100.0 * share, 2),
            ])

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "snapshots", "a1_corr", "a1_rmse", "a2_corr",
                    "a2_rmse", "mean_error", "max_u_peak", "max_eta_final",
                    "heat_rel_drift", "salt_rel_drift"])
        w.writerows(rows)
    with (out / "region_shares.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "region", "count", "mean_error", "rmse", "sse_share"])
        w.writerows(region_rows)

    base_row = rows[0]
    lines = [
        "# Surface Forcing Sensitivity Results", "",
        "## Runs", "",
        "| run | snaps | A1 corr | A1 RMSE | A2 corr | A2 RMSE | mean error | max-u | max-eta | heat drift | salt drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines += ["", "## Regional SSE shares", "",
              "| run | near_wall | coast | shallow_interior | deep_interior |",
              "|---|---:|---:|---:|---:|"]
    for run_name, _ in runs:
        shares = {r[1]: r[5] for r in region_rows if r[0] == run_name}
        lines.append(f"| {run_name} | {shares.get('near_wall')}% | "
                     f"{shares.get('coast')}% | {shares.get('shallow_interior')}% | "
                     f"{shares.get('deep_interior')}% |")
    lines += [
        "", "## Reading", "",
        "- `lambda1` is the existing baseline (`bulk_lambda_mult=1`).",
        "- Lower values weaken restoring; higher values strengthen restoring.",
        "- The key criterion is global A2 RMSE, not only coastal RMSE.",
        "- A full 2D WOA SST restoring target is deliberately not used because",
        "  it would make the A2 score circular.",
    ]
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'analysis.md'}")


if __name__ == "__main__":
    main()




