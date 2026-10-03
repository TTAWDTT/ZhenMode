#!/usr/bin/env python3
"""Quantify near-wall SST gradients and local advective cooling at 0.7 degree.

This is a diagnostic-only analysis. It compares the locked GM0 candidate with
the matched GM500 run over the last 90 days. The purpose is to identify whether
the remaining North Atlantic cold bias is tied to temperature gradients that
are aligned with the mean current.
"""

from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from ocean_solver.validation.benchmarks.climatology import smooth_2d_global

EARTH_RADIUS_M = 6371.0e3


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


def gradient_ocean(field, ocean, dx2d, dy):
    """Horizontal gradient with periodic longitude and ocean-aware one-sided y."""
    nx, ny = field.shape
    tx = np.zeros_like(field, dtype=np.float64)
    ty = np.zeros_like(field, dtype=np.float64)

    # Periodic central/one-sided zonal differences. Use both neighbors when wet.
    ip = np.roll(np.arange(nx), -1)
    im = np.roll(np.arange(nx), 1)
    for j in range(ny):
        for i in range(nx):
            if not ocean[i, j]:
                continue
            right = ocean[ip[i], j]
            left = ocean[im[i], j]
            if right and left:
                tx[i, j] = (field[ip[i], j] - field[im[i], j]) / (2.0 * dx2d[i, j])
            elif right:
                tx[i, j] = (field[ip[i], j] - field[i, j]) / dx2d[i, j]
            elif left:
                tx[i, j] = (field[i, j] - field[im[i], j]) / dx2d[i, j]

    # Uniform lat centers; use wet one-sided differences at poleward boundaries.
    for j in range(ny):
        for i in range(nx):
            if not ocean[i, j]:
                continue
            down = j - 1
            up = j + 1
            if down >= 0 and up < ny and ocean[i, down] and ocean[i, up]:
                # Central spacing is dy because lat centers are uniform.
                ty[i, j] = (field[i, up] - field[i, down]) / (2.0 * dy)
            elif up < ny and ocean[i, up]:
                ty[i, j] = (field[i, up] - field[i, j]) / dy
            elif down >= 0 and ocean[i, down]:
                ty[i, j] = (field[i, j] - field[i, down]) / dy
    return tx, ty


def summarize(mask, error, tx, ty, speed, alignment, adv_proxy, saved_adv, area, sst, woa, cold_mask):
    grad_mag = np.hypot(tx, ty)
    return {
        "cells": int(mask.sum()),
        "area_m2": float(np.sum(area[mask])) if mask.any() else 0.0,
        "sst_bias_c": weighted_mean(sst[mask] - woa[mask], area[mask]),
        "sst_rmse_c": weighted_rms_from_error(error[mask], area[mask]),
        "grad_east_mean_k_per_100km": weighted_mean(tx[mask] * 1.0e5, area[mask]),
        "grad_north_mean_k_per_100km": weighted_mean(ty[mask] * 1.0e5, area[mask]),
        "grad_mag_mean_k_per_100km": weighted_mean(grad_mag[mask] * 1.0e5, area[mask]),
        "grad_mag_p90_k_per_100km": weighted_percentile(grad_mag[mask], area[mask], 90) * 1.0e5,
        "speed_mean_m_s": weighted_mean(speed[mask], area[mask]),
        "alignment_mean": weighted_mean(alignment[mask], area[mask]),
        "alignment_mean_cold": weighted_mean(alignment[mask & cold_mask], area[mask & cold_mask]) if np.any(mask & cold_mask) else float("nan"),
        "adv_tend_mean_k_per_day": weighted_mean(adv_proxy[mask], area[mask]),
        "adv_tend_rms_k_per_day": weighted_rms(adv_proxy[mask], area[mask]),
        "saved_adv_mean_k_per_day": weighted_mean(saved_adv[mask], area[mask]),
        "saved_adv_rms_k_per_day": weighted_rms(saved_adv[mask], area[mask]),
        "error_saved_adv_corr": weighted_corr(error[mask], saved_adv[mask], area[mask]),
        "cold_error_saved_adv_corr": weighted_corr(error[mask & cold_mask], saved_adv[mask & cold_mask], area[mask & cold_mask]) if np.any(mask & cold_mask) else float("nan"),
        "cold_lt_minus1_share": float(np.sum(error[mask] <= -1.0) / mask.sum()) if mask.any() else float("nan"),
        "cold_lt_minus2_share": float(np.sum(error[mask] <= -2.0) / mask.sum()) if mask.any() else float("nan"),
    }


def weighted_rms_from_error(error, weights):
    return weighted_rms(error, weights)


def weighted_corr(a, b, weights):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ma = weighted_mean(a, w); mb = weighted_mean(b, w)
    cov = weighted_mean((a - ma) * (b - mb), w)
    va = weighted_mean((a - ma) ** 2, w)
    vb = weighted_mean((b - mb) ** 2, w)
    return float(cov / np.sqrt(max(va * vb, 1e-30)))


def weighted_percentile(values, weights, q):
    order = np.argsort(values)
    vals = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cw = np.cumsum(w)
    cutoff = 0.01 * q * cw[-1]
    return float(vals[min(np.searchsorted(cw, cutoff), len(vals) - 1)])


def load_run(path):
    z = np.load(path, allow_pickle=True)
    days = np.asarray(z["days"])
    steady = days >= days[-1] - 90.0 + 1e-9
    idx = np.where(steady)[0]
    if len(idx) == 0:
        raise RuntimeError(f"No snapshots in final 90d: {path}")
    run_base = Path(path).with_suffix("")
    states = [np.load(run_base.parent / (run_base.name + "_3d") / f"snap_{i:05d}.npy",
                      allow_pickle=True) for i in idx]
    term_frames = [np.load(run_base.parent / (run_base.name + "_terms") /
                            f"terms_{i:05d}.npy", allow_pickle=True) for i in idx]
    # Saved term 0 is total advection in K/s from the solver.
    adv = np.mean(np.stack([term_frames[i][0] for i in range(len(term_frames))]),
                  axis=0)[:, :, 0] * 86400.0
    T = np.mean(np.stack([s[0] for s in states]), axis=0)
    u = np.mean(np.stack([s[1] for s in states]), axis=0)
    v = np.mean(np.stack([s[2] for s in states]), axis=0)
    return z, idx, T[:, :, 0], u[:, :, 0], v[:, :, 0], adv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gm0", default="results/gm0_attribution_07/global_gm0_3dterms.npz")
    parser.add_argument("--gm500", default="results/gm0_attribution_07/global_gm500_3dterms.npz")
    parser.add_argument("--out", default="research/experiments/near_wall_gradient_07")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    per_run = {}
    worst_cells = None
    reference = None

    for label, path in [("gm0", args.gm0), ("gm500", args.gm500)]:
        z, idx, sst, u, v, saved_adv = load_run(path)
        ocean = np.asarray(z["wet_mask"]) > 0.5
        lat = np.asarray(z["lat"])
        lon = np.asarray(z["lon"])
        lon2, lat2 = np.meshgrid(lon, lat, indexing="ij")
        woa = np.asarray(z["T_init"])[:, :, 0]
        ny = len(lat)
        dy = EARTH_RADIUS_M * math.radians(abs(float(lat[1] - lat[0])))
        cos_lat = np.cos(np.radians(lat))
        dx2d = np.repeat((EARTH_RADIUS_M * math.radians(float(lon[1] - lon[0])) *
                          cos_lat)[None, :], len(lon), axis=0)
        area = dx2d * dy

        # Official-style large-scale A2 for context.
        sst_sm = smooth_2d_global(sst, lat, deg=2.0)
        woa_sm = smooth_2d_global(woa, lat, deg=2.0)
        raw_error = sst - woa
        a2_error = sst_sm - woa_sm
        global_a2 = float(np.sqrt(np.mean(a2_error[ocean] ** 2)))

        tx, ty = gradient_ocean(sst, ocean, dx2d, dy)
        grad_mag = np.hypot(tx, ty)
        speed = np.hypot(u, v)
        alignment = (u * tx + v * ty) / np.maximum(speed * grad_mag, 1e-30)
        alignment = np.clip(alignment, -1.0, 1.0)
        # Advective temperature tendency: -u dot grad(T), K/day.
        adv_proxy = -(u * tx + v * ty) * 86400.0

        # Region masks. The "wall" is the northern closed boundary, so
        # 55..60N is the near-wall sector relevant to the cold bias.
        na = ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 40.0) & (lat2 <= 60.0)
        near_wall = ocean & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 55.0) & (lat2 <= 60.0)
        deep_na = na & (lat2 >= 40.0) & (lat2 < 55.0)
        regions = {
            "north_atlantic_40_60": na,
            "near_wall_55_60": near_wall,
            "interior_40_55": deep_na,
        }
        cold_mask = raw_error <= -1.0
        result = {
            "run": path,
            "snapshot_indices": idx.tolist(),
            "snapshot_days": np.asarray(z["days"])[idx].tolist(),
            "global_a2_rmse_c": global_a2,
            "regions": {},
        }
        for name, mask in regions.items():
            row = {
                "region": name,
                "run": label,
                **summarize(mask, raw_error, tx, ty, speed, alignment,
                            adv_proxy, saved_adv, area, sst, woa, cold_mask),
            }
            per_run.setdefault(label, {})[name] = row
            all_summaries.append(row)
            result["regions"][name] = row

        # Save the worst A2-error cells from the near-wall sector.
        sel = near_wall.copy()
        selected = np.where(sel)
        order = np.argsort(raw_error[selected])
        rows = []
        for rank, flat_idx in enumerate(order[:100], start=1):
            i, j = selected[0][flat_idx], selected[1][flat_idx]
            rows.append({
                "rank": rank,
                "run": label,
                "lon_e": float(lon[i]),
                "lat_n": float(lat[j]),
                "sst_c": float(sst[i, j]),
                "woa_c": float(woa[i, j]),
                "raw_error_c": float(raw_error[i, j]),
                "grad_east_k_per_100km": float(tx[i, j] * 1.0e5),
                "grad_north_k_per_100km": float(ty[i, j] * 1.0e5),
                "grad_mag_k_per_100km": float(grad_mag[i, j] * 1.0e5),
                "u_m_s": float(u[i, j]),
                "v_m_s": float(v[i, j]),
                "speed_m_s": float(speed[i, j]),
                "alignment": float(alignment[i, j]),
                "adv_tend_k_per_day": float(adv_proxy[i, j]),
            })
        if worst_cells is None:
            worst_cells = rows
        else:
            worst_cells.extend(rows)

        if label == "gm0":
            reference = {
                "lon2": lon2, "lat2": lat2, "raw_error": raw_error,
                "grad_mag": grad_mag, "speed": speed,
                "alignment": alignment, "adv_proxy": adv_proxy,
            }

    summary_fields = [
        "run", "region", "cells", "sst_bias_c", "sst_rmse_c",
        "grad_east_mean_k_per_100km", "grad_north_mean_k_per_100km",
        "grad_mag_mean_k_per_100km", "grad_mag_p90_k_per_100km",
        "speed_mean_m_s", "alignment_mean", "alignment_mean_cold",
        "adv_tend_mean_k_per_day", "adv_tend_rms_k_per_day",
        "saved_adv_mean_k_per_day", "saved_adv_rms_k_per_day",
        "error_saved_adv_corr", "cold_error_saved_adv_corr",
        "cold_lt_minus1_share", "cold_lt_minus2_share",
    ]
    with (out / "region_summary.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(summary_fields) + "\n")
        for row in all_summaries:
            f.write(",".join(str(row.get(k, "")) for k in summary_fields) + "\n")

    worst_fields = list(worst_cells[0].keys())
    with (out / "worst_near_wall_cells.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(worst_fields) + "\n")
        for row in worst_cells:
            f.write(",".join(str(row.get(k, "")) for k in worst_fields) + "\n")

    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"runs": per_run}, f, indent=2)

    print(f"Wrote {out / 'region_summary.csv'}")
    print(f"Wrote {out / 'worst_near_wall_cells.csv'}")
    print(f"Wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
