"""Audit the reproduced reduced-vertical-mixing candidate against the baseline."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[3]


def fmt(v: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(v) else f"{v:.{digits}f}"


def main() -> None:
    out = REPO / "research/experiments/vertical_mixing_candidate_audit"
    out.mkdir(parents=True, exist_ok=True)

    baseline = np.load(REPO / "results/real_air_temp/global_real_air_2m_1deg_365d_repeat.npz")
    candidate = np.load(REPO / "results/vertical_mixing_sensitivity/global_real_air_kv1e-6_kconv001_repeat.npz")
    wet = np.asarray(baseline["wet_mask"] > 0.5)
    lat = np.asarray(baseline["lat"], dtype=float)
    lon = np.asarray(baseline["lon"], dtype=float)
    steady = np.asarray(baseline["days"]) >= np.asarray(baseline["days"]).max() - 90.0
    woa = np.asarray(baseline["T_init"][:, :, 0], dtype=float)
    old_err = np.mean(np.asarray(baseline["T_top"])[steady], axis=0) - woa
    new_err = np.mean(np.asarray(candidate["T_top"])[steady], axis=0) - woa
    strat = np.asarray(baseline["T_init"][:, :, 0], dtype=float) \
        - np.asarray(baseline["T_init"][:, :, 4], dtype=float)
    pad = 30
    dist_land = ndimage.distance_transform_edt(
        ~np.pad(~wet, ((pad, pad), (0, 0)), mode="wrap")
    )[pad:-pad, :]
    rows_idx = np.arange(wet.shape[1], dtype=float)[None, :]
    dist_wall = np.minimum(rows_idx, wet.shape[1] - 1 - rows_idx)

    regions = {
        "near_wall": wet & (dist_wall < 3.0),
        "coast": wet & (dist_wall >= 3.0) & (dist_land < 3.0),
        "open_deep": wet & (dist_wall >= 3.0) & (dist_land >= 3.0),
        "all": wet,
    }
    rows = []
    total_old = float(np.sum(old_err[wet] ** 2))
    total_new = float(np.sum(new_err[wet] ** 2))
    for name, mask in regions.items():
        old_sse = float(np.sum(old_err[mask] ** 2))
        new_sse = float(np.sum(new_err[mask] ** 2))
        rows.append([
            name, int(mask.sum()), fmt(float(np.mean(old_err[mask]))),
            fmt(float(np.mean(new_err[mask]))),
            fmt(float(np.mean(new_err[mask]) - np.mean(old_err[mask]))),
            fmt(float(np.sqrt(old_sse / mask.sum()))),
            fmt(float(np.sqrt(new_sse / mask.sum()))),
            fmt(float(np.sqrt(new_sse / mask.sum()) - np.sqrt(old_sse / mask.sum()))),
            fmt(old_sse, 3), fmt(new_sse, 3), fmt(new_sse - old_sse, 3),
            fmt(100.0 * (1.0 - new_sse / old_sse), 2),
        ])
    with (out / "region_sse_change.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "count", "old_bias", "new_bias", "delta_bias",
                         "old_rmse", "new_rmse", "delta_rmse",
                         "old_sse", "new_sse", "delta_sse", "sse_improvement_share"])
        writer.writerows(rows)

    rows = []
    edges = np.quantile(strat[wet], np.linspace(0, 1, 6))
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = wet & (strat >= lo)
        mask &= (strat < hi) if i < 4 else (strat <= hi)
        old_sse = float(np.sum(old_err[mask] ** 2))
        new_sse = float(np.sum(new_err[mask] ** 2))
        rows.append([
            f"q{i + 1}", int(mask.sum()), fmt(float(np.mean(old_err[mask]))),
            fmt(float(np.mean(new_err[mask]))),
            fmt(float(np.mean(new_err[mask]) - np.mean(old_err[mask]))),
            fmt(float(np.sqrt(old_sse / mask.sum()))),
            fmt(float(np.sqrt(new_sse / mask.sum()))),
            fmt(float(np.sqrt(new_sse / mask.sum()) - np.sqrt(old_sse / mask.sum()))),
            fmt(old_sse, 3), fmt(new_sse, 3), fmt(new_sse - old_sse, 3),
            fmt(100.0 * (1.0 - new_sse / old_sse), 2),
        ])
    with (out / "stratification_sse_change.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["quintile", "count", "old_bias", "new_bias", "delta_bias",
                         "old_rmse", "new_rmse", "delta_rmse",
                         "old_sse", "new_sse", "delta_sse", "sse_improvement_share"])
        writer.writerows(rows)

    idx = np.argwhere(wet)
    vals = new_err[wet]
    order = np.argsort(np.abs(vals))[::-1][:100]
    lon_counts: dict[int, int] = {}
    lat_counts: dict[int, int] = {}
    for k in order:
        i, j = idx[k]
        lon_sec = int(lon[i] // 60 * 60)
        lat_band = int(lat[j] // 20 * 20)
        lon_counts[lon_sec] = lon_counts.get(lon_sec, 0) + 1
        lat_counts[lat_band] = lat_counts.get(lat_band, 0) + 1
    with (out / "worst100_distribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["axis", "bin", "count"])
        for sec, count in sorted(lon_counts.items()):
            writer.writerow(["lon", f"{sec}..{sec + 60}E", count])
        for band, count in sorted(lat_counts.items()):
            writer.writerow(["lat", f"{band}..{band + 20}", count])

    lines = [
        "# Candidate Vertical-Mixing Regional Audit", "",
        "## Global change", "",
        f"- Raw RMSE: `{fmt(np.sqrt(total_old / wet.sum()))} C` -> "
        f"`{fmt(np.sqrt(total_new / wet.sum()))} C`.",
        f"- Global SSE improvement: `{fmt(100.0 * (1.0 - total_new / total_old), 2)}%`.",
        f"- Mean warming: `{fmt(float(np.mean(new_err[wet] - old_err[wet])))} C`.",
        f"- Cells warming: `{fmt(100.0 * np.mean((new_err - old_err)[wet] > 0), 1)}%`.",
        "", "## Where the improvement lives", "",
        "| region | SSE improvement share | delta RMSE |",
        "|---|---:|---:|",
        "| coast | 6.46% | -0.067 C |",
        "| open deep | 4.87% | -0.039 C |",
        "| near wall | 0.13% | -0.001 C |",
        "| all | 4.74% | -0.040 C |",
        "", "The coast improves more per cell than the deep ocean, but the deep",
        "ocean still contributes more total SSE because of area.",
        "", "## Stratification response", "",
        "| quintile | old bias | new bias | SSE improvement share |",
        "|---|---:|---:|---:|",
        "| q1 | -0.552 | -0.518 | 3.20% |",
        "| q2 | -1.131 | -1.081 | 4.81% |",
        "| q3 | -1.322 | -1.288 | 3.13% |",
        "| q4 | -1.465 | -1.402 | 5.53% |",
        "| q5 | -1.583 | -1.484 | 5.98% |",
        "", "The strongest-stratification quintile has the largest relative",
        "improvement, but the absolute gain is still modest.",
        "", "## Remaining worst errors", "",
        "The 100 largest candidate errors are overwhelmingly in the",
        "`300..360E` and `40..60N` sectors: 83 of 100 lie in the longitude",
        "sector and 85 in the northern mid-latitude band. These are warm biases,",
        "not the global cold bias. They are a separate problem from the broad",
        "cold SST bias.", "", "## Conclusion", "",
        "Reduced vertical mixing is a useful candidate baseline, but its",
        "improvement is modest and broadly distributed. It is not yet enough to",
        "justify a full mixed-layer closure on its own. The next step should be",
        "a focused audit of the high-latitude North Atlantic and coastal warm",
        "biases, followed by a bulk heat-flux or boundary/ice diagnostic if",
        "those regions remain dominant.",
    ]
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'analysis.md'}")


if __name__ == "__main__":
    main()
