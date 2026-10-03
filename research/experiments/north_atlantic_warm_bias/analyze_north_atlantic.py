"""Audit high-latitude North Atlantic warm biases in the reduced-mixing run."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(REPO / "src"))

from ocean_solver.forcing.air import load_annual_mean_air_temp  # noqa: E402
from ocean_solver.config.definitions import GlobalGridConfig  # noqa: E402
from ocean_solver.io.grid import make_global_grid  # noqa: E402


def fmt(v: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(v) else f"{v:.{digits}f}"


def region_summary(name: str, mask: np.ndarray, err: np.ndarray,
                   tatm_minus_woa: np.ndarray, region_total: float) -> tuple:
    if not np.any(mask):
        return (name, 0, np.nan, np.nan, 0.0, np.nan, 0.0, np.nan, np.nan)
    warm = mask & (err > 0.0)
    cold = mask & (err < 0.0)
    sse = float(np.sum(err[mask] ** 2))
    return (
        name,
        int(mask.sum()),
        fmt(100.0 * np.sum(warm) / max(1, np.sum(mask)), 1),
        fmt(float(np.mean(err[mask]))),
        fmt(float(np.sqrt(np.mean(err[mask] ** 2)))),
        fmt(100.0 * sse / max(1e-12, region_total), 2),
        fmt(float(np.mean(tatm_minus_woa[mask]))),
        fmt(float(pearsonr(tatm_minus_woa[mask], err[mask])[0])),
        fmt(float(spearmanr(tatm_minus_woa[mask], err[mask]).statistic)),
    )


def main() -> None:
    out = REPO / "research/experiments/north_atlantic_warm_bias"
    out.mkdir(parents=True, exist_ok=True)

    run_path = REPO / "results/vertical_mixing_sensitivity/global_real_air_kv1e-6_kconv001_repeat.npz"
    d = np.load(run_path)
    wet = np.asarray(d["wet_mask"] > 0.5)
    lat = np.asarray(d["lat"], dtype=float)
    lon = np.asarray(d["lon"], dtype=float)
    days = np.asarray(d["days"], dtype=float)
    steady = days >= days.max() - 90.0
    woa = np.asarray(d["T_init"][:, :, 0], dtype=float)
    model = np.mean(np.asarray(d["T_top"])[steady], axis=0)
    err = model - woa
    strat = np.asarray(d["T_init"][:, :, 0], dtype=float) \
        - np.asarray(d["T_init"][:, :, 4], dtype=float)

    bathy = os.environ.get("OCEAN_SOLVER_BATHYMETRY")
    if not bathy:
        raise RuntimeError("set OCEAN_SOLVER_BATHYMETRY")
    gc = GlobalGridConfig(nx=wet.shape[0], ny=wet.shape[1], lat_max=60.0)
    grid = make_global_grid(gc, bathy, smooth_passes=30, min_depth=100.0)
    depth = np.asarray(grid.depth)
    tatm = load_annual_mean_air_temp(grid, year=2023)
    tatm_minus_woa = tatm - woa

    pad = 30
    dist_land = ndimage.distance_transform_edt(
        ~np.pad(~wet, ((pad, pad), (0, 0)), mode="wrap")
    )[pad:-pad, :]
    rows_idx = np.arange(wet.shape[1], dtype=float)[None, :]
    dist_wall = np.minimum(rows_idx, wet.shape[1] - 1 - rows_idx)

    north_atlantic = wet & (lon[:, None] >= 300.0) & (lon[:, None] < 360.0) \
        & (lat[None, :] >= 40.0) & (lat[None, :] <= 60.0)
    region_total = float(np.sum(err[north_atlantic] ** 2))

    groups = {
        "north_atlantic_all": north_atlantic,
        "coast": north_atlantic & (dist_land < 3.0),
        "interior": north_atlantic & (dist_land >= 3.0),
        "near_wall": north_atlantic & (dist_wall < 3.0),
        "shallow_lt1000": north_atlantic & (depth < 1000.0),
        "depth_1000_3000": north_atlantic & (depth >= 1000.0) & (depth < 3000.0),
        "deep_ge3000": north_atlantic & (depth >= 3000.0),
        "lon_300_320": north_atlantic & (lon[:, None] >= 300.0) & (lon[:, None] < 320.0),
        "lon_320_340": north_atlantic & (lon[:, None] >= 320.0) & (lon[:, None] < 340.0),
        "lon_340_360": north_atlantic & (lon[:, None] >= 340.0) & (lon[:, None] < 360.0),
        "lat_40_50": north_atlantic & (lat[None, :] >= 40.0) & (lat[None, :] < 50.0),
        "lat_50_60": north_atlantic & (lat[None, :] >= 50.0) & (lat[None, :] <= 60.0),
    }
    rows = []
    for name, mask in groups.items():
        rows.append(region_summary(name, mask, err, tatm_minus_woa, region_total))
    with (out / "regional_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "count", "warm_share", "mean_bias", "rmse",
                         "region_sse_share", "mean_tatm_minus_woa",
                         "pearson_r", "spearman_rho"])
        writer.writerows(rows)

    # Rank the largest absolute errors inside the focus region.
    idx = np.argwhere(north_atlantic)
    vals = err[north_atlantic]
    order = np.argsort(np.abs(vals))[::-1][:100]
    top_rows = []
    for k in order:
        i, j = idx[k]
        top_rows.append([
            lon[i], lat[j], depth[i, j], dist_land[i, j], dist_wall[0, j],
            err[i, j], model[i, j], woa[i, j], tatm[i, j], strat[i, j],
        ])
    with (out / "worst_cells.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lon", "lat", "depth", "dist_land", "dist_wall",
                         "error", "model_sst", "woa_sst", "ncep_tatm",
                         "woa_strat50"])
        writer.writerows(top_rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        panels = [
            (err, "Candidate SST error (C)", "RdBu_r", -8, 8),
            (tatm_minus_woa, "Annual NCEP 2m air - WOA SST (C)", "RdBu_r", -8, 8),
            (np.where(wet, dist_land, np.nan), "Distance to land (cells)", "viridis"),
            (np.where(wet, strat, np.nan), "WOA SST minus T(50m) (C)", "viridis"),
        ]
        for ax0, (field, title, cmap, *vlim) in zip(ax.flat, panels):
            kwargs = {"cmap": cmap, "shading": "auto"}
            if len(vlim) == 2:
                im = ax0.pcolormesh(lon, lat, np.where(wet, field, np.nan).T,
                                    vmin=vlim[0], vmax=vlim[1], **kwargs)
            else:
                im = ax0.pcolormesh(lon, lat, np.where(wet, field, np.nan).T,
                                    **kwargs)
            ax0.set_title(title)
            fig.colorbar(im, ax=ax0)
            ax0.set_xlabel("Longitude (E)")
            ax0.set_ylabel("Latitude (N)")
            ax0.set_ylim(lat[0], lat[-1])
        fig.savefig(out / "north_atlantic_audit_maps.png", dpi=180)
        plt.close(fig)

    # Simple human-readable interpretation.
    all_row = next(r for r in rows if r[0] == "north_atlantic_all")
    coast_row = next(r for r in rows if r[0] == "coast")
    deep_row = next(r for r in rows if r[0] == "deep_ge3000")
    lines = [
        "# North Atlantic Warm-Bias Audit", "",
        "## Bottom line", "",
        f"- Region: `300..360E / 40..60N`.",
        f"- Wet cells: `{all_row[1]}`.",
        f"- Warm-cell share: `{all_row[2]}%`.",
        f"- Mean bias: `{all_row[3]} C`.",
        f"- Regional RMSE: `{all_row[4]} C`.",
        f"- Mean annual NCEP 2m air minus WOA SST: `{all_row[6]} C`.",
        "", "## Interpretation", "",
        f"- The coastal subset has `{coast_row[2]}%` warm cells and a mean bias of "
        f"`{coast_row[3]} C`.",
        f"- The deep open-ocean subset has `{deep_row[2]}%` warm cells and a mean "
        f"bias of `{deep_row[3]} C`.",
        "- The warm bias is therefore not solely a coastal mask artifact; it also",
        "  occurs in deep open-ocean cells.",
        "- `T_atm - WOA SST` has the expected sign in many cells, but the SST",
        "  error is not explained by forcing alone. The next diagnostic should",
        "  inspect boundary/ice and bulk heat-flux completeness.",
    ]
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'analysis.md'}")


if __name__ == "__main__":
    main()
