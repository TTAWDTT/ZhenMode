"""Attribute remaining SST error after annual real-air forcing.

This diagnostic does not run the model. It analyzes the reproduced 365-day
annual real-air experiment and writes grouped tables plus a compact map figure.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(REPO / "src"))

from ocean_solver.validation.benchmarks.climatology import smooth_2d_global  # noqa: E402
from ocean_solver.config.definitions import GlobalGridConfig  # noqa: E402
from ocean_solver.io.grid import make_global_grid  # noqa: E402
from ocean_solver.forcing.wind import real_wind_forcing  # noqa: E402


def smooth_2d_masked(field: np.ndarray, wet: np.ndarray, win: int = 2) -> np.ndarray:
    """2-degree mean over wet cells only; lon periodic, lat edge-clamped."""
    f = np.asarray(field, dtype=float) * wet
    w = wet.astype(float)
    pad_lon = ((win, win), (0, 0))
    pad_lat = ((0, 0), (win, win))
    fp = np.pad(np.pad(f, pad_lon, mode="wrap"), pad_lat, mode="edge")
    wp = np.pad(np.pad(w, pad_lon, mode="constant"), pad_lat, mode="edge")
    csf = np.cumsum(np.cumsum(fp, axis=0), axis=1)
    csw = np.cumsum(np.cumsum(wp, axis=0), axis=1)
    csf = np.pad(csf, ((1, 0), (1, 0)), mode="constant")
    csw = np.pad(csw, ((1, 0), (1, 0)), mode="constant")
    n = 2 * win + 1
    out = np.full(field.shape, np.nan, dtype=float)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            i0, j0 = i, j
            i1, j1 = i + n, j + n
            sf = (csf[i1, j1] - csf[i0, j1]
                  - csf[i1, j0] + csf[i0, j0])
            sw = (csw[i1, j1] - csw[i0, j1]
                  - csw[i1, j0] + csw[i0, j0])
            if sw > 0:
                out[i, j] = sf / sw
    return out


def wind_curl(tx: np.ndarray, ty: np.ndarray, dx_2d: np.ndarray,
              dy: float) -> np.ndarray:
    """Approximate annual mean wind-stress curl on the solver grid."""
    dty_dx = (np.roll(tx, -1, axis=0) - np.roll(tx, 1, axis=0)) / (2.0 * dx_2d)
    dtx_dy = np.gradient(tx, dy, axis=1)
    return dty_dx - dtx_dy


def write_csv(path: Path, rows: list[tuple], header: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def fmt(v: float, digits: int = 3) -> str:
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def group_summary(name: str, mask: np.ndarray, err_raw: np.ndarray,
                  err_a2: np.ndarray, total_raw: float,
                  total_a2: float) -> tuple:
    if not np.any(mask):
        return (name, 0, np.nan, np.nan, 0.0, np.nan, np.nan, 0.0, np.nan)
    return (
        name,
        int(np.sum(mask)),
        float(np.mean(err_raw[mask])),
        float(np.sqrt(np.mean(err_raw[mask] ** 2))),
        float(np.sum(err_raw[mask] ** 2) / total_raw),
        float(np.mean(err_a2[mask])),
        float(np.sqrt(np.mean(err_a2[mask] ** 2))),
        float(np.sum(err_a2[mask] ** 2) / total_a2),
        float(np.max(np.abs(err_a2[mask]))),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run",
                    default="results/real_air_temp/global_real_air_2m_1deg_365d_repeat.npz")
    ap.add_argument("--out-dir",
                    default="research/experiments/coastal_vertical_attribution")
    ap.add_argument("--bathymetry",
                    default=os.environ.get("OCEAN_SOLVER_BATHYMETRY", ""))
    ap.add_argument("--steady-days", type=float, default=90.0)
    args = ap.parse_args()
    if not args.bathymetry:
        raise RuntimeError("set OCEAN_SOLVER_BATHYMETRY or pass --bathymetry")

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(REPO / args.run)
    wet = np.asarray(d["wet_mask"] > 0.5)
    lat = np.asarray(d["lat"], dtype=float)
    lon = np.asarray(d["lon"], dtype=float)
    days = np.asarray(d["days"], dtype=float)
    steady = days >= days.max() - args.steady_days
    if steady.sum() < 2:
        steady = np.ones_like(days, dtype=bool)
    model = np.mean(np.asarray(d["T_top"])[steady], axis=0)
    woa = np.asarray(d["T_init"][:, :, 0], dtype=float)
    err_raw = model - woa

    gc = GlobalGridConfig(nx=wet.shape[0], ny=wet.shape[1], lat_max=60.0)
    grid = make_global_grid(gc, args.bathymetry,
                            smooth_passes=30, min_depth=100.0)
    depth = np.asarray(grid.depth)
    dist_land = ndimage.distance_transform_edt(
        ~np.pad(~wet, ((30, 30), (0, 0)), mode="wrap")
    )[30:-30, :]
    rows_idx = np.arange(wet.shape[1], dtype=float)[None, :]
    dist_wall = np.minimum(rows_idx, wet.shape[1] - 1 - rows_idx)

    # Annual mean stress uses the exact monthly forcing snapshots in the run.
    tau_x = []
    tau_y = []
    for month in range(12):
        tx, ty = real_wind_forcing(month_idx=900 + month, grid=grid)
        tau_x.append(np.asarray(tx, dtype=float))
        tau_y.append(np.asarray(ty, dtype=float))
    tau_x = np.mean(tau_x, axis=0)
    tau_y = np.mean(tau_y, axis=0)
    wind_mag = np.sqrt(tau_x ** 2 + tau_y ** 2)
    dty_dx = ((np.roll(tau_y, -1, axis=0) - np.roll(tau_y, 1, axis=0))
              / (2.0 * grid.dx_2d))
    dtx_dy = np.gradient(tau_x, grid.dy, axis=1)
    curl = dty_dx - dtx_dy
    abs_curl = np.abs(curl)
    strat50 = np.asarray(d["T_init"][:, :, 0], dtype=float) \
        - np.asarray(d["T_init"][:, :, 4], dtype=float)

    # Official A2 error and an ocean-only smoother used as a coast artifact test.
    model_a2 = smooth_2d_global(model, lat, deg=2.0)
    woa_a2 = smooth_2d_global(woa, lat, deg=2.0)
    err_a2 = model_a2 - woa_a2
    model_masked = smooth_2d_masked(model, wet)
    woa_masked = smooth_2d_masked(woa, wet)
    err_masked = model_masked - woa_masked

    # Exclusive regions. The wall group prevents boundary cells from being
    # double counted as ordinary coast.
    regions = {
        "near_wall": wet & (dist_wall < 3.0),
        "coast_shallow": wet & (dist_wall >= 3.0) & (dist_land < 3.0) & (depth < 1000.0),
        "coast_deep": wet & (dist_wall >= 3.0) & (dist_land < 3.0) & (depth >= 1000.0),
        "open_shallow": wet & (dist_wall >= 3.0) & (dist_land >= 3.0) & (depth < 1000.0),
        "open_deep": wet & (dist_wall >= 3.0) & (dist_land >= 3.0) & (depth >= 1000.0),
    }
    total_raw = float(np.sum(err_raw[wet] ** 2))
    total_a2 = float(np.sum(err_a2[wet] ** 2))
    region_rows = []
    for name, mask in regions.items():
        row = group_summary(name, mask, err_raw, err_a2, total_raw, total_a2)
        remaining = float(np.sqrt(max(0.0, total_a2 - total_a2 * row[7]) / wet.sum()))
        region_rows.append(tuple(row) + (remaining,))
    write_csv(out / "region_shares.csv", region_rows,
              ["region", "count", "raw_mean", "raw_rmse", "raw_sse_share",
               "a2_mean", "a2_rmse", "a2_sse_share", "a2_max_abs",
               "a2_zero_rmse"])

    # Grouped diagnostics on raw and official A2 errors.
    grouped = {
        "distance": [
            ("1..2", wet & (dist_land >= 1) & (dist_land < 2)),
            ("2..3", wet & (dist_land >= 2) & (dist_land < 3)),
            ("3..5", wet & (dist_land >= 3) & (dist_land < 5)),
            ("5..10", wet & (dist_land >= 5) & (dist_land < 10)),
            ("10..20", wet & (dist_land >= 10) & (dist_land < 20)),
            (">=20", wet & (dist_land >= 20)),
        ],
        "depth": [
            ("<200", wet & (depth < 200)),
            ("200..500", wet & (depth >= 200) & (depth < 500)),
            ("500..1000", wet & (depth >= 500) & (depth < 1000)),
            ("1000..2000", wet & (depth >= 1000) & (depth < 2000)),
            ("2000..4000", wet & (depth >= 2000) & (depth < 4000)),
            (">=4000", wet & (depth >= 4000)),
        ],
        "latitude": [
            ("-60..-40", wet & (lat[None, :] >= -60) & (lat[None, :] < -40)),
            ("-40..-20", wet & (lat[None, :] >= -40) & (lat[None, :] < -20)),
            ("-20..0", wet & (lat[None, :] >= -20) & (lat[None, :] < 0)),
            ("0..20", wet & (lat[None, :] >= 0) & (lat[None, :] < 20)),
            ("20..40", wet & (lat[None, :] >= 20) & (lat[None, :] < 40)),
            ("40..60", wet & (lat[None, :] >= 40) & (lat[None, :] < 60)),
        ],
    }
    for family, bins in grouped.items():
        rows = []
        for name, mask in bins:
            rows.append(group_summary(name, mask, err_raw, err_a2,
                                      total_raw, total_a2))
        write_csv(out / f"by_{family}.csv", rows,
                  ["group", "count", "raw_mean", "raw_rmse", "raw_sse_share",
                   "a2_mean", "a2_rmse", "a2_sse_share", "a2_max_abs"])

    # Quintile groups for wind and stratification.
    for field_name, field in [("wind", wind_mag),
                              ("abs_curl", abs_curl),
                              ("strat50", strat50)]:
        qs = np.quantile(field[wet], np.linspace(0, 1, 6))
        rows = []
        for i, (lo, hi) in enumerate(zip(qs[:-1], qs[1:])):
            mask = wet & (field >= lo)
            mask &= (field < hi) if i < 4 else (field <= hi)
            rows.append(group_summary(f"q{i + 1}", mask, err_raw, err_a2,
                                      total_raw, total_a2))
        write_csv(out / f"by_{field_name}.csv", rows,
                  ["group", "count", "raw_mean", "raw_rmse", "raw_sse_share",
                   "a2_mean", "a2_rmse", "a2_sse_share", "a2_max_abs"])

    # Coast-only latitude table: separates coastal cold bias from global area.
    coast = wet & (dist_wall >= 3.0) & (dist_land < 3.0)
    coast_sse = float(np.sum(err_raw[coast] ** 2))
    rows = []
    for lo, hi in [(-60, -40), (-40, -20), (-20, 0), (0, 20), (20, 40), (40, 60)]:
        mask = coast & (lat[None, :] >= lo) & (lat[None, :] < hi)
        if not np.any(mask):
            continue
        rows.append([
            f"{lo}..{hi}", int(mask.sum()), fmt(np.mean(err_raw[mask])),
            fmt(np.sqrt(np.mean(err_raw[mask] ** 2))),
            fmt(100.0 * np.sum(err_raw[mask] ** 2) / coast_sse, 2),
            fmt(100.0 * np.sum(err_raw[mask] ** 2) / total_raw, 2),
            fmt(float(np.max(np.abs(err_raw[mask])))),
        ])
    write_csv(out / "coastal_by_latitude.csv", rows,
              ["lat_band", "count", "mean_error", "rmse",
               "coast_sse_share", "global_sse_share", "max_abs"])

    # Coast-only wind quintiles: a simple upwelling sanity check.
    qs = np.quantile(wind_mag[coast], np.linspace(0, 1, 6))
    rows = []
    for i, (lo, hi) in enumerate(zip(qs[:-1], qs[1:])):
        mask = coast & (wind_mag >= lo)
        mask &= (wind_mag < hi) if i < 4 else (wind_mag <= hi)
        rows.append([
            f"q{i + 1}", int(mask.sum()), fmt(np.mean(err_raw[mask])),
            fmt(np.sqrt(np.mean(err_raw[mask] ** 2))),
            fmt(np.sqrt(np.mean(err_a2[mask] ** 2))),
        ])
    write_csv(out / "coastal_by_wind.csv", rows,
              ["group", "count", "raw_mean", "raw_rmse", "a2_rmse"])

    # Cold/warm decomposition by physically distinct regions.
    sign_regions = {
        "all": wet,
        "coast": coast,
        "open_deep": regions["open_deep"],
        "near_wall": regions["near_wall"],
    }
    rows = []
    for name, base in sign_regions.items():
        for sign_name, mask in [
            ("cold", base & (err_raw < 0.0)),
            ("warm", base & (err_raw > 0.0)),
        ]:
            if not np.any(mask):
                continue
            rows.append([
                name, sign_name, int(mask.sum()),
                fmt(float(np.sum(mask) / max(1, np.sum(base))), 3),
                fmt(float(np.mean(err_raw[mask]))),
                fmt(float(np.sqrt(np.mean(err_raw[mask] ** 2)))),
                fmt(float(np.sum(err_raw[mask] ** 2)
                          / max(1.0, np.sum(err_raw[base] ** 2))), 3),
            ])
    write_csv(out / "cold_warm_by_region.csv", rows,
              ["region", "sign", "count", "cell_share", "mean_error",
               "rmse", "sse_share_in_region"])

    # Correlations on all wet cells and coast-only cells.
    variables = [
        ("dist_land", dist_land),
        ("depth", depth),
        ("latitude", np.broadcast_to(lat[None, :], wet.shape)),
        ("woa_sst", woa),
        ("strat50", strat50),
        ("wind_mag", wind_mag),
        ("abs_wind_curl", abs_curl),
    ]
    correlation_rows = []
    for base_name, base_mask in [("all", wet), ("coast", coast)]:
        for name, field in variables:
            x = field[base_mask]
            y = err_raw[base_mask]
            correlation_rows.append([
                base_name, name,
                fmt(float(pearsonr(x, y)[0])),
                fmt(float(spearmanr(x, y).statistic)),
            ])
    rows = correlation_rows
    write_csv(out / "correlations.csv", correlation_rows,
              ["domain", "variable", "pearson_r", "spearman_rho"])

    # Worst individual cells by raw absolute error.
    idx = np.argwhere(wet)
    vals = err_raw[wet]
    order = np.argsort(np.abs(vals))[::-1][:100]
    rows = []
    for k in order:
        i, j = idx[k]
        rows.append([
            lon[i], lat[j], depth[i, j], dist_land[i, j], err_raw[i, j],
            model[i, j], woa[i, j], wind_mag[i, j], strat50[i, j],
        ])
    write_csv(out / "worst_cells.csv", rows,
              ["lon", "lat", "depth", "dist_land", "raw_error",
               "model_sst", "woa_sst", "wind_mag", "strat50"])

    # Compact map audit.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt is not None:
        fig, ax = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
        panels = [
            (err_raw, "Raw SST error (C)", "RdBu_r", -6, 6),
            (err_a2, "Official A2 smoothed error (C)", "RdBu_r", -10, 10),
            (err_masked, "Ocean-only smoothed error (C)", "RdBu_r", -10, 10),
            (dist_land, "Distance to land (cells)", "viridis"),
            (wind_mag, "Annual wind-stress magnitude (N/m2)", "viridis"),
            (strat50, "WOA SST minus T(50m) (C)", "viridis"),
        ]
        for ax0, (field, title, cmap, *vlim) in zip(ax.flat, panels):
            kwargs = {"cmap": cmap, "shading": "auto"}
            if len(vlim) == 2:
                im = ax0.pcolormesh(lon, lat, np.where(wet, field, np.nan).T,
                                    vmin=vlim[0], vmax=vlim[1], **kwargs)
            else:
                im = ax0.pcolormesh(lon, lat, np.where(wet, field, np.nan).T,
                                    **kwargs)
            ax0.set_title(field if False else title)
            fig.colorbar(im, ax=ax0)
            ax0.set_xlabel("Longitude (E)")
            ax0.set_ylabel("Latitude (N)")
            ax0.set_ylim(lat[0], lat[-1])
        fig.savefig(out / "diagnostic_maps.png", dpi=180)
        plt.close(fig)

    # Human-readable diagnostic note.
    raw_all = group_summary("all", wet, err_raw, err_a2, total_raw, total_a2)
    coast_all = group_summary("coast", coast, err_raw, err_a2, total_raw, total_a2)
    open_deep = group_summary("open_deep", regions["open_deep"],
                              err_raw, err_a2, total_raw, total_a2)
    corr = {r[1]: (float(r[2]), float(r[3])) for r in rows if r[0] == "all"}
    coast_corr = {r[1]: (float(r[2]), float(r[3])) for r in rows if r[0] == "coast"}
    lines = [
        "# Coastal / Vertical Attribution Diagnostic", "",
        "## Bottom line", "",
        f"- Raw ocean SST RMSE: `{fmt(raw_all[3])} C`; mean bias: "
        f"`{fmt(raw_all[2])} C`.",
        f"- Official A2 RMSE: `{fmt(total_a2 and np.sqrt(total_a2 / wet.sum()))} C`.",
        f"- Deep open ocean dominates raw SSE (`{fmt(100.0 * group_summary('open_deep', regions['open_deep'], err_raw, err_a2, total_raw, total_a2)[4], 1)}%`),",
        "  simply because it contains most ocean area.",
        f"- The coast has much higher per-cell RMSE (`{fmt(group_summary('coast', coast, err_raw, err_a2, total_raw, total_a2)[3])} C`),",
        f"  but only `{fmt(100.0 * group_summary('coast', wet & (dist_wall >= 3) & (dist_land < 3), err_raw, err_a2, total_raw, total_a2)[4], 1)}%` of raw SSE.",
        "", "## Interpretation", "",
        "1. The global model is generally too cold, but the strongest individual",
        "   errors are regional warm anomalies in western-boundary-current-like",
        "   regions. This is not a single uniform cold bias.",
        "2. Closer to land is colder on average, so coastal physics matters,",
        "   but the deep open ocean still dominates total squared error.",
        "3. Stronger annual wind stress does **not** produce a colder coastal SST;",
        "   it is associated with a warmer model. Wind alone is therefore not a",
        "   clean coastal-upwelling attribution variable here.",
        "4. Stronger WOA surface-to-50m stratification coincides with a colder",
        "   model surface. This is the clearest signal pointing toward vertical",
        "   mixing / mixed-layer treatment, but it is a correlation, not proof.",
        "5. The official A2 smoother amplifies the coastal contribution relative",
        "   to raw ocean error, so mask-aware smoothing should be checked before",
        "   tuning coastal physics.",
        "", "## Next diagnostic conclusion", "",
        "The next model experiment should be a vertical-mixing sensitivity test,",
        "not another transport or scalar lambda change. Keep annual real 2m air",
        "forcing as the preferred baseline.",
    ]
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'analysis.md'}")


if __name__ == "__main__":
    main()
