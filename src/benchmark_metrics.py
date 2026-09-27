"""Reusable benchmark metrics for the standardized ocean_solver protocol.

This module scores a saved run snapshot table plus, when available, the
optional 3D snapshots.  The goal is a stable, reportable benchmark format
rather than yet another ad-hoc analysis script.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

R_EARTH = 6.371e6
RHO_0 = 1025.0
T_REF = 15.0
ALPHA_T = 2.0e-4
BETA_S = 7.6e-4
S_REF = 35.0


def smooth_2d_global(field: np.ndarray, lat: np.ndarray,
                     deg: float = 2.0) -> np.ndarray:
    """Smooth a global (nx, ny) field with a periodic zonal box mean.

    This matches the definition used by ``bench_climatology_global.py`` so old
    reports and the standardized benchmark remain comparable.
    """
    f = np.asarray(field, dtype=float)
    nx, ny = f.shape
    win = max(1, int(round(deg)))
    padded = np.empty((nx + 2 * win, ny + 2 * win), dtype=float)
    padded[win:win + nx, win:win + ny] = f
    padded[:win, win:win + ny] = f[-win:, :]
    padded[win + nx:, win:win + ny] = f[:win, :]
    padded[:, :win] = padded[:, win:win + 1]
    padded[:, win + ny:] = padded[:, win + ny - 1:win + ny]
    csum = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)))
    w2 = 2 * win + 1
    out = np.empty_like(f)
    for i in range(nx):
        for j in range(ny):
            out[i, j] = (csum[i + w2, j + w2]
                         - csum[i, j + w2]
                         - csum[i + w2, j]
                         + csum[i, j]) / (w2 * w2)
    return out


def cell_area(lat: np.ndarray, lon: np.ndarray,
              radius: float = R_EARTH) -> np.ndarray:
    """Area of regular lat-lon cells, shape (nx, ny), in square metres."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.size < 2 or lon.size < 2:
        raise ValueError("lat/lon must have at least two points")
    dlat = float(np.mean(np.abs(np.diff(lat))))
    dlon = float(np.mean(np.abs(np.diff(lon))))
    area = (radius * radius * np.radians(dlon) * np.radians(dlat) *
            np.cos(np.radians(lat))[None, :])
    return np.broadcast_to(area, (lon.size, lat.size)).copy()


def regional_masks(lat: np.ndarray, lon: np.ndarray,
                   ocean: np.ndarray) -> dict[str, np.ndarray]:
    """Pre-registered North Atlantic and near-wall masks."""
    lon2, lat2 = np.meshgrid(np.asarray(lon), np.asarray(lat), indexing="ij")
    return {
        "north_atlantic_40_60": ocean
        & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 40.0) & (lat2 <= 60.0),
        "near_wall_55_60": ocean
        & (lon2 >= 300.0) & (lon2 < 360.0) & (lat2 >= 55.0) & (lat2 <= 60.0),
    }





def latitude_depth_metrics(error: np.ndarray,
                           ocean3d: np.ndarray,
                           lat: np.ndarray,
                           z: np.ndarray,
                           bands: list[tuple[float, float]] | None = None) -> dict:
    """Standardized latitude-band x depth bias/RMSE for 3D snapshots.

    Bands are inclusive at the lower edge and exclusive at the upper edge,
    except that the final band includes its upper edge.  Depth keys use positive
    metres below the surface and are useful for diagnosing upper-ocean
    ventilation errors.
    """
    if bands is None:
        bands = [(-60.0, -40.0), (-40.0, -20.0), (-20.0, 0.0),
                 (0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
    error = np.asarray(error, dtype=float)
    ocean3d = np.asarray(ocean3d, dtype=bool)
    lat = np.asarray(lat, dtype=float)
    z = np.asarray(z, dtype=float)
    out: dict[str, dict[str, dict]] = {}
    for lo, hi in bands:
        mask_lat = ((lat >= lo) & (lat < hi)) if hi < bands[-1][1] else ((lat >= lo) & (lat <= hi))
        band_key = f"{int(lo)}_{int(hi)}"
        band: dict[str, dict] = {}
        for k, depth in enumerate(z):
            mask = ocean3d[:, :, k] & mask_lat[None, :]
            if not mask.any():
                continue
            band[f"z{int(-float(depth)):04d}m"] = regional_error_metrics(
                error[:, :, k], mask)
        if band:
            out[band_key] = band
    return out



def mixed_layer_depth_metrics(model_mld: np.ndarray,
                              reference_mld: np.ndarray,
                              ocean: np.ndarray | None = None) -> dict:
    """Standardized mixed-layer-depth bias/RMSE on the same wet cells."""
    model_mld = np.asarray(model_mld, dtype=float)
    reference_mld = np.asarray(reference_mld, dtype=float)
    wet = np.isfinite(model_mld) & np.isfinite(reference_mld)
    if ocean is not None:
        wet &= np.asarray(ocean, dtype=bool)
    if not wet.any():
        return {"n": 0, "raw_bias_m": float("nan"), "raw_rmse_m": float("nan")}
    diff = model_mld[wet] - reference_mld[wet]
    return {
        "n": int(wet.sum()),
        "raw_bias_m": float(diff.mean()),
        "raw_rmse_m": float(np.sqrt(np.mean(diff ** 2))),
        "mean_model_m": float(model_mld[wet].mean()),
        "mean_reference_m": float(reference_mld[wet].mean()),
    }

def latitude_band_mld_metrics(model_mld: np.ndarray,
                              reference_mld: np.ndarray,
                              lat: np.ndarray,
                              ocean: np.ndarray,
                              bands: list[tuple[float, float]] | None = None) -> dict:
    """Standardized latitude-band mixed-layer-depth bias/RMSE.

    This complements the global MLD diagnostic by isolating high-latitude
    overdeepening from subtropical shallowing.
    """
    if bands is None:
        bands = [(-60.0, -40.0), (-40.0, -20.0), (-20.0, 0.0),
                 (0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
    lat = np.asarray(lat, dtype=float)
    ocean = np.asarray(ocean, dtype=bool)
    out: dict[str, dict] = {}
    for lo, hi in bands:
        inclusive = (hi >= bands[-1][1])
        mask = ocean & ((lat >= lo) & (lat <= hi if inclusive else lat < hi))[None, :]
        if not mask.any():
            continue
        out[f"{int(lo)}_{int(hi)}"] = mixed_layer_depth_metrics(
            model_mld, reference_mld, ocean=mask)
    return out

def global_pattern_metrics(model_sst: np.ndarray,
                           reference_sst: np.ndarray,
                           ocean: np.ndarray,
                           lat: np.ndarray) -> dict:
    """Pre-registered A1/A2 large-scale SST metrics."""
    ocean = np.asarray(ocean, dtype=bool)
    zonal_m = np.array([
        model_sst[ocean[:, j], j].mean() if ocean[:, j].any() else np.nan
        for j in range(len(lat))])
    zonal_w = np.array([
        reference_sst[ocean[:, j], j].mean() if ocean[:, j].any() else np.nan
        for j in range(len(lat))])
    good = np.isfinite(zonal_m) & np.isfinite(zonal_w)
    a1_corr = (float(np.corrcoef(zonal_m[good], zonal_w[good])[0, 1])
               if good.sum() >= 2 else float("nan"))
    a1_rmse = float(np.sqrt(np.mean(
        (zonal_m[good] - zonal_w[good]) ** 2))) if good.any() else float("nan")
    model_sm = smooth_2d_global(model_sst, lat, deg=2.0)
    reference_sm = smooth_2d_global(reference_sst, lat, deg=2.0)
    a_model = model_sm[ocean] - model_sm[ocean].mean()
    a_ref = reference_sm[ocean] - reference_sm[ocean].mean()
    a2_corr = (float(np.corrcoef(a_model, a_ref)[0, 1])
               if np.std(a_model) > 0 else float("nan"))
    a2_rmse = float(np.sqrt(np.mean((model_sm[ocean] - reference_sm[ocean]) ** 2)))
    return {
        "a1_corr": a1_corr,
        "a1_rmse_c": a1_rmse,
        "a2_corr": a2_corr,
        "global_a2_rmse_c": a2_rmse,
    }

def regional_error_metrics(error: np.ndarray, mask: np.ndarray) -> dict:
    """Raw regional SST error statistics."""
    raw = np.asarray(error, dtype=float)[mask]
    return {
        "n": int(mask.sum()),
        "raw_bias": float(raw.mean()) if raw.size else float("nan"),
        "raw_rmse": float(np.sqrt(np.mean(raw ** 2))) if raw.size else float("nan"),
    }


def sea_ice_metrics(sst: np.ndarray, ocean: np.ndarray,
                    freeze_temp: float = -1.8,
                    area: np.ndarray | None = None,
                    ice_thickness: np.ndarray | None = None) -> dict:
    """Sea-ice diagnostics.

    With ``ice_thickness`` the mask is the model's explicit ice state.  Without
    it, this falls back to the legacy freezing-SST proxy so older runs remain
    directly comparable.
    """
    wet = np.asarray(ocean, dtype=bool)
    if ice_thickness is not None:
        thickness = np.asarray(ice_thickness, dtype=float)
        frozen = wet & np.isfinite(thickness) & (thickness > 0.0)
        source = "explicit_ice_thickness"
    else:
        thickness = None
        frozen = wet & np.isfinite(sst) & (np.asarray(sst, dtype=float) <= freeze_temp)
        source = "freezing_sst_proxy"
    n = int(frozen.sum())
    result = {
        "source": source,
        "freeze_temp_c": float(freeze_temp),
        "n_cells": n,
        "fraction": float(n / max(wet.sum(), 1)),
    }
    if thickness is not None:
        if n:
            result["mean_thickness_m"] = float(np.mean(thickness[frozen]))
            result["max_thickness_m"] = float(np.max(thickness[frozen]))
        else:
            result["mean_thickness_m"] = 0.0
            result["max_thickness_m"] = 0.0
    if area is not None:
        area = np.asarray(area, dtype=float)
        total = float(area[wet].sum())
        frozen_area = float(area[frozen].sum()) if n else 0.0
        result["extent_m2"] = frozen_area
        result["extent_fraction"] = frozen_area / total if total else 0.0
    return result


def seawater_density(T: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Linear-equation-of-state density used by the current solver."""
    return RHO_0 * (1.0 - ALPHA_T * (np.asarray(T, dtype=float) - T_REF)
                    + BETA_S * (np.asarray(S, dtype=float) - S_REF))


def mixed_layer_depth(T: np.ndarray, S: np.ndarray, z: np.ndarray,
                      ocean: np.ndarray | None = None,
                      ref_depth: float = 10.0,
                      density_delta: float = 0.03) -> np.ndarray:
    """Column mixed-layer depth in metres positive down.

    Uses the common density-threshold definition and the current solver's
    linear equation of state.  This is the smallest defensible MLD diagnostic:
    independent of a particular mixing parameterization and applicable to WOA,
    a saved 3D snapshot, or a model restart.
    """
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)
    depth = -np.asarray(z, dtype=float)
    rho = RHO_0 * (1.0 - ALPHA_T * (T - T_REF)
                   + BETA_S * (S - S_REF))
    nx, ny, _ = rho.shape
    wet = np.ones((nx, ny), dtype=bool) if ocean is None \
        else np.asarray(ocean, dtype=bool)
    ref_k = int(np.argmin(np.abs(depth - ref_depth)))
    mld = np.full((nx, ny), np.nan, dtype=float)
    for i in range(nx):
        for j in range(ny):
            if not wet[i, j]:
                continue
            valid = np.isfinite(rho[i, j]) & np.isfinite(S[i, j])
            if not valid.any():
                continue
            delta = rho[i, j] - rho[i, j, ref_k]
            below = np.where(valid & (depth >= ref_depth)
                             & (delta >= density_delta))[0]
            if not below.size:
                wet_k = np.where(valid)[0]
                mld[i, j] = float(depth[wet_k[-1]]) if wet_k.size else np.nan
            else:
                mld[i, j] = float(depth[int(below[0])])
    return mld


def score_snapshot(sst: np.ndarray, reference_sst: np.ndarray,
                   ocean: np.ndarray, lat: np.ndarray,
                   lon: np.ndarray) -> dict:
    """Score one 2D SST field against the same-grid WOA reference."""
    ocean = np.asarray(ocean, dtype=bool)
    raw_error = (np.asarray(sst, dtype=float)
                 - np.asarray(reference_sst, dtype=float))
    result = {
        "global": {
            "n": int(ocean.sum()),
            "raw_bias": float(np.mean(raw_error[ocean])),
            "raw_rmse": float(np.sqrt(np.mean(raw_error[ocean] ** 2))),
        }
    }
    result.update(global_pattern_metrics(sst, reference_sst, ocean, lat))
    for name, mask in regional_masks(lat, lon, ocean).items():
        result[name] = regional_error_metrics(raw_error, mask)
    return result


def score_npz(path: str | os.PathLike,
              steady_days: float = 90.0,
              freeze_temp: float = -1.8) -> dict:
    """Score a saved run without requiring external NetCDF input."""
    z = np.load(path, allow_pickle=True)
    days = np.asarray(z["days"], dtype=float)
    steady = days >= days[-1] - steady_days
    if not steady.any():
        steady = np.ones_like(days, dtype=bool)
    sst = np.mean(np.asarray(z["T_top"], dtype=float)[steady], axis=0)
    reference = np.asarray(z["T_init"], dtype=float)[:, :, 0]
    ocean = np.asarray(z["wet_mask"], dtype=bool)
    lat = np.asarray(z["lat"], dtype=float)
    lon = np.asarray(z["lon"], dtype=float)
    raw_error = sst - reference
    result = {
        "path": str(Path(path)),
        "verdict": str(z["verdict"]),
        "days_end": float(days[-1]),
        "steady_window_days": [float(days[steady][0]), float(days[-1])],
        "max_u_peak": float(z["max_u_peak"]),
        "max_eta_last": float(z["max_eta"][-1]),
        "heat_drift_percent": _relative_drift(z["heat_content_J"]),
        "salt_drift_percent": _relative_drift(z["salt_content_kg"]),
    }
    result.update(global_pattern_metrics(sst, reference, ocean, lat))
    result.update(regional_error_metrics(raw_error, ocean))
    for name, mask in regional_masks(lat, lon, ocean).items():
        result[name] = regional_error_metrics(raw_error, mask)
    result["ice"] = sea_ice_metrics(
        sst, ocean, freeze_temp=freeze_temp,
        area=cell_area(lat, lon),
        ice_thickness=(np.asarray(z["ice_top"][-1], dtype=float)
                       if "ice_top" in z else None))
    if "mixed_layer_depth_applied" in z:
        mld = np.asarray(z["mixed_layer_depth_applied"], dtype=float)
        mld_source = "solver_effective"
    elif "S_init" in z:
        mld = mixed_layer_depth(np.asarray(z["T_init"], dtype=float),
                                np.asarray(z["S_init"], dtype=float),
                                np.asarray(z["z"], dtype=float),
                                ocean=ocean)
        mld_source = "initial_T_S"
    else:
        mld = None
        mld_source = "unavailable"
    if mld is not None:
        finite = np.isfinite(mld) & ocean
        if finite.any():
            result["mld"] = {
                "definition": "solver_effective_or_density_threshold_0.03_kg_m3_ref10m",
                "source": mld_source,
                "mean_m": float(np.mean(mld[finite])),
                "median_m": float(np.median(mld[finite])),
                "p90_m": float(np.percentile(mld[finite], 90)),
            }
    return result


def _relative_drift(series: np.ndarray) -> float:
    series = np.asarray(series, dtype=float)
    a = float(series[0])
    b = float(series[-1])
    return float(100.0 * (b - a) / abs(a)) if a else 0.0


def score_3d_snapshot(snapshot: np.ndarray, z: np.ndarray,
                      ocean: np.ndarray) -> dict:
    """Convenience wrapper for one optional 3D snapshot file."""
    if snapshot.ndim != 4 or snapshot.shape[0] != 4:
        raise ValueError("expected a 4-field solver snapshot [T,u,v,S]")
    T = np.asarray(snapshot[0], dtype=float)
    S = np.asarray(snapshot[3], dtype=float)
    mld = mixed_layer_depth(T, S, z, ocean=ocean)
    finite = np.isfinite(mld)
    if not finite.any():
        return {"n": 0}
    return {
        "n": int(finite.sum()),
        "mean_mld_m": float(np.mean(mld[finite])),
        "median_mld_m": float(np.median(mld[finite])),
        "p90_mld_m": float(np.percentile(mld[finite], 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a saved ocean_solver run with the standardized protocol.")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", default=None,
                        help="output JSON path; default is next to the npz")
    parser.add_argument("--steady-days", type=float, default=90.0)
    args = parser.parse_args()
    result = score_npz(args.npz, steady_days=args.steady_days)
    out = args.out or str(
        Path(args.npz).with_name(Path(args.npz).stem + "_benchmark.json"))
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

