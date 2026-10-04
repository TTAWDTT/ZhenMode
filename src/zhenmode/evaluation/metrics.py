"""Reusable benchmark metrics for the standardized zhenmode protocol.

This module scores a saved run snapshot table plus, when available, the
optional 3D snapshots.  The goal is a stable, reportable benchmark format
rather than yet another ad-hoc analysis script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_array

from zhenmode.model.diagnostics.mixed_layer import mixed_layer_depth

R_EARTH = 6.371e6
RHO_0 = 1025.0
T_REF = 15.0
ALPHA_T = 2.0e-4
BETA_S = 7.6e-4
S_REF = 35.0
METRIC_DEFINITION = "area_weighted_angular_box_v2"


def _float_array(values):
    return np.asarray(np.ma.filled(np.ma.asarray(values, dtype=float), np.nan), dtype=float)


def _weighted_mean(values, weights):
    anchor = values.flat[0]
    return anchor + np.sum(weights * (values - anchor)) / np.sum(weights)


def smooth_2d_global(field: np.ndarray, lat: np.ndarray,
                     deg: float = 2.0, *, lon: np.ndarray,
                     ocean: np.ndarray | None = None,
                     area: np.ndarray | None = None) -> np.ndarray:
    """Wet-area angular box mean with periodic longitude and bounded latitude."""
    field = _float_array(field)
    lat, lon = _float_array(lat), _float_array(lon)
    if not np.isfinite(deg) or deg < 0.:
        raise ValueError("angular filter half-width must be finite and nonnegative")
    ocean = np.ones(field.shape, dtype=bool) if ocean is None else np.asarray(ocean, bool)
    area = cell_area(lat, lon) if area is None else _float_array(area)
    _validate_spatial_arrays(field, ocean, area, lat, lon)
    longitude_distance = np.abs((lon[:, None] - lon[None, :] + 180.) % 360. - 180.)
    longitude_window = csr_array(longitude_distance <= deg)
    latitude_window = csr_array(np.abs(lat[:, None] - lat[None, :]) <= deg)
    weights = np.where(ocean, area, 0.)
    anchor = field[ocean][0] if ocean.any() else 0.
    weighted_field = np.where(ocean, field - anchor, 0.) * weights
    numerator = latitude_window.dot(longitude_window.dot(weighted_field).T).T
    denominator = latitude_window.dot(longitude_window.dot(weights).T).T
    return anchor + np.divide(numerator, denominator, out=np.full(field.shape, np.nan),
                              where=denominator > 0.)


def cell_area(lat: np.ndarray, lon: np.ndarray,
              radius: float = R_EARTH) -> np.ndarray:
    """Spherical rectangle areas using inferred midpoint/extrapolated edges."""
    lat, lon = _float_array(lat), _float_array(lon)
    if not np.isfinite(radius) or radius <= 0.:
        raise ValueError("radius must be finite and positive")
    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 2:
        raise ValueError("lat/lon must have at least two one-dimensional points")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)) or np.any(np.abs(lat) > 90.):
        raise ValueError("lat/lon must be finite physical centers")
    latitude = np.radians(lat)
    longitude = np.unwrap(np.radians(lon))
    edges = []
    for centers in (longitude, latitude):
        differences = np.diff(centers)
        if not (np.all(differences > 0.) or np.all(differences < 0.)):
            raise ValueError("lat/lon centers must be distinct and monotonically ordered")
        edges.append(np.concatenate(([centers[0] - .5 * differences[0]],
                                     .5 * (centers[:-1] + centers[1:]),
                                     [centers[-1] + .5 * differences[-1]])))
    longitude_edges, latitude_edges = edges
    if abs(longitude_edges[-1] - longitude_edges[0]) > 2. * np.pi + 1e-12:
        raise ValueError("inferred longitude cells exceed one revolution")
    latitude_edges = np.clip(latitude_edges, -.5 * np.pi, .5 * np.pi)
    return radius ** 2 * np.abs(np.diff(longitude_edges))[:, None] * np.abs(np.diff(np.sin(latitude_edges)))[None, :]


def _validate_spatial_arrays(field, ocean, area, lat, lon):
    if field.ndim != 2 or field.shape != ocean.shape or field.shape != area.shape:
        raise ValueError("field, mask and area shapes must agree")
    if lat.ndim != 1 or lon.ndim != 1 or field.shape != (lon.size, lat.size):
        raise ValueError("field shape must match longitude/latitude centers")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)) or np.any(np.abs(lat) > 90.):
        raise ValueError("coordinates must be finite physical centers")
    if np.unique(lat).size != lat.size or np.unique(lon % 360.).size != lon.size:
        raise ValueError("coordinates must be distinct")
    if np.any(~np.isfinite(area[ocean])) or np.any(area[ocean] <= 0.):
        raise ValueError("area weights must be finite and positive on scored wet cells")


def _weighted_correlation(left, right, weights):
    left, right, weights = (np.asarray(values, dtype=float) for values in (left, right, weights))
    if left.size < 2 or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return float("nan")
    weights = weights / weights.sum()
    left = left - _weighted_mean(left, weights)
    right = right - _weighted_mean(right, weights)
    variance_left, variance_right = np.sum(weights * left ** 2), np.sum(weights * right ** 2)
    if variance_left <= 0. or variance_right <= 0.:
        return float("nan")
    return float(np.sum(weights * left * right) / np.sqrt(variance_left * variance_right))


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



def global_pattern_metrics(model_sst: np.ndarray,
                           reference_sst: np.ndarray,
                           ocean: np.ndarray,
                           lat: np.ndarray, lon: np.ndarray,
                           area: np.ndarray) -> dict:
    """Area-weighted zonal and wet-only angular-filtered SST patterns."""
    ocean = np.asarray(ocean, dtype=bool)
    zonal_m = np.array([
        _weighted_mean(model_sst[ocean[:, j], j], area[ocean[:, j], j]) if ocean[:, j].any() else np.nan
        for j in range(len(lat))])
    zonal_w = np.array([
        _weighted_mean(reference_sst[ocean[:, j], j], area[ocean[:, j], j]) if ocean[:, j].any() else np.nan
        for j in range(len(lat))])
    zonal_area = np.sum(np.where(ocean, area, 0.), axis=0)
    good = zonal_area > 0.
    a1_corr = _weighted_correlation(zonal_m[good], zonal_w[good], zonal_area[good])
    a1_rmse = float(np.sqrt(np.average((zonal_m[good] - zonal_w[good]) ** 2,
                                      weights=zonal_area[good]))) if good.any() else float("nan")
    model_sm = smooth_2d_global(model_sst, lat, lon=lon, ocean=ocean, area=area)
    reference_sm = smooth_2d_global(reference_sst, lat, lon=lon, ocean=ocean, area=area)
    a2_corr = _weighted_correlation(model_sm[ocean], reference_sm[ocean], area[ocean])
    a2_rmse = regional_error_metrics(model_sm - reference_sm, ocean, weights=area)["raw_rmse"]
    return {
        "a1_corr": a1_corr,
        "a1_rmse_c": a1_rmse,
        "a2_corr": a2_corr,
        "global_a2_rmse_c": a2_rmse,
    }

def regional_error_metrics(error: np.ndarray, mask: np.ndarray,
                           weights: np.ndarray | None = None) -> dict:
    """Weighted error statistics without silently dropping missing wet values."""
    error, mask = _float_array(error), np.asarray(mask, dtype=bool)
    weights = np.ones(error.shape) if weights is None else _float_array(weights)
    if error.shape != mask.shape or error.shape != weights.shape:
        raise ValueError("error, mask and weight shapes must agree")
    raw, weights = error[mask], weights[mask]
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.):
        raise ValueError("scored weights must be finite and positive")
    valid = raw.size > 0 and np.all(np.isfinite(raw))
    return {
        "n": int(mask.sum()),
        "weight_sum": float(weights.sum()),
        "raw_bias": float(_weighted_mean(raw, weights)) if valid else float("nan"),
        "raw_rmse": float(np.sqrt(np.average(raw ** 2, weights=weights))) if valid else float("nan"),
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






def score_snapshot(sst: np.ndarray, reference_sst: np.ndarray,
                   ocean: np.ndarray, lat: np.ndarray,
                   lon: np.ndarray, *, area: np.ndarray | None = None) -> dict:
    """Score one same-grid SST pair; reference independence is not inferred."""
    sst, reference_sst = _float_array(sst), _float_array(reference_sst)
    ocean = np.asarray(ocean, dtype=bool)
    lat, lon = _float_array(lat), _float_array(lon)
    area_source = "inferred_center_edges" if area is None else "provided_cell_area"
    area = cell_area(lat, lon) if area is None else _float_array(area)
    _validate_spatial_arrays(sst, ocean, area, lat, lon)
    if sst.shape != reference_sst.shape:
        raise ValueError("model and reference shapes must agree")
    raw_error = sst - reference_sst
    domain = hashlib.sha256()
    for values in (lat, lon, ocean.astype(float), np.where(ocean, area, 0.)):
        domain.update(np.asarray(values.shape, dtype="<i8").tobytes())
        domain.update(np.asarray(values, dtype="<f8").tobytes())
    result = {
        "metric_definition": METRIC_DEFINITION,
        "comparison_domain_sha256": domain.hexdigest(),
        "comparison_reference_sha256": hashlib.sha256(
            np.asarray(np.where(ocean, reference_sst, 0.), dtype="<f8").tobytes()).hexdigest(),
        "area_source": area_source,
        "spatial_filter": "wet_area_angular_box_half_width_2_degrees_not_constant_km",
        "reference_role": "unspecified_not_independence_certified",
        "coverage_complete": bool(ocean.any() and np.all(np.isfinite(sst[ocean]))
                                  and np.all(np.isfinite(reference_sst[ocean]))),
        "global": regional_error_metrics(raw_error, ocean, weights=area),
    }
    result.update(global_pattern_metrics(sst, reference_sst, ocean, lat, lon, area))
    for name, mask in regional_masks(lat, lon, ocean).items():
        result[name] = regional_error_metrics(raw_error, mask, weights=area)
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
    area = np.asarray(z["cell_area_m2"], dtype=float) if "cell_area_m2" in z else cell_area(lat, lon)
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
    spatial = score_snapshot(sst, reference, ocean, lat, lon,
                             area=area if "cell_area_m2" in z else None)
    result.update(spatial)
    result.update(spatial["global"])
    result["reference_role"] = "initialization_field_not_independent_validation"
    result["verdict_scope"] = "integration_watchdog_not_climate_accuracy"
    result["temporal_averaging"] = "arithmetic_saved_records_not_time_bounds_weighted"
    result["budget_drift_scope"] = "endpoint_content_change_not_budget_residual"
    result["ice"] = sea_ice_metrics(
        sst, ocean, freeze_temp=freeze_temp,
        area=area,
        ice_thickness=(np.asarray(z["ice_top"][-1], dtype=float)
                       if "ice_top" in z else None))
    if "S_init" in z:
        mld = mixed_layer_depth(np.asarray(z["T_init"], dtype=float),
                                np.asarray(z["S_init"], dtype=float),
                                np.asarray(z["z"], dtype=float),
                                ocean=ocean)
        finite = np.isfinite(mld)
        if finite.any():
            result["mld"] = {
                "definition": "density_threshold_0.03_kg_m3_ref10m", "source": "initial_T_S",
                "mean_m": float(np.mean(mld[finite])),
                "median_m": float(np.median(mld[finite])),
                "p90_m": float(np.percentile(mld[finite], 90)),
            }
    return result


def _relative_drift(series: np.ndarray) -> float:
    values = np.asarray(np.ma.filled(series, np.nan), dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)) or values[0] == 0.:
        return float("nan")
    return float(100.0 * (values[-1] - values[0]) / abs(values[0]))


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
        description="Score a saved zhenmode run with the standardized protocol.")
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
