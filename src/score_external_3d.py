"""Score an external 3D ocean field against the shared solver reference."""
from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import netCDF4
import numpy as np

from benchmark_metrics import regional_masks, regional_error_metrics
from score_external_model import _read_centers, _steady_mask, _relative_drift


def _steady_mean(field: np.ndarray, steady_days: float) -> np.ndarray:
    if field.ndim != 4:
        raise RuntimeError("expected (time,z,y,x)")
    time_var_index = next((i for i, name in enumerate(
        ("time", "Time")) if name in field.dimensions), 0)
    if field.shape[time_var_index] == 0:
        raise RuntimeError("external output has no time records")
    days = np.arange(field.shape[time_var_index], dtype=float)
    keep = _steady_mask(days, steady_days)
    arr = np.asarray(field[keep], dtype=float)
    return np.mean(arr, axis=0)


def _layer_metrics(error: np.ndarray, ocean3d: np.ndarray, z: np.ndarray) -> dict:
    out = {}
    for k, depth in enumerate(z):
        mask = ocean3d[:, :, k]
        if not mask.any():
            continue
        out[f"z{int(depth):04d}m"] = regional_error_metrics(error[:, :, k], mask)
    return out


def _vertical_mask(depth: np.ndarray, ocean2d: np.ndarray,
                   zlevels: np.ndarray) -> np.ndarray:
    if depth.shape != ocean2d.shape:
        raise RuntimeError(f"depth shape mismatch: {depth.shape} vs {ocean2d.shape}")
    return ocean2d[:, :, None] & (
        np.abs(zlevels)[None, None, :] <= depth[:, :, None])


def score_external_3d(path, *, variable, reference_path, lat_var=None,
                      lon_var=None, wet_var=None, geometry=None,
                      depth_var=None, steady_days=0.0,
                      replace_land_with_reference=True) -> dict:
    ref = np.load(reference_path, allow_pickle=True)
    if variable == "temp":
        reference = np.asarray(ref["T_init"], dtype=float)
    elif variable == "salt":
        reference = np.asarray(ref["S_init"], dtype=float)
    else:
        raise ValueError("variable must be temp or salt")
    if reference.ndim != 3:
        raise RuntimeError("reference must be (x,y,z)")
    ref_ocean = np.asarray(ref["wet_mask"], dtype=bool)
    ref_lat = np.asarray(ref["lat"], dtype=float)
    ref_lon = np.asarray(ref["lon"], dtype=float)
    z = np.asarray(ref["z"], dtype=float)

    ds = netCDF4.Dataset(path)
    geom = netCDF4.Dataset(geometry) if geometry is not None else ds
    lat = _read_centers(geom, lat_var, ref_lat, "lat")
    lon = _read_centers(geom, lon_var, ref_lon, "lon")
    if lat.shape != ref_lat.shape or lon.shape != ref_lon.shape:
        raise RuntimeError("coordinate mismatch")
    if not np.allclose(ref_lat, lat, atol=1e-6) or not np.allclose(ref_lon, lon, atol=1e-6):
        raise RuntimeError("external-model grid centers differ from reference")

    field = ds[variable]
    field3d = _steady_mean(field, steady_days)
    if field3d.shape == (reference.shape[2], reference.shape[1], reference.shape[0]):
        # MOM6 stores (z,y,x); transpose to (x,y,z).
        field3d = np.transpose(field3d, (2, 1, 0))
    if field3d.shape != reference.shape:
        raise RuntimeError(f"field shape mismatch: {field3d.shape} vs {reference.shape}")

    wet = (np.asarray(geom[wet_var][:], dtype=bool).T if wet_var is not None else ref_ocean)
    ocean2d = ref_ocean & wet
    if depth_var is not None:
        depth = np.asarray(geom[depth_var][:], dtype=float).T
        ocean3d = _vertical_mask(depth, ocean2d, z)
    elif "wet_mask_z" in ref.files:
        ocean3d = np.asarray(ref["wet_mask_z"], dtype=bool)
    else:
        ocean3d = ocean2d[:, :, None] & np.ones(
            reference.shape[2], dtype=bool)[None, None, :]
    if replace_land_with_reference:
        field3d = np.where(ocean3d, field3d, reference)

    error = field3d - reference
    result = {
        "global_3d": regional_error_metrics(error, ocean3d),
        "layers": _layer_metrics(error, ocean3d, z),
        "n_model_wet": int(wet.sum()),
        "n_reference_wet": int(ref_ocean.sum()),
        "n_scored_cells": int(ocean3d.sum()),
        "depth_mask_applied": bool(depth_var is not None or "wet_mask_z" in ref.files),
        "verdict": "PASS" if np.isfinite(error[ocean3d]).all() else "FAIL",
        "variable": variable,
        "steady_days": steady_days,
        "source_file": str(Path(path)),
        "source_geometry": str(Path(geometry)) if geometry else None,
        "depth_var": depth_var,
    }
    for name, mask2d in regional_masks(ref_lat, ref_lon, ocean2d).items():
        mask3d = mask2d[:, :, None] & ocean3d
        result[name + "_3d"] = regional_error_metrics(error, mask3d)
    return result


def main() -> None:
    p = ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--variable", required=True, choices=["temp", "salt"])
    p.add_argument("--reference-npz", required=True)
    p.add_argument("--geometry", default=None)
    p.add_argument("--wet-var", default=None)
    p.add_argument("--lat-var", default=None)
    p.add_argument("--lon-var", default=None)
    p.add_argument("--depth-var", default=None,
                   help="column depth variable on (lat,lon), e.g. D")
    p.add_argument("--steady-days", type=float, default=10.0)
    p.add_argument("--no-land-fill", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    result = score_external_3d(args.input, variable=args.variable,
                               reference_path=args.reference_npz,
                               lat_var=args.lat_var, lon_var=args.lon_var,
                               wet_var=args.wet_var, geometry=args.geometry,
                               depth_var=args.depth_var,
                               steady_days=args.steady_days,
                               replace_land_with_reference=not args.no_land_fill)
    out = Path(args.out or Path(args.input).with_name("external_3d_benchmark.json"))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
