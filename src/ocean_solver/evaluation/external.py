"""Score an external ocean-model NetCDF on the shared benchmark grid.

The first target is MOM6, but the CLI is intentionally model-neutral so the
same command can score other structured output on an already shared grid.
It performs no regridding.
"""
from __future__ import annotations

import json
from argparse import ArgumentParser
from contextlib import ExitStack
from pathlib import Path

import netCDF4
import numpy as np

from ocean_solver.evaluation.metrics import _relative_drift, score_snapshot


def _read_1d(ds: netCDF4.Dataset, name: str | None, fallback: np.ndarray | None = None,
             default_name: str | None = None) -> np.ndarray:
    name = name or default_name
    if not name:
        raise ValueError("latitude/longitude variable is required unless the input has standard 1D lat/lon")
    return np.asarray(ds[name][:], dtype=float)


def _steady_mask(days: np.ndarray, steady_days: float) -> np.ndarray:
    if days.size <= 1 or steady_days <= 0:
        return np.ones(days.shape, dtype=bool)
    return np.asarray(days >= days[-1] - steady_days, dtype=bool)


def _time_days(variable) -> np.ndarray:
    units = getattr(variable, "units", None)
    if not units:
        raise ValueError("time coordinate must declare CF units")
    values = np.asarray(np.ma.filled(variable[:], np.nan), dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("time coordinate must be nonempty, finite and one-dimensional")
    calendar = getattr(variable, "calendar", "standard")
    origin = netCDF4.num2date(0.0, units=units, calendar=calendar)
    dates = netCDF4.num2date(values, units=units, calendar=calendar)
    days = np.asarray([(date - origin).total_seconds() / 86400.0 for date in dates])
    if np.any(np.diff(days) <= 0):
        raise ValueError("time coordinate must be strictly increasing")
    return days


def score_external_field(path: str | Path, *, variable: str,
                         reference_path: str | Path, lat_var: str | None = None,
                         lon_var: str | None = None, wet_var: str | None = None,
                         geometry: str | Path | None = None,
                         level: int = 0, steady_days: float = 0.0,
                         replace_land_with_reference: bool = True) -> dict:
    """Return standardized SST metrics for an external structured NetCDF file.

    The function intentionally does not know model-specific names.  The caller
    supplies the variable and mask names so the same code can score MOM6, a
    NEMO-style file, or an interpolated external-model NPZ-like slice.
    """
    with np.load(reference_path, allow_pickle=True) as ref:
        reference = np.asarray(ref["T_init"], dtype=float)[:, :, 0]
        ref_ocean = np.asarray(ref["wet_mask"], dtype=bool)
        ref_lat = np.asarray(ref["lat"], dtype=float)
        ref_lon = np.asarray(ref["lon"], dtype=float)
        ref_area = np.asarray(ref["cell_area_m2"], dtype=float) if "cell_area_m2" in ref else None
    with ExitStack() as stack:
        dataset = stack.enter_context(netCDF4.Dataset(path))
        geometry_dataset = (stack.enter_context(netCDF4.Dataset(geometry))
                            if geometry is not None else dataset)
        lat = _read_1d(geometry_dataset, lat_var, ref_lat, "lat")
        lon = _read_1d(geometry_dataset, lon_var, ref_lon, "lon")
        if lat.shape != ref_lat.shape or lon.shape != ref_lon.shape:
            raise RuntimeError(f"coordinate mismatch: external {lat.shape}, reference {ref_lat.shape}")
        if not np.allclose(ref_lat, lat, atol=1e-6) or not np.allclose(ref_lon, lon, atol=1e-6):
            raise RuntimeError("external-model grid centers differ from reference")
        field = dataset[variable]
        if "time" in field.dimensions:
            if field.dimensions[0] != "time" or "time" not in dataset.variables:
                raise ValueError("expected a leading time dimension with a time coordinate")
            days = _time_days(dataset["time"])
            keep = _steady_mask(days, steady_days)
            slices = (keep,) + (slice(None),) * (field.ndim - 1)
        else:
            days = np.asarray([], dtype=float)
            keep = np.ones((1,), dtype=bool)
            slices = (slice(None),) * field.ndim
        arr = np.asarray(np.ma.filled(field[slices], np.nan), dtype=float)
        if field.ndim == 3 and "time" in field.dimensions:
            sst = np.mean(arr, axis=0)
        elif field.ndim == 4 and "time" in field.dimensions:
            sst = np.mean(arr[:, int(level), :, :], axis=0)
        elif field.ndim == 2:
            sst = arr
        else:
            raise RuntimeError(f"unsupported field dimensions {field.dimensions}")
        wet = (np.asarray(geometry_dataset[wet_var][:], dtype=bool).T
               if wet_var is not None else ref_ocean)
    if sst.T.shape == reference.shape:
        sst = sst.T
    elif sst.shape == reference.shape:
        pass
    else:
        raise RuntimeError(f"field shape mismatch: {sst.shape} vs reference {reference.shape}")

    if wet.shape != ref_ocean.shape:
        raise RuntimeError("external-model wet mask shape differs from reference")
    ocean = ref_ocean & wet
    if replace_land_with_reference:
        sst = np.where(ocean, sst, reference)
    result = score_snapshot(sst, reference, ocean, ref_lat, ref_lon, area=ref_area)
    finite_coverage = result["coverage_complete"]
    reference_count = int(ref_ocean.sum())
    scored_count = int(ocean.sum())
    result["coverage_fraction"] = scored_count / reference_count if reference_count else 0.0
    result["coverage_complete"] = bool(finite_coverage and scored_count == reference_count and reference_count > 0)
    result["reference_role"] = "shared_initialization_field_not_independent_validation"
    result["temporal_averaging"] = "arithmetic_saved_records_not_time_bounds_weighted"
    result["budget_drift_scope"] = "endpoint_content_change_not_budget_residual"
    result["verdict"] = ("PASS" if result["coverage_complete"]
                         and np.isfinite(result["global"]["raw_rmse"]) else "FAIL")
    result["verdict_scope"] = "finite_sst_and_complete_reference_coverage"
    result["days_end"] = float(days[-1]) if days.size else None
    result["time_records"] = int(days.size)
    result["time_first"] = float(days[0]) if days.size else None
    result["time_last"] = float(days[-1]) if days.size else None
    result["steady_window_days"] = ([float(days[keep][0]), float(days[-1])]
                                    if days.size > 1 else [None, None])
    result["n_model_wet"] = int(wet.sum())
    result["n_reference_wet"] = int(ref_ocean.sum())
    result["n_scored"] = int(ocean.sum())
    return result


def main():
    p = ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--variable", required=True)
    p.add_argument("--reference-npz", required=True)
    p.add_argument("--geometry", default=None)
    p.add_argument("--wet-var", default=None)
    p.add_argument("--lat-var", default=None)
    p.add_argument("--lon-var", default=None)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--steady-days", type=float, default=0.0)
    p.add_argument("--no-land-fill", action="store_true",
                   help="do not replace land with reference value")
    p.add_argument("--model", default="external_model")
    p.add_argument("--run-id", default="")
    p.add_argument("--stats", default=None)
    p.add_argument("--heat-variable", default="Heat")
    p.add_argument("--salt-variable", default="Salt")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    result = score_external_field(args.input, variable=args.variable,
                                  reference_path=args.reference_npz,
                                  lat_var=args.lat_var, lon_var=args.lon_var,
                                  wet_var=args.wet_var, geometry=args.geometry,
                                  level=args.level,
                                  replace_land_with_reference=not args.no_land_fill,
                                  steady_days=args.steady_days)
    if args.stats:
        with netCDF4.Dataset(args.stats) as ds:
            result["heat_drift_percent"] = _relative_drift(np.asarray(ds[args.heat_variable][:], dtype=float))
            result["salt_drift_percent"] = _relative_drift(np.asarray(ds[args.salt_variable][:], dtype=float))
    result["model"] = args.model
    result["run_id"] = args.run_id
    result["source_file"] = str(Path(args.input))
    out = Path(args.out or Path(args.input).with_name("external_benchmark.json"))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))




if __name__ == "__main__":
    main()
