"""Score an external ocean-model NetCDF on the shared benchmark grid.

The first target is MOM6, but the CLI is intentionally model-neutral so the
same command can score NEMO/ROMS-style structured output after remapping.
"""
from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import netCDF4
import numpy as np

from benchmark_metrics import score_snapshot


def _read_centers(ds: netCDF4.Dataset, name: str | None,
                  reference_centers: np.ndarray,
                  default_name: str | None = None) -> np.ndarray:
    """Read 1D or MOM6-style 2D center coordinates and validate against the reference."""
    name = name or default_name
    if not name:
        raise ValueError("latitude/longitude variable is required unless the input has standard 1D lat/lon")
    values = np.asarray(ds[name][:], dtype=float)
    if values.ndim == 1:
        centers = values
    elif values.ndim == 2:
        # MOM6 ocean_geometry.nc stores geolat/geolon on the cell-center tile. The
        # shared structured slice has latitude varying along rows and longitude
        # varying along columns, so take those vectors for validation.
        if name in {"lat", "geolat", "latitude"}:
            centers = values[:, 0]
        elif name in {"lon", "geolon", "longitude"}:
            centers = values[0, :]
        elif values.shape[1] == reference_centers.size:
            centers = values[0, :]
        else:
            centers = values[:, 0]
    else:
        centers = values
    return centers


def _steady_mask(days: np.ndarray, steady_days: float) -> np.ndarray:
    if days.size <= 1 or steady_days <= 0:
        return np.ones(days.shape, dtype=bool)
    return np.asarray(days >= days[-1] - steady_days, dtype=bool)


def _relative_drift(series: np.ndarray) -> float:
    a, b = float(np.asarray(series)[0]), float(np.asarray(series)[-1])
    return float(100.0 * (b - a) / abs(a)) if a else 0.0


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
    ref = np.load(reference_path, allow_pickle=True)
    reference = np.asarray(ref["T_init"], dtype=float)[:, :, 0]
    ref_ocean = np.asarray(ref["wet_mask"], dtype=bool)
    ref_lat = np.asarray(ref["lat"], dtype=float)
    ref_lon = np.asarray(ref["lon"], dtype=float)

    ds = netCDF4.Dataset(path)
    geometry = netCDF4.Dataset(geometry) if geometry is not None else ds

    lat = _read_centers(geometry, lat_var, ref_lat, "lat")
    lon = _read_centers(geometry, lon_var, ref_lon, "lon")
    if lat.shape != ref_lat.shape or lon.shape != ref_lon.shape:
        raise RuntimeError(f"coordinate mismatch: external {lat.shape}, reference {ref_lat.shape}")
    if not np.allclose(ref_lat, lat, atol=1e-6) or not np.allclose(ref_lon, lon, atol=1e-6):
        raise RuntimeError("external-model grid centers differ from reference")

    field = ds[variable]
    if "time" in field.dimensions and field.shape[0] == 0:
        raise RuntimeError(f"external output has no time records: {field.shape}")
    if "time" in field.dimensions:
        time_var = next((name for name in ds.variables
                         if "time" in ds[name].dimensions and ds[name].ndim == 1), None)
        days = np.asarray(ds[time_var][:], dtype=float) if time_var else np.arange(field.shape[0], dtype=float)
        keep = _steady_mask(days, steady_days)
        slices = [keep if i == 0 else slice(None) for i in range(field.ndim)]
    else:
        days = np.asarray([], dtype=float)
        keep = np.ones((1,), dtype=bool)
        slices = [slice(None)] * field.ndim
    arr = np.asarray(field[tuple(slices)], dtype=float)
    if field.ndim == 3:
        sst = np.mean(arr, axis=0)
    elif field.ndim == 4:
        sst = np.mean(arr[:, int(level), :, :], axis=0)
    elif field.ndim == 2:
        sst = arr
    else:
        raise RuntimeError(f"unsupported field rank {field.ndim}; expected 2D, 3D or (time,z,y,x)")

    if replace_land_with_reference and sst.T.shape == reference.shape:
        sst = sst.T
    elif sst.shape == reference.shape:
        pass
    else:
        raise RuntimeError(f"field shape mismatch: {sst.shape} vs reference {reference.shape}")

    wet = (np.asarray(geometry[wet_var][:], dtype=bool).T
           if wet_var is not None
           else ref_ocean)
    ocean = ref_ocean & wet
    sst = np.where(ocean, sst, reference)
    result = score_snapshot(sst, reference, ocean, ref_lat, ref_lon)
    result["verdict"] = "PASS" if np.isfinite(result["global"]["raw_rmse"]) else "FAIL"
    result["days_end"] = float(days[-1]) if days.size else None
    result["time_records"] = int(days.size)
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


