"""Original JRA window -> explicit rectilinear native input and mapping receipts.

Atmospheric state is bilinear, surface cell means conservative, and terrestrial
discharge routes to declared coastal cells with its original grid-cell area.
This prepares data only; it never certifies an ocean or climate experiment.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import netCDF4
import numpy as np
from scipy.spatial import cKDTree

from zhenmode.model.inputs.forcing.jra55 import (
    FIELDS,
    TIME_UNITS,
    JRA55Forcing,
    _seconds,
    bilinear_rectilinear_weights,
    conservative_rectilinear_weights,
    remap_rectilinear_means,
)
from zhenmode.provenance.sources import load_json, sha256_file

AREA_SOURCE = "https://github.com/HiroyukiTsujino/JRA55-do/blob/30c8e1a84386c1db8a436d980826b884b2075089/doc/user_manual/User_manual_jra55_do_v1_5.tex"


def coastal_routing(source_lon, source_lat, lon, lat, wet, *, maximum_distance_m,
                    land_fraction=None):
    """Frozen nearest-coast routing in spherical distance, not weather infilling.

    Every original discharge cell has one recipient. Routing weights act on
    kg/s, then divide by actual target wet-cell area. No discharge is dropped.
    Longitude is periodic; the polar boundary is not classified as land.
    """
    wet = np.asarray(wet)
    if (
        wet.shape != (len(lat), len(lon))
        or not np.isin(wet, [0, 1]).all()
        or isinstance(maximum_distance_m, bool)
        or not np.isfinite(maximum_distance_m)
        or maximum_distance_m <= 0
    ):
        raise ValueError("invalid routing mask or maximum spherical distance")
    wet = wet.astype(bool)
    land_adjacent = ~np.roll(wet, 1, axis=1) | ~np.roll(wet, -1, axis=1)
    land_adjacent[1:] |= ~wet[:-1]
    land_adjacent[:-1] |= ~wet[1:]
    if land_fraction is not None:
        fraction = np.asarray(land_fraction)
        if (fraction.shape != wet.shape or not np.isfinite(fraction).all()
                or np.any(fraction < 0) or np.any(fraction > 1)):
            raise ValueError('coastal land fraction must explicitly lie in [0,1]')
        # Unresolved islands may share a wet cell; do not send their rivers
        # thousands of kilometres to the next fully dry model cell.
        land_adjacent |= fraction > 0
    coast = np.flatnonzero(wet & land_adjacent)
    if not len(coast):
        raise ValueError("no declared wet coastal recipient for terrestrial discharge")

    def xyz(x, y):
        x, y = np.meshgrid(np.deg2rad(x), np.deg2rad(y))
        return np.stack((np.cos(y) * np.cos(x), np.cos(y) * np.sin(x), np.sin(y)), axis=-1).reshape(
            -1, 3
        )

    distance, recipient = cKDTree(xyz(lon, lat)[coast]).query(
        xyz(source_lon, source_lat), workers=1
    )
    return (
        coast[recipient],
        2 * 6371000 * np.arcsin(np.minimum(distance / 2, 1)),
        float(maximum_distance_m),
    )


def route_discharge(values, source_area, target_area, routing):
    """Preserve independently measured original kg/s, refusing distant nonzero input."""
    data = np.ma.asarray(values)
    area = np.ma.asarray(source_area)
    target_area = np.asarray(target_area)
    recipient, distance, limit = routing
    if (
        data.shape != area.shape
        or data.size != len(recipient)
        or np.asarray(recipient).dtype.kind not in "iu"
        or np.any(recipient < 0)
        or np.any(recipient >= target_area.size)
        or np.asarray(distance).shape != np.asarray(recipient).shape
        or not np.isfinite(distance).all()
        or np.any(distance < 0)
        or np.ma.is_masked(data)
        or np.ma.is_masked(area)
        or not np.isfinite(data).all()
        or not np.isfinite(area).all()
        or np.any(data < 0)
        or np.any(area <= 0)
        or not np.isfinite(target_area).all()
        or np.any(target_area <= 0)
    ):
        raise ValueError("invalid discharge/area or routing arrays")
    far = (np.asarray(data).ravel() != 0) & (distance > limit)
    if far.any():
        raise ValueError(f'{int(far.sum())} nonzero discharge cells exceed routing limit '
                         f'{limit:.0f}m; maximum {float(distance[far].max()):.0f}m')
    mass = np.bincount(
        recipient, weights=(np.asarray(data, dtype=np.float64) * np.asarray(area, dtype=np.float64)).ravel(),
        minlength=target_area.size
    )
    return mass.reshape(target_area.shape) / target_area


def _original_field(stack, acquisitions, received, field, data_kind):
    """Open checked annual shards and index records without loading 3D arrays."""
    variable, units, cadence, interpretation, height = FIELDS[field]
    datasets, records, identities = [], [], []
    expected_methods = ('area: mean time: point' if interpretation == 'instant' else {
        'friver':'area: mean where sea time: mean',
        'licalvf':'area: time: mean where ice_sheet'}.get(variable, 'area: time: mean'))
    for acquisition, receipt in zip(acquisitions, received):
        identity = receipt['files'][variable]
        original = acquisition.parent/identity['filename']
        if original.stat().st_size != identity['bytes'] or sha256_file(original) != identity['sha256']:
            raise ValueError('original forcing identity changed: '+variable)
        ds = stack.enter_context(netCDF4.Dataset(original))
        var = ds[variable]
        if (ds.source_id != 'MRI-JRA55-do-1-4-0' or var.units not in units
                or var.dimensions != ('time','lat','lon')
                or getattr(ds,'data_kind','observed') != data_kind
                or ds['lon'].units != 'degrees_east' or ds['lat'].units != 'degrees_north'
                or not str(getattr(ds,'license','')).strip()):
            raise ValueError('original forcing metadata mismatch: '+variable)
        if height is not None and (ds['height'].size != 1 or ds['height'].units != 'm'
                                   or ds['height'][:].item() != height):
            raise ValueError('original state height must be 10m')
        if var.cell_methods != expected_methods:
            raise ValueError('unrecognized original spatial/temporal semantics: '+variable)
        if datasets:
            for axis in ('lon','lat'):
                if not np.array_equal(ds[axis][:],datasets[0][axis][:]):
                    raise ValueError('annual shard coordinates disagree: '+variable)
                if interpretation == 'mean' and not np.array_equal(
                        ds[ds[axis].bounds][:], datasets[0][datasets[0][axis].bounds][:]):
                    raise ValueError('annual shard cell bounds disagree: '+variable)
            if ds.license != datasets[0].license:
                raise ValueError('annual shard licenses disagree: '+variable)
        datasets.append(ds)
        raw_times = ds['time'][:]
        raw_bounds = ds[ds['time'].bounds][:] if interpretation == 'mean' else None
        if np.ma.is_masked(raw_times) or (raw_bounds is not None and np.ma.is_masked(raw_bounds)):
            raise ValueError('missing original time or bounds: '+variable)
        times = _seconds(raw_times,ds['time'])
        bounds = _seconds(raw_bounds,ds['time']) if raw_bounds is not None else None
        if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                or (len(times)>1 and not np.allclose(np.diff(times),cadence,rtol=0,atol=1e-6))):
            raise ValueError('invalid original record cadence: '+variable)
        if bounds is not None and (bounds.shape != (len(times),2) or not np.isfinite(bounds).all()
                or not np.allclose(bounds[:,1]-bounds[:,0],cadence,rtol=0,atol=1e-6)
                or not np.allclose(bounds.mean(axis=1),times,rtol=0,atol=1e-6)):
            raise ValueError('invalid original mean bounds: '+variable)
        records.extend((float(time), ds, index, None if bounds is None else bounds[index], identity['sha256'])
                       for index,time in enumerate(times))
        identities.append(identity | {'acquisition_sha256':sha256_file(acquisition)})
    records.sort(key=lambda record:record[0])
    times = np.array([record[0] for record in records])
    if len(times)>1 and not np.allclose(np.diff(times),cadence,rtol=0,atol=1e-6):
        raise ValueError('annual shards overlap or leave a time gap: '+variable)
    bounds = np.array([record[3] for record in records]) if interpretation == 'mean' else None
    return datasets[0], times, bounds, records, identities


def prepare_jra_window(
    acquisition,
    grid_path,
    runoff_area_path,
    output,
    *,
    start,
    end,
    maximum_routing_distance_m=500000,
):
    """Identity-checked originals with retained semantics; streamed selected records.

    Grid NPZ: lon, lat, lon_bounds, lat_bounds, area, wet_mask; arrays use lon,lat.
    Caller bounds numerical preprocessing to one CPU / 180s / 4GiB.
    """
    acquisitions = [Path(path) for path in acquisition] if isinstance(acquisition,(list,tuple)) else [Path(acquisition)]
    if not acquisitions or len(set(path.resolve() for path in acquisitions)) != len(acquisitions):
        raise ValueError('nonempty distinct annual acquisition receipts required')
    grid_path, runoff_area_path = map(Path,(grid_path,runoff_area_path))
    start, end = float(start), float(end)
    received = [load_json(path) for path in acquisitions]
    for receipt in received:
        if (receipt['product'] != 'JRA55-do' or receipt['version'] != '1.4.0'
                or receipt['execution_status'] != 'completed'
                or set(receipt['files']) != {item[0] for item in FIELDS.values()}
                or receipt['verified'] != {name:row['sha256'] for name,row in receipt['files'].items()}):
            raise ValueError('complete original v1.4.0 publisher-identity receipt required')
    data_kinds = {receipt.get('data_kind','observed') for receipt in received}
    if len(data_kinds) != 1:
        raise ValueError('annual acquisition data roles disagree')
    if not np.isfinite([start, end]).all() or start >= end:
        raise ValueError("invalid preprocessing interval")
    with np.load(grid_path, allow_pickle=False) as source:
        required = {"lon", "lat", "lon_bounds", "lat_bounds", "area", "wet_mask"}
        if set(source.files) not in (required, required | {'land_fraction'}):
            raise ValueError("explicit native coordinates/bounds/area/wet_mask required")
        grid = {name: np.array(source[name]) for name in source.files}
    lon, lat, wet, area = (grid[name] for name in ("lon", "lat", "wet_mask", "area"))
    # Validate global bounds and area using the same independently tested overlap contract.
    conservative_rectilinear_weights(
        grid["lon_bounds"], grid["lat_bounds"], grid["lon_bounds"], grid["lat_bounds"]
    )
    if (
        wet.shape != (len(lon), len(lat))
        or area.shape != wet.shape
        or not np.isin(wet, [0, 1]).all()
        or not wet.any()
        or not np.isfinite(area).all()
        or np.any(area <= 0)
        or not np.allclose(lon, grid["lon_bounds"].mean(axis=1), atol=1e-10, rtol=0)
        or not np.allclose(lat, grid["lat_bounds"].mean(axis=1), atol=1e-10, rtol=0)
    ):
        raise ValueError("native centers/mask/positive areas disagree")
    # Reusing an incorrect supplied area for both division and a budget check
    # would cancel its error. Derive the physical rectangle independently.
    derived_area = (6371000.0**2 * np.deg2rad(np.diff(grid['lon_bounds'], axis=1))
                    * np.diff(np.sin(np.deg2rad(grid['lat_bounds'])), axis=1).T)
    if not np.allclose(area, derived_area, rtol=1e-10, atol=1e-6):
        raise ValueError('native area disagrees with spherical cell bounds and 6371000m radius')
    original_area = load_json(runoff_area_path)
    path = runoff_area_path.parent / original_area["path"]
    if (
        path.stat().st_size != original_area["bytes"]
        or sha256_file(path) != original_area["sha256"]
    ):
        raise ValueError("runoff area input identity mismatch")
    with netCDF4.Dataset(path) as ds:
        if ds.source_id != "MRI-JRA55-do-1-4-0" or ds[original_area["variable"]].units != "m2":
            raise ValueError("incorrect original discharge area source/units")
        runoff_area = ds[original_area["variable"]][:]
        runoff_lon, runoff_lat = ds["lon"][:], ds["lat"][:]
    routing = coastal_routing(
        runoff_lon, runoff_lat, lon, lat, wet.T, maximum_distance_m=maximum_routing_distance_m,
        land_fraction=grid['land_fraction'].T if 'land_fraction' in grid else None,
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez(
        output / "routing.npz",
        recipient=routing[0],
        distance_m=routing[1],
        maximum_distance_m=routing[2],
    )
    np.savez(output / "grid.npz", lon=lon, lat=lat, wet_mask=wet)
    result = {
        "schema_version": 1,
        "product": "JRA55-do",
        "version": "1.4.0",
        "data_kind": received[0].get("data_kind", "observed"),
        "files": [],
    }
    report = {
        "status": "running",
        "executed_preparation_sha256": sha256_file(Path(__file__)),
        "executed_reader_mapping_sha256": sha256_file(
            Path(__file__).parents[1] / 'model/inputs/forcing/jra55.py'),
        "numerical_environment": {"numpy": np.__version__, "netCDF4": netCDF4.__version__},
        "acquisition_sha256": sha256_file(acquisitions[0]) if len(acquisitions)==1 else
                              [sha256_file(path) for path in acquisitions],
        "grid_sha256": sha256_file(grid_path),
        "native_area_validation": {"radius_m": 6371000.0,
            "definition": "R2_delta_longitude_delta_sine_latitude",
            "maximum_relative_difference": float(np.max(np.abs(area-derived_area)/derived_area))},
        "runoff_area_sha256": sha256_file(path),
        "area_semantics_source": AREA_SOURCE,
        "window_seconds": [start, end],
        "routing": {
            "definition": "nearest_fractional_or_four_neighbor_coast_spherical_v1" if 'land_fraction' in grid else "nearest_four_neighbor_coast_spherical_v1",
            "sha256": sha256_file(output / "routing.npz"),
            "maximum_distance_m": routing[2],
        },
        "fields": {},
        "execution_ready": False,
        "climate_qualification": False,
    }
    try:
        for field, (variable, units, _, interpretation, height) in FIELDS.items():
            with ExitStack() as stack:
                ds,times,bounds,records,identities = _original_field(
                    stack,acquisitions,received,field,result['data_kind'])
                identity = identities[0]
                var = ds[variable]
                methods = var.cell_methods
                if interpretation == "instant":
                    begin = max(0, np.searchsorted(times, start, side="right") - 1)
                    finish = min(len(times), np.searchsorted(times, end, side="left") + 1)
                    indices = np.arange(begin, finish)
                else:
                    indices = np.flatnonzero((bounds[:, 1] > start) & (bounds[:, 0] < end))
                if not len(indices):
                    raise ValueError("original has no records in the requested window")
                target = output / (variable + ".nc")
                terrestrial = variable in {"friver", "licalvf"}
                if terrestrial:
                    if not np.array_equal(ds["lon"][:], runoff_lon) or not np.array_equal(
                        ds["lat"][:], runoff_lat
                    ):
                        raise ValueError("original discharge area/grid mismatch")
                    weights = None
                else:
                    weights = (
                        bilinear_rectilinear_weights(ds["lon"][:], ds["lat"][:], lon, lat)
                        if interpretation == "instant"
                        else conservative_rectilinear_weights(
                            ds[ds["lon"].bounds][:],
                            ds[ds["lat"].bounds][:],
                            grid["lon_bounds"],
                            grid["lat_bounds"],
                        )
                    )
                    np.savez(
                        output / (variable + "-weights.npz"),
                        latitude=weights[0],
                        longitude=weights[1],
                    )
                integrals = []
                with netCDF4.Dataset(target, "w") as native:
                    native.source_id = ds.source_id
                    native.data_kind = result["data_kind"]
                    native.license = ds.license
                    for axis, values, unit in [
                        ("lon", lon, "degrees_east"),
                        ("lat", lat, "degrees_north"),
                    ]:
                        native.createDimension(axis, len(values))
                        coordinate = native.createVariable(axis, "f8", (axis,))
                        coordinate.units = unit
                        coordinate[:] = values
                    native.createDimension("time", len(indices))
                    native.createDimension("bounds", 2)
                    time = native.createVariable("time", "f8", ("time",))
                    time.units = TIME_UNITS
                    time.calendar = "proleptic_gregorian"
                    time[:] = times[indices]
                    if bounds is not None:
                        time.bounds = "time_bounds"
                        native.createVariable("time_bounds", "f8", ("time", "bounds"))[:] = bounds[
                            indices
                        ]
                    if height is not None:
                        h = native.createVariable("height", "f8")
                        h.units = "m"
                        h.assignValue(height)
                    out = native.createVariable(variable, "f8", ("time", "lat", "lon"), zlib=True)
                    out.units = var.units
                    out.cell_methods = (
                        "time: point" if interpretation == "instant" else "time: mean"
                    )
                    for index, original_index in enumerate(indices):
                        _, source_dataset, source_index, _, source_sha256 = records[original_index]
                        values = source_dataset[variable][source_index]
                        mapped = (
                            route_discharge(values, runoff_area, area.T, routing)
                            if terrestrial
                            else remap_rectilinear_means(values, weights)
                        )
                        out[index] = mapped
                        if terrestrial:
                            before = float(np.einsum('ij,ij->', np.asarray(values, dtype=np.float64),
                                                     np.asarray(runoff_area, dtype=np.float64)))
                            after = float(np.sum(mapped * area.T))
                            if not np.isclose(before, after, rtol=1e-12, atol=1e-6):
                                raise ValueError("terrestrial mass integral changed during routing")
                            integrals.append(
                                {
                                    "index": int(original_index),
                                    "source_index": int(source_index),
                                    "source_sha256": source_sha256,
                                    "original_kg_s": before,
                                    "native_kg_s": after,
                                }
                            )
                result["files"].append(
                    {
                        "field": field,
                        "path": target.name,
                        "sha256": sha256_file(target),
                        "bytes": target.stat().st_size,
                        "source_url": identity["source_url"],
                        "license": ds.license,
                    }
                )
                report["fields"][field] = {
                    "original_sha256": identity["sha256"],
                    "original_sources": identities,
                    "source_url_semantics": "first_original_all_shards_in_original_sources",
                    "original_cell_methods": methods,
                    "indices": indices.tolist(),
                    "selected_records": [{"time_seconds":records[index][0],
                        "source_index":records[index][2], "source_sha256":records[index][4]}
                        for index in indices],
                    "mapping": "coastal_mass"
                    if terrestrial
                    else "bilinear_state"
                    if interpretation == "instant"
                    else "spherical_area_mean",
                    "weights_sha256": None
                    if terrestrial
                    else sha256_file(output / (variable + "-weights.npz")),
                    "mass_integrals": integrals,
                    "prepared_sha256": sha256_file(target),
                }
        (output / "forcing.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        JRA55Forcing(
            output / "forcing.json",
            lon=lon,
            lat=lat,
            wet_mask=wet,
            start_seconds=start,
            end_seconds=end,
        ).sample(start, interval_end_seconds=min(end, start + 10800))
        report["status"] = "prepared_and_reader_verified"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["reason"] = str(error) or type(error).__name__
        report["reason_type"] = type(error).__name__
        raise
    finally:
        (output / "preparation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report
