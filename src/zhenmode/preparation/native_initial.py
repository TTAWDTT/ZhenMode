"""Identity-bound fixed native columns and paired WOA preparation.

This first native route requires equal horizontal source/native coordinates.
Unanchored support is written explicitly and blocks complete initialization;
it is never repaired implicitly by a downstream reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
from scipy.sparse import save_npz

from zhenmode.model.config import CP0_TEOS10, RHO_0
from zhenmode.model.inputs.forcing.jra55 import conservative_rectilinear_weights
from zhenmode.model.inputs.initial_conditions import smooth_native_paired_holes
from zhenmode.model.solver.geometry.grid import fixed_reference_nodal_cells
from zhenmode.model.solver.physics.teos10 import conservative_from_potential, validate_state
from zhenmode.provenance.sources import (
    checked_file,
    load_json,
    package_source_hashes,
    sha256_file,
)


def prepare_native_initialization(source_prepared, geometry, nodes_file, output):
    """Prepare explicit fixed cells, PT/SP for MOM and equivalent CT/SR for FD.

    Native point samples differ from MOM finite-volume layer means. This exports
    a Z-field for MOM's subsequent native vertical remapping; it does not label
    it as a MOM layer initial state. Original holes/paired support, same-depth
    harmonic infill, deepest-source extension and unresolved components remain
    distinct masks. Initial inventories include resolved support only until all
    native wet values are qualified. Complete case execution is a later gate.
    """
    source_prepared, geometry, nodes_file, output = map(
        Path, (source_prepared, geometry, nodes_file, output)
    )
    receipt_paths = (
        source_prepared / "initialization.json",
        geometry / "geometry.json",
        nodes_file,
    )
    receipt_shas = {str(p): sha256_file(p) for p in receipt_paths}
    origin, geo, nodes = (load_json(p) for p in receipt_paths)
    if geo.get("geometry_kind") == "binary_coast_channels_v1" and geo.get("status") != "native_geometry_prepared":
        raise ValueError("binary coastal geometry still has unresolved native connections")
    if origin.get("status") != "original_grid_thermodynamics_prepared":
        raise ValueError("completed original-grid WOA preparation is required")
    if set(nodes) != {"z_nodes_m"}:
        raise ValueError("nodes JSON must contain exactly z_nodes_m (negative downward)")
    z = np.asarray(nodes["z_nodes_m"])
    if z.dtype.kind not in "fiu":
        raise ValueError("native nodes must be real numeric metres")
    source = checked_file(source_prepared, origin["output"])
    grid_path, bed_path = geometry / "grid.npz", geometry / "bathymetry.npz"
    identities = {
        source: origin["output"]["sha256"],
        grid_path: geo["grid_sha256"],
        bed_path: geo["bathymetry_sha256"],
    }
    for path, expected in identities.items():
        if sha256_file(path) != expected:
            raise ValueError("native preparation input identity mismatch: " + str(path))
    with (
        np.load(grid_path, allow_pickle=False) as grid,
        np.load(bed_path, allow_pickle=False) as bathymetry,
    ):
        lon, lat, wet, area = (
            np.asarray(grid[k]).copy() for k in ("lon", "lat", "wet_mask", "area")
        )
        bed = np.asarray(bathymetry["depth"]).copy()
        bounds = {k: np.asarray(grid[k]).copy() for k in ("lon_bounds", "lat_bounds")}
        if (
            not np.array_equal(lon, bathymetry["lon"])
            or not np.array_equal(lat, bathymetry["lat"])
            or area.shape != (len(lon), len(lat))
            or not np.isfinite(area).all()
            or np.any(area <= 0)
        ):
            raise ValueError("native area/bathymetry coordinates are invalid")
    conservative_rectilinear_weights(
        bounds["lon_bounds"], bounds["lat_bounds"], bounds["lon_bounds"], bounds["lat_bounds"]
    )
    derived_area = (
        6371000.0**2
        * np.deg2rad(np.diff(bounds["lon_bounds"], axis=1))
        * np.diff(np.sin(np.deg2rad(bounds["lat_bounds"])), axis=1).T
    )
    if (
        not np.allclose(lon, bounds["lon_bounds"].mean(axis=1), rtol=0, atol=1e-10)
        or not np.allclose(lat, bounds["lat_bounds"].mean(axis=1), rtol=0, atol=1e-10)
        or not np.allclose(area, derived_area, rtol=1e-10, atol=1e-6)
    ):
        raise ValueError("native area/centres disagree with spherical bounds and 6371000m radius")
    cells = fixed_reference_nodal_cells(z, bed, wet)
    wet3 = cells["wet_node_mask"]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "data_kind": origin["data_kind"],
        "geometry": "fixed_partial_v1",
        "source_receipt_sha256": receipt_shas[str(receipt_paths[0])],
        "geometry_receipt_sha256": receipt_shas[str(receipt_paths[1])],
        "bathymetry_source_provenance": {
            k: geo.get(k)
            for k in (
                "source",
                "source_sha256",
                "source_metadata",
                "original_NOAA_bytes_reverified",
            )
        },
        "nodes_file_sha256": receipt_shas[str(nodes_file)],
        "input_sha256": {str(p): s for p, s in identities.items()},
        "mapping_order": "source_T_SP_to_PT_SR_then_native_point_mapping_infill_extension_then_CT_from_PT_SR",
        "initial_datetime": "1958-01-01T00:00:00",
        "source_time_definition": origin["source_time_definition"],
        "horizontal_mapping": "coordinate_identity_with_periodic_longitude_reordering",
        "vertical_mapping": "linear_point_interpolation_zero_gradient_beyond_deepest_WOA_depth",
        "infill": "spherical_four_neighbour_Dirichlet_with_original_paired_values_held_exact",
        "salinity_definition": "SP_for_MOM_SR_approximates_SA_for_FD_no_geographic_anomaly",
        "native_initialization_ready": False,
        "mom_layer_initialization_ready": False,
        "execution_ready": False,
        "climate_qualification": False,
        "levels": [],
        "package_source_sha256": package_source_hashes(__file__),
    }
    target = output / "native_point_fields.nc"
    try:
        with netCDF4.Dataset(source) as original, netCDF4.Dataset(target, "w") as native:
            original_lon, original_lat, depths = (
                np.asarray(original[k][:]) for k in ("lon", "lat", "depth")
            )
            order = np.argsort(np.mod(original_lon, 360.0))
            if (
                not np.array_equal(np.mod(original_lon, 360.0)[order], lon)
                or not np.array_equal(original_lat, lat)
                or depths[0] != 0.0
                or not np.isfinite(depths).all()
                or np.any(np.diff(depths) <= 0)
            ):
                raise ValueError(
                    "native point preparation requires matching horizontal coordinates and ordered source depths"
                )
            for name, unit in (("ptemp", "degrees_celsius"), ("sr", "g kg-1")):
                if (
                    original[name].dimensions != ("time", "depth", "lat", "lon")
                    or original[name].units != unit
                ):
                    raise ValueError("native WOA source field dimensions/units disagree")
            if original["paired_source_support"].dimensions != ("time", "depth", "lat", "lon"):
                raise ValueError("source paired support dimensions disagree")
            native.title = "WOA native point preparation; MOM layer remapping is separate"
            native.Conventions = "CF-1.6"
            native.data_kind = origin["data_kind"]
            native.license = original.license
            native.createDimension("time", 1)
            native.createVariable("time", "f8", ("time",))[:] = 0.0
            native["time"].units = "days since 1958-01-01 00:00:00"
            native["time"].calendar = "proleptic_gregorian"
            native["time"].cartesian_axis = "T"
            for name, values, unit, axis in (
                ("lon", lon, "degrees_east", "X"),
                ("lat", lat, "degrees_north", "Y"),
                ("depth", -z, "meters", "Z"),
            ):
                native.createDimension(name, len(values))
                variable = native.createVariable(name, "f8", (name,))
                variable[:] = values
                variable.units = unit
                variable.cartesian_axis = axis
            native["depth"].positive = "down"
            deep_thickness = (
                np.maximum(
                    cells["cell_bottom_m"] - np.maximum(cells["cell_top_m"], depths[-1]), 0.0
                )
                * wet3
            )
            deep_variable = native.createVariable(
                "deep_extension_thickness_m", "f8", ("depth", "lat", "lon")
            )
            deep_variable.units = "m"
            deep_variable[:] = np.ascontiguousarray(deep_thickness.transpose(2, 1, 0))
            fields = {}
            for name, unit in (
                ("ptemp", "degrees_celsius"),
                ("ct", "degrees_celsius"),
                ("sr", "g kg-1"),
                ("salt", "1"),
            ):
                fields[name] = native.createVariable(
                    name,
                    "f8",
                    ("time", "depth", "lat", "lon"),
                    fill_value=9.96921e36,
                    zlib=True,
                    complevel=1,
                )
                fields[name].units = unit
            fields["ptemp"].standard_name = "sea_water_potential_temperature"
            fields["ptemp"].reference_pressure_dbar = 0.0
            fields["ct"].standard_name = "sea_water_conservative_temperature"
            fields["salt"].standard_name = "sea_water_salinity"
            mask_names = (
                "wet_node_mask",
                "original_paired_support",
                "horizontal_infill",
                "deep_extension",
                "unresolved_support",
            )
            masks = {
                name: native.createVariable(name, "i1", ("depth", "lat", "lon"))
                for name in mask_names
            }
            for name, value in cells.items():
                if value.ndim == 3 and name in {"cell_top_m", "cell_bottom_m", "thickness_m"}:
                    v = native.createVariable(name, "f8", ("depth", "lat", "lon"))
                    v.units = "m"
                    v[:] = np.ascontiguousarray(value.transpose(2, 1, 0))
            anchors = native.createVariable(
                "nearest_original_anchor_flat_index", "i8", ("depth", "lat", "lon")
            )
            distances = native.createVariable(
                "nearest_anchor_path_distance_m", "f8", ("depth", "lat", "lon")
            )
            distances.units = "m"
            total_resolved_volume, total_ct_volume, total_sr_volume = 0.0, 0.0, 0.0
            deep_reference = None
            for level, node in enumerate(-z):
                mapped_depth = min(float(node), float(depths[-1]))
                upper = int(np.searchsorted(depths, mapped_depth))
                lower = upper if depths[upper] == mapped_depth else upper - 1
                weight = (
                    0.0
                    if upper == lower
                    else float((mapped_depth - depths[lower]) / (depths[upper] - depths[lower]))
                )
                read = []
                for k in sorted({lower, upper}):
                    raw = np.stack(
                        [
                            np.ma.asarray(original[name][0, k], dtype=float).filled(np.nan).T[order]
                            for name in ("ptemp", "sr")
                        ]
                    )
                    support = np.asarray(original["paired_source_support"][0, k]).T[order]
                    if (
                        not np.isin(support, [0, 1]).all()
                        or not np.array_equal(support > 0, np.isfinite(raw).all(axis=0))
                        or np.isinf(raw).any()
                    ):
                        raise ValueError("native source paired support is inconsistent")
                    read.append(raw)
                raw = read[0] if lower == upper else (1.0 - weight) * read[0] + weight * read[1]
                if node > depths[-1]:
                    # Extend the prepared deepest-source plane, rather than
                    # solving a new horizontal problem on the deeper footprint.
                    # A shallower sill may disconnect that deeper footprint;
                    # it must not change the already declared zero-gradient
                    # continuation or remove its resolved source-depth anchor.
                    if deep_reference is None:
                        reference_wet = (bed >= depths[-1]) & (wet > 0)
                        deep_reference = smooth_native_paired_holes(raw, reference_wet, lon, lat)
                    keep = wet3[..., level]
                    filled = dict(deep_reference)
                    filled["fields"] = np.where(keep[None], deep_reference["fields"], np.nan)
                    for name in ("original_paired_mask", "infill_mask", "unresolved_mask"):
                        filled[name] = deep_reference[name] & keep
                    for name in ("nearest_original_anchor_flat_index", "nearest_anchor_path_distance_m"):
                        filled[name] = np.where(keep, deep_reference[name], -1)
                    filled["unsupported_components"] = [
                        {"native_flat_indices": selected}
                        for component in deep_reference["unsupported_components"]
                        if (selected := [k for k in component["native_flat_indices"] if keep.flat[k]])
                    ]
                    if "deep_extension_reference" not in report:
                        reference_path = output / "deep-extension-reference.npz"
                        np.savez(
                            reference_path, fields=deep_reference["fields"],
                            wet_mask=(bed >= depths[-1]) & (wet > 0),
                            original_paired_mask=deep_reference["original_paired_mask"],
                            horizontal_infill_mask=deep_reference["infill_mask"],
                            unresolved_mask=deep_reference["unresolved_mask"],
                            nearest_original_anchor_flat_index=deep_reference["nearest_original_anchor_flat_index"],
                            nearest_anchor_path_distance_m=deep_reference["nearest_anchor_path_distance_m"],
                        )
                        report["deep_extension_reference"] = {
                            "path": reference_path.name, "sha256": sha256_file(reference_path),
                            "source_depth_m": float(depths[-1]),
                            "rule": "zero_gradient_of_prepared_PT_SR_on_same_column",
                            "trace_domain": "deepest_source_depth_wet_footprint_not_deeper_native_footprint",
                        }
                else:
                    filled = smooth_native_paired_holes(raw, wet3[..., level], lon, lat)
                    if node == depths[-1]:
                        deep_reference = filled
                pt, sr = filled["fields"]
                resolved = np.isfinite(pt) & np.isfinite(sr) & wet3[..., level]
                safe_pt, safe_sr = np.where(resolved, pt, 0.0), np.where(resolved, sr, 35.0)
                validate_state(safe_sr, safe_pt, 0.0)
                ct = np.asarray(conservative_from_potential(safe_sr, safe_pt))
                validate_state(safe_sr, ct, 0.0)
                for name, value in (
                    ("ptemp", pt),
                    ("ct", ct),
                    ("sr", sr),
                    ("salt", sr * 35.0 / 35.16504),
                ):
                    fields[name][0, level] = np.ma.array(
                        np.ascontiguousarray(value.T), mask=np.ascontiguousarray(~resolved.T)
                    )
                for name, value in zip(
                    mask_names,
                    (
                        wet3[..., level],
                        filled["original_paired_mask"],
                        filled["infill_mask"],
                        deep_thickness[..., level] > 0.0,
                        filled["unresolved_mask"],
                    ),
                    strict=True,
                ):
                    masks[name][level] = np.ascontiguousarray(value.T, dtype=np.int8)
                anchors[level] = np.ascontiguousarray(
                    filled["nearest_original_anchor_flat_index"].T
                )
                distances[level] = np.ascontiguousarray(filled["nearest_anchor_path_distance_m"].T)
                matrix_path = output / f"infill-system-{level:02d}.npz"
                save_npz(matrix_path, filled["infill_matrix"])
                unknown_path = output / f"infill-unknowns-{level:02d}.npy"
                np.save(unknown_path, filled["infill_unknown_flat_indices"], allow_pickle=False)
                report["levels"].append(
                    {
                        "node_depth_m": float(node),
                        "source_depth_bracket_m": [float(depths[lower]), float(depths[upper])],
                        "source_upper_weight": weight,
                        "original_paired_wet_nodes": int(filled["original_paired_mask"].sum()),
                        "horizontal_infill_nodes": int(filled["infill_mask"].sum()),
                        "unresolved_nodes": int(filled["unresolved_mask"].sum()),
                        "deep_extension_nodes": int((wet3[..., level] & (node > depths[-1])).sum()),
                        "deep_extension_cell_volume_m3": float(
                            np.sum(area * deep_thickness[..., level])
                        ),
                        "unsupported_components": filled["unsupported_components"],
                        "linear_system_relative_residual": filled[
                            "linear_system_relative_residual"
                        ],
                        "infill_system": {
                            "path": matrix_path.name,
                            "sha256": sha256_file(matrix_path),
                        },
                        "infill_unknowns": {
                            "path": unknown_path.name,
                            "sha256": sha256_file(unknown_path),
                        },
                        "infill_system_reference_depth_m": float(min(node, depths[-1])),
                        "infill_inherited_from_deepest_prepared_plane": bool(node > depths[-1]),
                    }
                )
                volume = area * cells["thickness_m"][..., level] * resolved
                total_resolved_volume += float(volume.sum())
                total_ct_volume += float(np.sum(np.where(resolved, ct, 0.0) * volume))
                total_sr_volume += float(np.sum(np.where(resolved, sr, 0.0) * volume))
            native.complete_wet_support = int(
                not any(r["unresolved_nodes"] for r in report["levels"])
            )
        np.savez(
            output / "native_geometry.npz",
            lon=lon,
            lat=lat,
            area=area,
            depth=bed,
            wet_mask=wet,
            **bounds,
            **cells,
        )
        report["water_inventory"] = {
            "geometry_volume_m3": float(np.sum(area * bed)),
            "deep_extension_volume_m3": float(np.sum(area[..., None] * deep_thickness)),
            "resolved_initial_volume_m3": total_resolved_volume,
            "resolved_reference_CT_enthalpy_J": RHO_0 * CP0_TEOS10 * total_ct_volume,
            "resolved_reference_salt_kg": RHO_0 / 1000.0 * total_sr_volume,
            "scope": "resolved_point_quadrature_not_MOM_layer_inventory",
            "area_definition": "input_geographic_cell_area",
        }
        for path, expected in identities.items():
            if sha256_file(path) != expected:
                raise ValueError("native input changed during preparation")
        for path in receipt_paths:
            if sha256_file(path) != receipt_shas[str(path)]:
                raise ValueError("native input receipt changed during preparation")
        sources = package_source_hashes(__file__)
        if sources != report["package_source_sha256"]:
            raise ValueError("executed native preparation package changed")
        for row in report["levels"]:
            for field in ("infill_system", "infill_unknowns"):
                if sha256_file(output / row[field]["path"]) != row[field]["sha256"]:
                    raise ValueError("native infill trace changed during preparation")
        if "deep_extension_reference" in report:
            ref = report["deep_extension_reference"]
            if sha256_file(output / ref["path"]) != ref["sha256"]:
                raise ValueError("native deep extension trace changed during preparation")
        complete = not any(row["unresolved_nodes"] for row in report["levels"])
        report.update(
            status="native_points_prepared" if complete else "blocked_unanchored_native_support",
            complete_wet_support=complete,
            native_initialization_ready=False,
            output={
                "path": target.name,
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            },
            geometry_output={
                "path": "native_geometry.npz",
                "sha256": sha256_file(output / "native_geometry.npz"),
            },
        )
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", reason_type=type(error).__name__, reason=str(error))
        raise
    finally:
        (output / "native_initialization.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return report
