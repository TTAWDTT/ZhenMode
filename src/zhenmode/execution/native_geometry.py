"""Auditable whole-cell ocean geometry from independent shoreline and relief.

This entry prepares the fixed 360x180 grid used by the first native WOA route.
It records sampled coastline and binary area/volume errors; successful data
preparation never certifies dynamics, local geography or a complete case.
"""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
from scipy.sparse import csr_matrix

from zhenmode.model.inputs.coastline import (
    connect_binary_channels,
    marine_native_edges,
    marine_regions,
    rasterize_gshhg,
    wet_components,
)
from zhenmode.model.solver.geometry.grid import _overlap_matrix
from zhenmode.provenance.sources import (
    load_json,
    production_source_modules,
    sha256_file,
    source_paths,
    source_root,
)


def prepare_native_geometry(parent_geometry, shoreline_acquisition, policy_file, output):
    parent_geometry, shoreline_acquisition, policy_file, output = map(
        Path, (parent_geometry, shoreline_acquisition, policy_file, output)
    )
    parent_path = parent_geometry / "geometry.json"
    parent, acquisition, policy = (
        load_json(p) for p in (parent_path, shoreline_acquisition, policy_file)
    )
    if (
        not {"policy", "regional_refinements"}
        <= set(policy)
        <= {"policy", "regional_refinements", "reference_dry_exclusions"}
        or policy["policy"] != "binary_coast_channels_v1"
        or not isinstance(policy["regional_refinements"], list)
        or not isinstance(policy.get("reference_dry_exclusions", []), list)
    ):
        raise ValueError(
            "geometry policy must select binary_coast_channels_v1 and explicit refinements"
        )
    if acquisition.get("product") != "GSHHG" or acquisition.get("version") != "2.3.7":
        raise ValueError("GSHHG2.3.7 acquisition is required")
    row = next((r for r in acquisition["files"] if r["path"] == "gshhs_h.b"), None)
    if row is None:
        raise ValueError("high-resolution shoreline source is required")
    coast = shoreline_acquisition.parent / row["path"]
    manufactured = acquisition.get("data_kind") == "manufactured"
    if not manufactured and (
        row["sha256"] != "380f585415006f2b648a1a99383f50bf06518728d62991d98d48e7c2de48ec8f"
        or acquisition.get("publisher_sha256")
        != "28600e8f7a08645aab43079326df6504212ec5ccb2b4bcf3b5f4f12ed60e82bc"
        or acquisition.get("actual_sha256") != acquisition.get("publisher_sha256")
        or acquisition.get("publisher_checksum_verified") is not True
    ):
        raise ValueError("original shoreline must match verified publisher acquisition")
    source = Path(parent["source"])
    identities = {
        parent_path: sha256_file(parent_path),
        shoreline_acquisition: sha256_file(shoreline_acquisition),
        policy_file: sha256_file(policy_file),
        coast: row["sha256"],
        source: parent["source_sha256"],
        parent_geometry / "grid.npz": parent["grid_sha256"],
        parent_geometry / "bathymetry.npz": parent["bathymetry_sha256"],
    }
    for path, expected in identities.items():
        if sha256_file(path) != expected:
            raise ValueError("native geometry input identity mismatch: " + str(path))
    if coast.stat().st_size != row["bytes"]:
        raise ValueError("shoreline source byte count mismatch")
    with np.load(parent_geometry / "grid.npz", allow_pickle=False) as data:
        grid = {k: data[k].copy() for k in data.files}
    with np.load(parent_geometry / "bathymetry.npz", allow_pickle=False) as data:
        old_bed = data["depth"].copy()
        if not np.array_equal(data["lon"], grid["lon"]) or not np.array_equal(
            data["lat"], grid["lat"]
        ):
            raise ValueError("parent depth coordinates disagree")
    lon, lat = np.arange(360) + 0.5, np.arange(180) - 89.5
    if (
        old_bed.shape != (360, 180)
        or not np.isfinite(old_bed).all()
        or np.any(old_bed < 0)
        or not np.array_equal(grid["wet_mask"], old_bed > 0)
    ):
        raise ValueError("parent wet mask/depth contract is inconsistent")
    if (
        not np.array_equal(grid["lon"], lon)
        or not np.array_equal(grid["lat"], lat)
        or not np.array_equal(
            grid["lon_bounds"], np.column_stack((np.arange(360), np.arange(1, 361)))
        )
        or not np.array_equal(
            grid["lat_bounds"], np.column_stack((np.arange(180) - 90, np.arange(1, 181) - 90))
        )
    ):
        raise ValueError("this native geometry route requires global one-degree cell bounds")
    area = (
        6371000.0**2 * np.deg2rad(1.0) * np.diff(np.sin(np.deg2rad(np.arange(181) - 90)))[None, :]
    )
    area = np.broadcast_to(area, (360, 180)).copy()
    if not np.allclose(grid["area"], area, rtol=1e-10, atol=1e-6):
        raise ValueError("parent area disagrees with spherical cell bounds")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "geometry_kind": "binary_coast_channels_v1",
        "data_kind": "manufactured" if manufactured else "derived_original_sources",
        "input_sha256": {str(p): s for p, s in identities.items()},
        "source": str(source),
        "source_sha256": parent["source_sha256"],
        "source_metadata": parent.get("source_metadata"),
        "original_NOAA_bytes_reverified": parent.get("original_NOAA_bytes_reverified", False),
        "shoreline_version": "GSHHG2.3.7 high ice-front; freshwater lakes excluded",
        "shoreline_sha256": row["sha256"],
        "smoothing_passes": 0,
        "minimum_depth_filter": None,
        "depth_mapping": "conditional spherical area mean of negative marine relief support",
        "wet_rule": "marine fraction>=0.5 union previous wet cells with marine support; named reference-dry conflicts excluded; then evidenced binary channels",
        "topology_sampling": "0.1degree centres aligned with native bounds; explicit regional refinements",
        "area_exact_polygon_qualification": False,
        "source_geography_certified": False,
        "subcell_topology_merging_is_approximate": True,
        "execution_ready": False,
        "climate_qualification": False,
        "package_source_sha256": {
            n: sha256_file(p)
            for n, p in source_paths(source_root(__file__), production_source_modules()).items()
        },
    }
    try:
        with netCDF4.Dataset(source) as data:
            if data["z"].dimensions != ("lat", "lon") or data["z"].units not in {"m", "meters"}:
                raise ValueError("relief must be positive-up metres on lat,lon")
            source_lon, source_lat = (
                np.ma.asarray(data[k][:]).filled(np.nan) for k in ("lon", "lat")
            )
            z = np.ma.asarray(data["z"][:], dtype=float).filled(np.nan).T
            if getattr(data, "data_kind", None) == "manufactured":
                report["data_kind"] = "manufactured"
        nx, ny = len(source_lon), len(source_lat)
        if (
            nx < 2
            or ny < 2
            or z.shape != (nx, ny)
            or not np.allclose(source_lon, np.arange(nx) * 360.0 / nx, rtol=0, atol=1e-8)
            or not np.allclose(
                source_lat, (np.arange(ny) + 0.5) * 180.0 / ny - 90, rtol=0, atol=1e-8
            )
        ):
            raise ValueError("relief source must have longitude-zero regular global point centres")
        marine = rasterize_gshhg(coast, source_lon, source_lat, expected_sha256=row["sha256"]) == 0
        if not np.isfinite(z[marine]).all():
            raise ValueError("marine relief has unresolved source data")
        wx = csr_matrix(
            _overlap_matrix(np.r_[0.0, (np.arange(nx) + 0.5) * 360.0 / nx, 360.0], np.arange(361))
        )
        wy = csr_matrix(
            _overlap_matrix(
                np.linspace(-90.0, 90.0, ny + 1), np.arange(181) - 90.0, sine_weight=True
            )
        )

        def mapped(values):
            extended = np.concatenate((values, values[:1]), axis=0)
            return (wy @ (wx @ extended).T).T

        fraction = mapped(marine.astype(float))
        negative = marine & (z < 0.0)
        support = mapped(negative.astype(float))
        corrections = {}
        for name, values in (
            ("marine_fraction", fraction),
            ("negative_depth_support_fraction", support),
        ):
            if (
                not np.isfinite(values).all()
                or np.any(values < -1e-12)
                or np.any(values > 1 + 1e-12)
            ):
                raise ValueError("mapped source fractions violate roundoff bounds")
            corrected = np.clip(values, 0.0, 1.0)
            corrections[name] = float(np.max(np.abs(corrected - values)))
            values[...] = corrected
        report["fraction_roundoff"] = {
            "absolute_limit": 1e-12,
            "maximum_corrections": corrections,
            "majority_tie_tolerance": 1e-12,
        }
        source_volume_per_area = mapped(np.where(negative, -z, 0.0))
        depth = source_volume_per_area / np.where(support > 0, support, 1.0)
        wet = (fraction >= 0.5 - 1e-12) | ((old_bed > 0) & (fraction > 0))
        exclusions = []
        seen_exclusions = set()
        for item in policy.get("reference_dry_exclusions", []):
            if (
                not isinstance(item, dict)
                or set(item) != {"lon_lat", "basis"}
                or item["basis"] != "retain_parent_dry_no_negative_marine_relief"
                or not isinstance(item["lon_lat"], list)
                or len(item["lon_lat"]) != 2
            ):
                raise ValueError("invalid reference dry exclusion policy")
            xx, yy = item["lon_lat"]
            if (
                not isinstance(xx, (int, float))
                or not isinstance(yy, (int, float))
                or not 0 <= xx < 360
                or not -90 <= yy < 90
                or xx % 1 != 0.5
                or yy % 1 != 0.5
            ):
                raise ValueError("reference dry exclusion must name a native centre")
            i, j = int(xx), int(yy + 90)
            if (
                (i, j) in seen_exclusions
                or old_bed[i, j] != 0
                or support[i, j] != 0
                or not wet[i, j]
            ):
                raise ValueError(
                    "reference exclusion cannot delete known water or an unrelated dry cell"
                )
            seen_exclusions.add((i, j))
            exclusions.append(
                {
                    "native_flat_index": i * 180 + j,
                    **item,
                    "sampled_marine_fraction": float(fraction[i, j]),
                    "sampled_conflicting_marine_area_m2": float(fraction[i, j] * area[i, j]),
                    "geography_resolved": False,
                    "scope": "retain_reference_dry_pending_reliable_regional_depth",
                }
            )
            wet[i, j] = False
        report["reference_dry_source_conflicts"] = exclusions
        report["native_geographic_qualification"] = False
        report["all_selected_depth_conflicts_explicitly_accounted"] = not bool(
            np.any(wet & (support <= 0))
        )
        if np.any(wet & (support <= 0)):
            report["unsupported_marine_depth_cells"] = [
                {
                    "native_flat_index": int(i * 180 + j),
                    "lon_lat": [float(lon[i]), float(lat[j])],
                    "sampled_marine_fraction": float(fraction[i, j]),
                    "parent_depth_m": float(old_bed[i, j]),
                }
                for i, j in zip(*np.nonzero(wet & (support <= 0)), strict=True)
            ]
            raise ValueError(
                "selected marine cells lack negative relief; do not silently delete them"
            )
        topology_lon, topology_lat = (
            (np.arange(3600) + 0.5) / 10.0,
            (np.arange(1800) + 0.5) / 10.0 - 90.0,
        )
        topology = rasterize_gshhg(coast, topology_lon, topology_lat, expected_sha256=row["sha256"])
        regions, main = marine_regions(topology == 0, periodic_longitude=True)
        possible, x, y = marine_native_edges(regions == main, 10, periodic_longitude=True)
        np.savez_compressed(
            output / "shoreline-support.npz",
            source_lon=source_lon,
            source_lat=source_lat,
            marine_samples=marine,
            topology_lon=topology_lon,
            topology_lat=topology_lat,
            topology_levels=topology,
        )
        original_wet = wet.copy()
        wet, records = connect_binary_channels(wet, support, possible, x, y, lon, lat)
        np.savez_compressed(
            output / "native-edge-evidence.npz", possible_cells=possible, x_faces=x, y_faces=y
        )
        refinements, seen = [], set()
        for item in policy["regional_refinements"]:
            if (
                set(item) != {"id", "lon_bounds_deg", "lat_bounds_deg", "seed_lon_lat"}
                or not isinstance(item["id"], str)
                or not item["id"].replace("-", "").isalnum()
            ):
                raise ValueError("invalid regional refinement declaration")
            if item["id"] in seen:
                raise ValueError("duplicate regional refinement identity")
            seen.add(item["id"])
            west, east = item["lon_bounds_deg"]
            south, north = item["lat_bounds_deg"]
            if (
                any(type(v) is not int for v in (west, east, south, north))
                or not 0 <= west < east <= 360
                or not -90 <= south < north <= 90
                or (east - west) * (north - south) > 100
            ):
                raise ValueError(
                    "regional refinement must be integer native bounds, at most100 cells"
                )
            ll, yy = (
                west + (np.arange((east - west) * 100) + 0.5) / 100.0,
                south + (np.arange((north - south) * 100) + 0.5) / 100.0,
            )
            levels = rasterize_gshhg(coast, ll, yy, expected_sha256=row["sha256"])
            regional_labels, _ = marine_regions(levels == 0, periodic_longitude=False)
            seed_x, seed_y = item["seed_lon_lat"]
            if (
                not west <= seed_x < east
                or not south <= seed_y < north
                or seed_x % 1 != 0.5
                or seed_y % 1 != 0.5
            ):
                raise ValueError("regional seed must be a native centre inside its bounds")
            ii, jj = int((seed_x - west) * 100), int((seed_y - south) * 100)
            group_id = regional_labels[ii, jj]
            if not group_id:
                raise ValueError("regional refinement seed has no marine sample")
            selected = regional_labels == group_id
            pp, xx, yx = marine_native_edges(selected, 100, periodic_longitude=False)
            full = [np.zeros_like(wet) for _ in range(3)]
            sl = np.s_[west:east, south + 90 : north + 90]
            for target, value in zip(full, (pp, xx, yx), strict=True):
                target[sl] = value
            seed_flat = int(seed_x) * 180 + int(seed_y + 90)
            wet, regional_records = connect_binary_channels(
                wet, support, *full, lon, lat, targets=[seed_flat]
            )
            target = output / f"regional-{item['id']}.npz"
            np.savez_compressed(
                target,
                lon=ll,
                lat=yy,
                levels=levels,
                selected_marine_component=selected,
                possible_cells=pp,
                x_faces=xx,
                y_faces=yx,
            )
            refinements.append(
                {
                    **item,
                    "output": target.name,
                    "sha256": sha256_file(target),
                    "component_reaches_regional_boundary": bool(
                        selected[0].any()
                        or selected[-1].any()
                        or selected[:, 0].any()
                        or selected[:, -1].any()
                    ),
                    "channel_records": regional_records,
                    "regional_climate_qualification": False,
                }
            )
        final_depth = depth * wet
        grid.update(wet_mask=wet.astype(float), area=area, land_fraction=1.0 - fraction)
        np.savez(output / "grid.npz", **grid)
        np.savez(output / "bathymetry.npz", lon=lon, lat=lat, depth=final_depth)
        np.savez_compressed(
            output / "coastline-mapping.npz",
            marine_fraction=fraction,
            negative_depth_support_fraction=support,
            conditional_depth=depth,
            source_volume_per_area=source_volume_per_area,
            initial_wet_mask=original_wet,
            channel_added_mask=wet & ~original_wet,
        )
        components = wet_components(wet)
        added = np.flatnonzero(wet & ~original_wet).tolist()
        ocean_area = float(np.sum(fraction * area))
        binary_area = float(np.sum(wet * area))
        source_volume = float(np.sum(source_volume_per_area * area))
        binary_volume = float(np.sum(final_depth * area))
        report.update(
            status="native_geometry_prepared"
            if len(components) == 1
            else "blocked_native_connections",
            shape=[360, 180],
            wet_cells=int(wet.sum()),
            native_components=len(components),
            channel_added_cells=added,
            channel_records=records,
            regional_refinements=refinements,
            removed_parent_wet_cells=np.flatnonzero((old_bed > 0) & ~wet).tolist(),
            sampled_marine_nonnegative_relief_points=int(np.sum(marine & (z >= 0))),
            sampled_marine_area_m2=ocean_area,
            binary_marine_area_m2=binary_area,
            binary_area_relative_error=(binary_area / ocean_area - 1),
            binary_area_overrepresented_m2=float(np.sum(np.maximum(wet - fraction, 0.0) * area)),
            sampled_marine_area_unrepresented_m2=float(
                np.sum(np.maximum(fraction - wet, 0.0) * area)
            ),
            negative_marine_source_volume_m3=source_volume,
            binary_volume_m3=binary_volume,
            binary_volume_relative_error=(binary_volume / source_volume - 1),
            binary_volume_overrepresented_m3=float(
                np.sum(np.maximum(final_depth - source_volume_per_area, 0.0) * area)
            ),
            negative_marine_volume_unrepresented_m3=float(
                np.sum(np.maximum(source_volume_per_area - final_depth, 0.0) * area)
            ),
            minimum_wet_depth_m=float(final_depth[wet].min()),
            maximum_wet_depth_m=float(final_depth.max()),
            original_topology_unwitnessed_binary_open_faces={
                "x": int(np.sum(wet * np.roll(wet, -1, axis=0) & ~x)),
                "y": int(np.sum((wet * np.roll(wet, -1, axis=1) & ~y)[:, :-1])),
            },
            grid_sha256=sha256_file(output / "grid.npz"),
            bathymetry_sha256=sha256_file(output / "bathymetry.npz"),
        )
        for path, expected in identities.items():
            if sha256_file(path) != expected:
                raise ValueError("native geometry source/receipt changed during preparation")
        if report["package_source_sha256"] != {
            n: sha256_file(p)
            for n, p in source_paths(source_root(__file__), production_source_modules()).items()
        }:
            raise ValueError("native geometry executed package changed")
        report["outputs"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in output.iterdir()
            if p.is_file()
        }
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", reason_type=type(error).__name__, reason=str(error))
        raise
    finally:
        (output / "geometry.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    return report
