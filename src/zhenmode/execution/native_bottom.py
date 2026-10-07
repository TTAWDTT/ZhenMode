"""Review-bound regional priority, same-column bottom fallback and its trace.

Complete paired point support is a data gate. MOM layer remapping, numeric
EOS eligibility, dynamical sensitivity and full-case budgets remain separate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np

from zhenmode.execution.initialization import _checked_file
from zhenmode.model.config import CP0_TEOS10, RHO_0
from zhenmode.model.inputs.initial_conditions import extend_native_bottom_pairs
from zhenmode.model.solver.geometry.grid import fixed_reference_nodal_cells
from zhenmode.model.solver.physics.teos10 import conservative_from_potential, validate_state
from zhenmode.provenance.sources import (
    load_json,
    production_source_modules,
    sha256_file,
    source_paths,
    source_root,
)


def complete_native_bottom(native_prepared, review_file, output):
    """Complete only missing bottom points, after a named per-column review.

    Review JSON selects regional_profiles_then_same_column_zero_gradient_v1,
    binds the native receipt and evidence files, and covers every affected
    column exactly once. Optional regional PT/SR point pairs take precedence;
    each names its evidence index and source trace. Unfilled surface/interior
    support remains blocked. Existing resolved fields are copied unchanged.
    """
    native_prepared, review_file, output = map(Path, (native_prepared, review_file, output))
    receipt = native_prepared / "native_initialization.json"
    receipt_sha, review_sha = sha256_file(receipt), sha256_file(review_file)
    parent, review = load_json(receipt), load_json(review_file)
    if parent.get("status") not in {"native_points_prepared", "blocked_unanchored_native_support"}:
        raise ValueError("native point preparation receipt is required")
    if (
        set(review) != {"policy", "native_receipt_sha256", "evidence", "columns"}
        or review["policy"] != "regional_profiles_then_same_column_zero_gradient_v1"
        or review["native_receipt_sha256"] != receipt_sha
        or not isinstance(review["evidence"], list)
        or not isinstance(review["columns"], list)
    ):
        raise ValueError("bottom review must bind native receipt and explicit regional priority")
    evidence = [_checked_file(review_file.parent, row) for row in review["evidence"]]
    if not evidence:
        raise ValueError(
            "regional review needs recorded evidence, including unavailable/unsupported profiles"
        )
    source = _checked_file(native_prepared, parent["output"])
    geometry = native_prepared / parent["geometry_output"]["path"]
    identities = {
        receipt: receipt_sha,
        review_file: review_sha,
        source: parent["output"]["sha256"],
        geometry: parent["geometry_output"]["sha256"],
    }
    identities.update(
        {p: row["sha256"] for p, row in zip(evidence, review["evidence"], strict=True)}
    )
    for p, expected in identities.items():
        if sha256_file(p) != expected:
            raise ValueError("bottom completion input identity mismatch")
    with np.load(geometry, allow_pickle=False) as g:
        geom = {k: g[k].copy() for k in g.files}
    rebuilt = fixed_reference_nodal_cells(-geom["node_depth_m"], geom["depth"], geom["wet_mask"])
    for name, values in rebuilt.items():
        if not np.array_equal(values, geom[name]):
            raise ValueError("native reference geometry is inconsistent")
    area_from_bounds = (
        6371000.0**2
        * np.deg2rad(np.diff(geom["lon_bounds"], axis=1))
        * np.diff(np.sin(np.deg2rad(geom["lat_bounds"])), axis=1).T
    )
    if (
        geom["area"].shape != geom["depth"].shape
        or not np.isfinite(geom["area"]).all()
        or np.any(geom["area"] <= 0)
        or not np.allclose(geom["area"], area_from_bounds, rtol=1e-10, atol=1e-6)
    ):
        raise ValueError("native area disagrees with spherical cell bounds")
    with netCDF4.Dataset(source) as ds:
        if (
            not np.array_equal(ds["lon"][:], geom["lon"])
            or not np.array_equal(ds["lat"][:], geom["lat"])
            or not np.array_equal(ds["depth"][:], geom["node_depth_m"])
        ):
            raise ValueError("bottom source coordinates disagree with reference geometry")
        for name, units in (
            ("ptemp", "degrees_celsius"),
            ("ct", "degrees_celsius"),
            ("sr", "g kg-1"),
            ("salt", "1"),
        ):
            if ds[name].dimensions != ("time", "depth", "lat", "lon") or ds[name].units != units:
                raise ValueError("bottom source tracer dimensions/units disagree")
        pairs = np.stack(
            [
                np.ma.asarray(ds[n][0], dtype=float).filled(np.nan).transpose(2, 1, 0)
                for n in ("ptemp", "sr")
            ]
        )
        old_ct = np.ma.asarray(ds["ct"][0], dtype=float).filled(np.nan).transpose(2, 1, 0)
        masks = {
            n: np.asarray(ds[n][:]).transpose(2, 1, 0)
            for n in (
                "wet_node_mask",
                "unresolved_support",
                "original_paired_support",
                "horizontal_infill",
                "deep_extension",
                "nearest_original_anchor_flat_index",
                "nearest_anchor_path_distance_m",
            )
        }
    wet = geom["wet_node_mask"].astype(bool)
    known = np.isfinite(pairs).all(axis=0) & wet
    if (
        not np.isin(masks["wet_node_mask"], [0, 1]).all()
        or not np.isin(masks["unresolved_support"], [0, 1]).all()
        or not np.array_equal(masks["wet_node_mask"] > 0, wet)
        or not np.array_equal(masks["unresolved_support"] > 0, wet & ~known)
        or not np.array_equal(np.isfinite(old_ct) & wet, known)
    ):
        raise ValueError("bottom source support masks disagree with paired fields")
    result = extend_native_bottom_pairs(pairs, wet, geom["node_depth_m"])
    fill = result["bottom_filled_mask"]
    ny, nz = wet.shape[1:]
    needed = set(np.flatnonzero(fill.any(axis=-1)))
    seen = set()
    regional = np.zeros_like(wet)
    evidence_index = np.full(wet.shape, -1, dtype=np.int64)
    for row in review["columns"]:
        if (
            not isinstance(row, dict)
            or set(row)
            != {"native_flat_index", "decision", "reason", "evidence_index", "regional_points"}
            or type(row["native_flat_index"]) is not int
            or row["native_flat_index"] not in needed
            or row["native_flat_index"] in seen
            or type(row["evidence_index"]) is not int
            or not 0 <= row["evidence_index"] < len(evidence)
            or not isinstance(row["reason"], str)
            or not row["reason"].strip()
            or row["decision"]
            not in {"no_qualified_regional_profile", "regional_profile_then_fallback"}
            or not isinstance(row["regional_points"], list)
            or (row["decision"] == "no_qualified_regional_profile" and row["regional_points"])
            or (row["decision"] == "regional_profile_then_fallback" and not row["regional_points"])
        ):
            raise ValueError(
                "bottom review must cover each affected column once with a traced regional decision"
            )
        seen.add(row["native_flat_index"])
        i, j = divmod(row["native_flat_index"], ny)
        evidence_index[i, j, fill[i, j]] = row["evidence_index"]
        for point in row["regional_points"]:
            if (
                not isinstance(point, dict)
                or set(point) != {"node_index", "ptemp", "sr", "source_trace"}
                or type(point["node_index"]) is not int
                or not 0 <= point["node_index"] < nz
                or not isinstance(point["source_trace"], str)
                or not point["source_trace"].strip()
            ):
                raise ValueError(
                    "regional point needs native index and recorded PT/SR source trace"
                )
            k = point["node_index"]
            if not fill[i, j, k] or regional[i, j, k]:
                raise ValueError("regional profile cannot replace resolved/non-bottom points")
            if any(type(point[n]) not in (int, float) for n in ("ptemp", "sr")):
                raise ValueError("regional PT/SR must be real numbers")
            result["fields"][:, i, j, k] = [point["ptemp"], point["sr"]]
            regional[i, j, k] = True
    if seen != needed:
        raise ValueError("regional review omits affected bottom columns")
    pt, sr = result["fields"]
    resolved = wet & np.isfinite(pt) & np.isfinite(sr)
    validate_state(np.where(resolved, sr, 35), np.where(resolved, pt, 0), 0)
    ct = np.asarray(
        conservative_from_potential(np.where(resolved, sr, 35), np.where(resolved, pt, 0))
    )
    validate_state(np.where(resolved, sr, 35), ct, 0)
    # Keep every already-resolved CT bit, including its original converter's
    # rounding; only newly assigned points use the new conversion above.
    ct = np.where(known, old_ct, ct)
    validate_state(np.where(resolved, sr, 35), np.where(resolved, ct, 0), 0)
    thickness = geom["thickness_m"]
    volume = geom["area"][..., None] * thickness
    alternate_pt, alternate_sr = result["linear_sensitivity_fields"]
    supported = result["linear_sensitivity_support"] & fill
    candidate = (
        supported
        & np.isfinite(alternate_pt)
        & np.isfinite(alternate_sr)
        & (alternate_pt >= -3)
        & (alternate_pt <= 40)
        & (alternate_sr >= 0)
        & (alternate_sr <= 42)
    )
    alternate_ct = np.asarray(
        conservative_from_potential(
            np.where(candidate, alternate_sr, 35), np.where(candidate, alternate_pt, 0)
        )
    )
    candidate &= np.isfinite(alternate_ct) & (alternate_ct >= -3) & (alternate_ct <= 40)
    second_donors = np.full(wet.shape, -1, dtype=np.int64)
    for i, j in zip(*np.nonzero(fill.any(axis=-1)), strict=True):
        donors = np.flatnonzero(known[i, j])
        if len(donors) >= 2:
            second_donors[i, j, fill[i, j]] = donors[-2]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "data_kind": parent["data_kind"],
        "policy": review["policy"],
        "input_sha256": {str(p): s for p, s in identities.items()},
        "parent_native_receipt_sha256": receipt_sha,
        "bottom_review_sha256": review_sha,
        "review_file_original_location": str(review_file.resolve()),
        "execution_ready": False,
        "native_initialization_ready": False,
        "mom_layer_initialization_ready": False,
        "climate_qualification": False,
        "geometry": parent["geometry"],
        "source_time_definition": parent["source_time_definition"],
        "bathymetry_source_provenance": parent["bathymetry_source_provenance"],
        "package_source_sha256": {
            n: sha256_file(p)
            for n, p in source_paths(source_root(__file__), production_source_modules()).items()
        },
    }
    try:
        target = output / "native_point_fields.nc"
        shutil.copyfile(source, target)
        shutil.copyfile(geometry, output / "native_geometry.npz")
        shutil.copyfile(review_file, output / "bottom-review.json")
        with netCDF4.Dataset(target, "a") as ds:

            def trace_variable(name, dtype):
                return ds.createVariable(
                    name, dtype, ("depth", "lat", "lon"), zlib=True, complevel=1
                )

            ds.title = "WOA native points with explicitly reviewed regional/bottom assignments"
            ds.bottom_policy = review["policy"]
            ds.complete_wet_support = int(resolved[wet].all())
            for name, values in (
                ("ptemp", pt),
                ("sr", sr),
                ("ct", ct),
                ("salt", sr * 35 / 35.16504),
            ):
                original = np.ma.asarray(ds[name][0]).copy()
                changed = fill.transpose(2, 1, 0)
                original[changed] = values.transpose(2, 1, 0)[changed]
                ds[name][0] = original
            ds["unresolved_support"][:] = (wet & ~resolved).transpose(2, 1, 0).astype(np.int8)
            for name, values in (
                ("pre_bottom_unresolved_support", wet & ~known),
                ("bottom_assignment", fill),
                ("regional_profile_assignment", regional),
                ("same_column_bottom_extension", fill & ~regional),
                ("linear_sensitivity_support", candidate),
            ):
                trace_variable(name, "i1")[:] = values.transpose(2, 1, 0).astype(np.int8)
            donors = result["donor_native_level_index"]
            # Regional assignments have their own evidence, not a claimed
            # vertical donor. Linear sensitivity still traces native anchors.
            fallback_donors = np.where(fill & ~regional, donors, -1)
            trace = {
                "bottom_donor_native_level_index": fallback_donors,
                "bottom_review_evidence_index": evidence_index,
                "linear_sensitivity_anchor_native_level_index": donors,
                "linear_sensitivity_second_anchor_native_level_index": second_donors,
            }
            for name, values in trace.items():
                trace_variable(name, "i8")[:] = values.transpose(2, 1, 0)
            distance = trace_variable("bottom_extension_distance_m", "f8")
            distance.units = "m"
            distance[:] = np.where(fill & ~regional, result["extension_distance_m"], 0).transpose(
                2, 1, 0
            )
            for name in (
                "nearest_original_anchor_flat_index",
                "nearest_anchor_path_distance_m",
                "original_paired_support",
                "horizontal_infill",
                "deep_extension",
            ):
                inherited = np.take_along_axis(masks[name], np.maximum(donors, 0), axis=-1)
                values = np.where(fill & ~regional, inherited, -1)
                variable = trace_variable(
                    "bottom_donor_" + name, "f8" if "distance" in name else "i8"
                )
                variable[:] = values.transpose(2, 1, 0)
        np.savez_compressed(
            output / "bottom-sensitivity.npz",
            targets=np.argwhere(fill),
            candidate_supported=candidate[fill],
            linear_pt=np.where(candidate, alternate_pt, np.nan)[fill],
            linear_sr=np.where(candidate, alternate_sr, np.nan)[fill],
            linear_ct=np.where(candidate, alternate_ct, np.nan)[fill],
            selected_pt=pt[fill],
            selected_sr=sr[fill],
            selected_ct=ct[fill],
            volume_m3=volume[fill],
            donor_native_level_index=result["donor_native_level_index"][fill],
            second_donor_native_level_index=second_donors[fill],
            extension_distance_m=result["extension_distance_m"][fill],
        )
        before_v = float(volume[known].sum())
        after_v = float(volume[resolved].sum())
        delta_heat = (
            RHO_0
            * CP0_TEOS10
            * float(np.sum(volume[candidate] * (alternate_ct[candidate] - ct[candidate])))
        )
        delta_salt = (
            RHO_0
            / 1000
            * float(np.sum(volume[candidate] * (alternate_sr[candidate] - sr[candidate])))
        )
        report.update(
            complete_wet_support=bool(resolved[wet].all()),
            bottom_assigned_nodes=int(fill.sum()),
            bottom_assigned_columns=len(needed),
            regional_profile_nodes=int(regional.sum()),
            same_column_extension_nodes=int((fill & ~regional).sum()),
            unresolved_nodes=int((wet & ~resolved).sum()),
            maximum_fallback_extension_m=float(
                np.max(np.where(fill & ~regional, result["extension_distance_m"], 0))
            ),
            water_inventory={
                "geometry_volume_m3": float(np.sum(geom["area"] * geom["depth"])),
                "resolved_before_m3": before_v,
                "resolved_after_m3": after_v,
                "bottom_assigned_volume_m3": float(volume[fill].sum()),
                "bottom_assigned_reference_CT_enthalpy_J": RHO_0
                * CP0_TEOS10
                * float(np.sum(volume[fill] * ct[fill])),
                "bottom_assigned_reference_salt_kg": RHO_0
                / 1000
                * float(np.sum(volume[fill] * sr[fill])),
                "resolved_reference_CT_enthalpy_J": RHO_0
                * CP0_TEOS10
                * float(np.sum(volume[resolved] * ct[resolved])),
                "resolved_reference_salt_kg": RHO_0
                / 1000
                * float(np.sum(volume[resolved] * sr[resolved])),
                "scope": "reference_fixed_point_quadrature_not_MOM_layer_inventory",
            },
            sensitivity={
                "method": "two_deepest_resolved_native_points_linear_PT_SR_continuation",
                "qualifies_dynamics": False,
                "comparison": "common_supported_assigned_bottom_points_only",
                "candidate_supported_nodes": int(candidate.sum()),
                "candidate_outside_numeric_range_nodes": int((supported & ~candidate).sum()),
                "no_second_native_anchor_nodes": int((fill & ~supported).sum()),
                "common_volume_m3": float(volume[candidate].sum()),
                "delta_reference_CT_enthalpy_J": delta_heat,
                "delta_reference_salt_kg": delta_salt,
                "max_abs_PT_difference_degC": float(
                    np.max(np.abs(alternate_pt[candidate] - pt[candidate]), initial=0)
                ),
                "max_abs_SR_difference_g_kg": float(
                    np.max(np.abs(alternate_sr[candidate] - sr[candidate]), initial=0)
                ),
            },
        )
        for p, expected in identities.items():
            if sha256_file(p) != expected:
                raise ValueError("bottom completion source/review changed during preparation")
        if report["package_source_sha256"] != {
            n: sha256_file(p)
            for n, p in source_paths(source_root(__file__), production_source_modules()).items()
        }:
            raise ValueError("executed bottom completion package changed")
        report.update(
            status="native_points_completed_with_bottom_assumptions"
            if report["complete_wet_support"]
            else "blocked_non_bottom_native_support",
            output={
                "path": target.name,
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            },
            geometry_output={
                "path": "native_geometry.npz",
                "sha256": sha256_file(output / "native_geometry.npz"),
            },
            outputs={
                p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in output.iterdir()
                if p.is_file()
            },
        )
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", reason_type=type(error).__name__, reason=str(error))
        raise
    finally:
        (output / "native_initialization.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
    return report
