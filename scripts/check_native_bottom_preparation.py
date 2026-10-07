"""Independent structural check of native completion; no product helpers.

Shared layers: recorded inputs, NetCDF/NumPy readers and declared constants.
This checks preservation, explicit assignment/trace, fixed-cell quadrature
and two-anchor arithmetic. It does not independently certify thermodynamic
conversion, source geography, regional selection or dynamical sensitivity.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import netCDF4
import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def read_fields(path):
    with netCDF4.Dataset(path) as data:
        fields = {
            n: np.ma.asarray(data[n][0], dtype=float).filled(np.nan).transpose(2, 1, 0)
            for n in ("ptemp", "sr", "ct", "salt")
        }
        trace = {
            n: np.asarray(data[n][:]).transpose(2, 1, 0)
            for n in data.variables
            if data[n].dimensions == ("depth", "lat", "lon")
        }
    return fields, trace


def assignments(before, after, trace, policy, depths, wet):
    known = np.isfinite(before["ptemp"]) & np.isfinite(before["sr"]) & wet
    missing = wet & ~known
    for name in before:
        # Float64 bit patterns are preserved, including signed zero.
        np.testing.assert_array_equal(
            before[name][known].view(np.uint64), after[name][known].view(np.uint64)
        )
        assert np.isfinite(after[name][wet]).all(), "a wet point is still missing"
    np.testing.assert_array_equal(trace["bottom_assignment"] > 0, missing)
    np.testing.assert_array_equal(trace["pre_bottom_unresolved_support"] > 0, missing)
    assert not trace["unresolved_support"].any()
    regional = {}
    for row in policy["columns"]:
        i, j = divmod(row["native_flat_index"], wet.shape[1])
        for p in row["regional_points"]:
            regional[i, j, p["node_index"]] = p
    for i, j, k in np.argwhere(missing):
        key = (i, j, k)
        if key in regional:
            assert trace["regional_profile_assignment"][key] == 1
            assert after["ptemp"][key] == regional[key]["ptemp"]
            assert after["sr"][key] == regional[key]["sr"]
            assert trace["bottom_donor_native_level_index"][key] == -1
        else:
            # Re-derive the donor by walking the source column, without the
            # producer's array-max, fill-mask or take-along-axis code.
            anchor = -1
            for level in range(len(depths)):
                if known[i, j, level]:
                    anchor = level
            assert 0 <= anchor < k
            assert trace["same_column_bottom_extension"][key] == 1
            assert trace["bottom_donor_native_level_index"][key] == anchor
            assert trace["bottom_extension_distance_m"][key] == depths[k] - depths[anchor]
            for name in ("ptemp", "sr"):
                assert after[name][key] == before[name][i, j, anchor]
    return known, missing


def verify(prepared, completed, output):
    prepared, completed = Path(prepared), Path(completed)
    parent = json.loads((prepared / "native_initialization.json").read_text())
    report = json.loads((completed / "native_initialization.json").read_text())
    assert report["parent_native_receipt_sha256"] == digest(
        prepared / "native_initialization.json"
    ), "supplied native parent receipt does not match completion"
    assert report["status"] == "native_points_completed_with_bottom_assumptions"
    for name, row in report["outputs"].items():
        assert digest(completed / name) == row["sha256"]
    assert digest(prepared / parent["output"]["path"]) == parent["output"]["sha256"]
    before, original_trace = read_fields(prepared / parent["output"]["path"])
    after, trace = read_fields(completed / report["output"]["path"])
    policy = json.loads((completed / "bottom-review.json").read_text())
    with np.load(completed / "native_geometry.npz") as data:
        geometry = {k: data[k].copy() for k in data.files}
    depth = geometry["node_depth_m"]
    bed = geometry["depth"]
    wet = (depth[None, None, :] <= bed[..., None]) & (bed[..., None] > 0)
    np.testing.assert_array_equal(wet, trace["wet_node_mask"] > 0)
    np.testing.assert_array_equal(trace["wet_node_mask"], original_trace["wet_node_mask"])
    known, missing = assignments(before, after, trace, policy, depth, wet)
    h = np.zeros_like(wet, dtype=float)
    for k in range(len(depth)):
        top = 0 if k == 0 else (depth[k - 1] + depth[k]) / 2
        following = wet[..., k + 1] if k + 1 < len(depth) else np.zeros(bed.shape, dtype=bool)
        bottom = np.where(
            following, (depth[k] + depth[k + 1]) / 2 if k + 1 < len(depth) else depth[k], bed
        )
        h[..., k] = np.where(wet[..., k], bottom - top, 0)
    np.testing.assert_array_equal(h, geometry["thickness_m"])
    np.testing.assert_allclose(h.sum(axis=-1), bed, rtol=0, atol=2e-12)
    area = (
        6371000.0**2
        * np.deg2rad(np.diff(geometry["lon_bounds"], axis=1))
        * np.diff(np.sin(np.deg2rad(geometry["lat_bounds"])), axis=1).T
    )
    np.testing.assert_allclose(area, geometry["area"], rtol=1e-12, atol=1e-6)
    volume = area[..., None] * h
    inventory = {
        "resolved_after_m3": math.fsum(volume[wet].flat),
        "bottom_assigned_volume_m3": math.fsum(volume[missing].flat),
        "resolved_reference_CT_enthalpy_J": 1025.0
        * 3991.86795711963
        * math.fsum((volume[wet] * after["ct"][wet]).flat),
        "resolved_reference_salt_kg": 1.025 * math.fsum((volume[wet] * after["sr"][wet]).flat),
    }
    for name, value in inventory.items():
        assert math.isclose(value, report["water_inventory"][name], rel_tol=1e-12, abs_tol=1e-12), (
            name
        )
    with np.load(completed / "bottom-sensitivity.npz") as data:
        for n, (i, j, k) in enumerate(data["targets"]):
            if not data["candidate_supported"][n]:
                continue
            indices = [level for level in range(k) if known[i, j, level]]
            lower, upper = indices[-2:]
            assert data["donor_native_level_index"][n] == upper
            assert data["second_donor_native_level_index"][n] == lower
            for tracer, field in (("ptemp", "linear_pt"), ("sr", "linear_sr")):
                a, b = before[tracer][i, j, lower], before[tracer][i, j, upper]
                expected = b + (b - a) * (depth[k] - depth[upper]) / (depth[upper] - depth[lower])
                assert math.isclose(data[field][n], expected, rel_tol=1e-12, abs_tol=1e-12)
    controls = []
    altered = {n: a.copy() for n, a in after.items()}
    altered["ptemp"][tuple(np.argwhere(known)[0])] += 1
    try:
        assignments(before, altered, trace, policy, depth, wet)
    except AssertionError:
        controls.append("changed_resolved_value_refused")
    else:
        raise AssertionError("preservation comparator passed planted change")
    if missing.any():
        altered = {n: a.copy() for n, a in after.items()}
        altered["ptemp"][tuple(np.argwhere(missing)[0])] = np.nan
        try:
            assignments(before, altered, trace, policy, depth, wet)
        except AssertionError:
            controls.append("missing_wet_value_refused")
        else:
            raise AssertionError("support comparator passed planted missing value")
    if trace["same_column_bottom_extension"].any():
        planted = {n: a.copy() for n, a in trace.items()}
        planted["bottom_donor_native_level_index"][
            tuple(np.argwhere(planted["same_column_bottom_extension"])[0])
        ] += 1
        try:
            assignments(before, after, planted, policy, depth, wet)
        except AssertionError:
            controls.append("wrong_native_donor_refused")
        else:
            raise AssertionError("donor comparator passed planted wrong donor")
    result = {
        "status": "structural_and_quadrature_checks_passed",
        "assigned_nodes": int(missing.sum()),
        "preserved_resolved_nodes": int(known.sum()),
        "inventory": inventory,
        "negative_controls": controls,
        "evaluator_sha256": digest(__file__),
        "completed_receipt_sha256": digest(completed / "native_initialization.json"),
        "relative_inventory_tolerance": 1e-12,
        "source_geography_certified": False,
        "thermodynamics_independently_checked": False,
        "dynamical_sensitivity_qualified": False,
        "lineage": "no product imports; shared input data, NumPy/NetCDF readers and declared constants",
    }
    with Path(output).open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-prepared", required=True)
    parser.add_argument("--completed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.native_prepared, args.completed, args.output)
