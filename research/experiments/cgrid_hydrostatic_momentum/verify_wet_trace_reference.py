"""Host wet-interval/primitive oracle, in addition to actual-Q inventory audit."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from verify_reference import ROOT, _path, verify


def adjacent(field, axis, direction):
    if axis == 0:
        return np.roll(field, -direction, axis=0)
    if direction > 0:
        return np.concatenate((field[:, 1:], np.zeros_like(field[:, :1])), axis=1)
    return np.concatenate((np.zeros_like(field[:, :1]), field[:, :-1]), axis=1)


def assert_flux_equal(actual, expected, scale):
    if not np.all(np.abs(actual - expected) <= (1e-12 + 64. * np.finfo(np.float64).eps) * scale):
        raise ValueError("wet trace flux/primitive mismatch")


def verify_trace(data, metric=False):
    volume, area, base = data["last_previous_volume"], data["area"], data["thickness"]
    height = volume / area[..., None]
    top = np.broadcast_to(data["interfaces"][:-1], height.shape).copy()
    top[..., 0] = base[..., 0] - height[..., 0]
    bottom = top + height
    wet = base > 0.
    east, north, vertical = (data[f"last_{name}_flux"] for name in ("east", "north", "vertical"))
    tops, bottoms, opened = [], [], []
    for axis in (0, 1):
        start = np.maximum(top, adjacent(top, axis, 1))
        end = np.minimum(bottom, adjacent(bottom, axis, 1))
        opens = wet & adjacent(wet, axis, 1) & (end > start)
        tops.extend((adjacent(start, axis, -1), start))
        bottoms.extend((adjacent(end, axis, -1), end))
        opened.extend((adjacent(opens, axis, -1), opens))
    side_top = np.stack([np.where(opens, start, top) for opens, start in zip(opened, tops)], axis=-1)
    side_bottom = np.stack([np.where(opens, end, top) for opens, end in zip(opened, bottoms)], axis=-1)
    side_flux = np.stack((adjacent(east, 0, -1), east, adjacent(north, 1, -1), north), axis=-1)
    width = side_bottom - side_top
    density = np.where(width > 0., side_flux / np.where(width > 0., width, 1.), 0.)
    net = east - adjacent(east, 0, -1) + north - adjacent(north, 1, -1) + vertical[..., 1:] - vertical[..., :-1]
    scale = np.sum(np.abs(side_flux), axis=-1) + np.abs(vertical[..., :-1]) + np.abs(vertical[..., 1:])
    names = ("cell_top", "cell_height", "side_top", "side_bottom", "side_density", "net_flux", "evaluated_top",
             "evaluated_bottom", "evaluated_middle_east", "evaluated_middle_north", "evaluated_middle_vertical")
    for name in names:
        field = data[f"last_trace_{name}"]
        shape = height.shape + (4,) if name.startswith("side_") else height.shape
        if field.shape != shape or field.dtype != np.float64 or not np.all(np.isfinite(field)):
            raise ValueError("wet trace snapshot shape/precision/finiteness mismatch")
    for name, expected in (("cell_top", top), ("cell_height", height), ("side_top", side_top), ("side_bottom", side_bottom)):
        np.testing.assert_allclose(data[f"last_trace_{name}"], expected, rtol=1e-12, atol=1e-12)
    assert_flux_equal(data["last_trace_side_density"] * width, side_flux, np.abs(side_flux))
    if np.any(data["last_trace_side_density"][width == 0.] != 0.):
        raise ValueError("closed wet trace density is nonzero")
    assert_flux_equal(data["last_trace_net_flux"], net, scale)
    for name, expected in (("evaluated_top", vertical[..., :-1]), ("evaluated_bottom", vertical[..., 1:])):
        assert_flux_equal(data[f"last_trace_{name}"], expected, scale)
    middle = top + .38123 * height
    traces = np.where((width > 0.) & (middle[..., None] >= side_top) & (middle[..., None] <= side_bottom), density, 0.)
    east_point = .63 * traces[..., 0] + .37 * traces[..., 1]
    north_point = .39 * traces[..., 2] + .61 * traces[..., 3]
    if metric:
        latitude = np.radians(data["latitude_edges"])
        width_latitude = np.diff(latitude)[None, :, None]
        sine_width = np.diff(np.sin(latitude))[None, :, None]
        phi = np.arcsin(np.sin(latitude[:-1])[None, :, None] + .61 * sine_width)
        weight = sine_width / (width_latitude * np.cos(phi))
        primitive_latitude = (phi - latitude[:-1][None, :, None]) / width_latitude
        east_point = weight * east_point
        north_point += (traces[..., 1] - traces[..., 0]) * (.61 - primitive_latitude)
        for direction, expected in (("south", latitude[:-1][None, :, None]), ("north", latitude[1:][None, :, None])):
            observed = data[f"last_trace_{direction}_latitude"]
            if observed.shape != expected.shape or observed.dtype != np.float64 or not np.all(np.isfinite(observed)):
                raise ValueError("latitude metric snapshot shape/precision/finiteness mismatch")
            np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
    primitive = density * np.clip(middle[..., None] - side_top, 0., width)
    vertical_point = (vertical[..., :-1] + .38123 * net - primitive[..., 1] + primitive[..., 0] - primitive[..., 3] + primitive[..., 2])
    density_scale = np.sum(np.abs(density), axis=-1)
    for name, expected, amount in (("middle_east", east_point, density_scale), ("middle_north", north_point, density_scale),
                                  ("middle_vertical", vertical_point, scale)):
        assert_flux_equal(data[f"last_trace_evaluated_{name}"], np.where(wet, expected, 0.), amount)
    return {"maximum_bottom_error_m3_s": float(np.max(np.abs(data["last_trace_evaluated_bottom"] - vertical[..., 1:]))),
            "maximum_side_integral_error_m3_s": float(np.max(np.abs(data["last_trace_side_density"] * width - side_flux)))}


def verify_dual(data):
    volume = data["last_previous_volume"]
    latitude = np.radians(data["latitude_edges"])
    middle = .5 * (latitude[:-1] + latitude[1:])
    south = (np.sin(middle) - np.sin(latitude[:-1])) / np.diff(np.sin(latitude))

    def mass_map(field, direction):
        if direction == "east":
            return .5 * (field + np.roll(field, -1, axis=0))
        output = np.zeros((field.shape[0], field.shape[1] + 1, field.shape[2]))
        for row, fraction in enumerate(south):
            output[:, row] += fraction * field[:, row]
            output[:, row + 1] += (1. - fraction) * field[:, row]
        return output

    def net(east, north_full, vertical):
        return east - np.roll(east, 1, axis=0) + north_full[:, 1:] - north_full[:, :-1] + vertical[..., 1:] - vertical[..., :-1]

    east, north, vertical = (data[f"last_{name}_flux"] for name in ("east", "north", "vertical"))
    north_full = np.concatenate((np.zeros_like(north[:, :1]), north), axis=1)
    primary_net = net(east, north_full, vertical)
    scale = np.abs(east) + np.abs(adjacent(east, 0, -1)) + np.abs(north_full[:, 1:]) + np.abs(north_full[:, :-1]) + np.abs(vertical[..., 1:]) + np.abs(vertical[..., :-1])
    nodes, weights = np.polynomial.legendre.leggauss(24)
    fractions = south[:, None] * (.5 + .5 * nodes)
    phi = np.arcsin(np.sin(latitude[:-1])[:, None] + fractions * np.diff(np.sin(latitude))[:, None])
    integrand = np.diff(np.sin(latitude))[:, None] / (np.diff(latitude)[:, None] * np.cos(phi))
    arc_half = np.sum(integrand * weights * .5 * south[:, None], axis=-1)
    south_east = east * arc_half[None, :, None]
    north_east = np.pad(south_east, ((0, 0), (0, 1), (0, 0))) + np.pad(east - south_east, ((0, 0), (1, 0), (0, 0)))
    center_north = ((1. - south[None, :, None]) * north_full[:, :-1] + south[None, :, None] * north_full[:, 1:]
                    + (south[None, :, None] - .5) * (east - adjacent(east, 0, -1)))
    candidates = {"east": tuple(mass_map(field, "east") for field in (east, north_full, vertical)),
                  "north": (north_east, np.pad(center_north, ((0, 0), (1, 1), (0, 0))), mass_map(vertical, "north"))}
    maximum_commutation = 0.
    for direction in ("east", "north"):
        mass = data[f"last_dual_{direction}_volume"]
        expected_mass = mass_map(volume, direction)
        if mass.shape != expected_mass.shape or mass.dtype != np.float64 or not np.all(np.isfinite(mass)):
            raise ValueError("half-prism mass shape/precision/finiteness mismatch")
        np.testing.assert_allclose(mass, expected_mass, rtol=1e-12, atol=1e-6)
        observed = []
        for axis, expected in zip(("east", "north", "vertical"), candidates[direction]):
            field = data[f"last_dual_{direction}_{axis}_flux"]
            if field.shape != expected.shape or field.dtype != np.float64 or not np.all(np.isfinite(field)):
                raise ValueError("dual flux shape/precision/finiteness mismatch")
            primal = {"east": east, "north": north_full, "vertical": vertical}[axis]
            flux_scale = mass_map(np.abs(primal), direction)
            if axis == "east" and direction == "north":
                flux_scale = np.pad(.5 * np.abs(east), ((0, 0), (0, 1), (0, 0))) + np.pad(.5 * np.abs(east), ((0, 0), (1, 0), (0, 0)))
            if axis == "north" and direction == "north":
                flux_scale = np.pad(scale, ((0, 0), (1, 1), (0, 0)))
            assert_flux_equal(field, expected, flux_scale)
            observed.append(field)
        expected_net = mass_map(primary_net, direction)
        residual = net(*observed) - expected_net
        assert_flux_equal(net(*observed), expected_net, mass_map(scale, direction))
        maximum_commutation = max(maximum_commutation, float(np.max(np.abs(residual))))
        increment = mass_map(data["final_volume"], direction) - mass
        transport = -60. * net(*observed)
        floor = 64. * np.finfo(float).eps * (np.abs(mass) + np.abs(mass_map(data["final_volume"], direction)))
        if np.any(np.abs(increment - transport) > 1e-12 * 60. * mass_map(scale, direction) + floor):
            raise ValueError("last dual local mass/Q increment mismatch")
    return {"maximum_absolute_commutation_m3_s": maximum_commutation, "scope": "last_spatial_dual_and_local_mass_not_nonlinear_momentum"}


def verify_wet(report):
    metric = report.get("wet_trace_contract") == "latitude_arc_wet_traces_paired_interior_flux_frozen_geometry_not_velocity"
    if not metric and report.get("wet_trace_contract") != "shared_wet_intervals_enriched_vertical_primitive_frozen_geometry_not_velocity":
        raise ValueError("wet trace formulation absent")
    required = {"src/wet_fluxes.py", "tests/test_wet_flux_reconstruction.py",
                "research/experiments/cgrid_hydrostatic_momentum/wet_trace_protocol.md"}
    if not required.issubset(report["provenance"]["source_sha256"]):
        raise ValueError("wet trace runtime sources absent")
    if metric:
        if report.get("dual_transport_contract") != "all_wet_half_prisms_metric_integrated_q_not_force_mass_or_nonlinear_momentum":
            raise ValueError("paired dual transport contract absent")
        required = {"research/experiments/cgrid_hydrostatic_momentum/metric_dual_protocol.md", "tests/test_wet_flux_metrics.py"}
        if not required.issubset(report["provenance"]["source_sha256"]):
            raise ValueError("metric/dual protocol and regression sources absent")
    result = verify(report)
    for run, entry in zip(report["runs"], result):
        if any(not row.get("wet_trace_valid") or row["wet_trace_bottom_relative"] > 1e-12 + 64. * np.finfo(np.float64).eps for row in run["history"]):
            raise ValueError("rejected intermediate wet trace")
        if metric and any(not row.get("dual_transport_valid") or row["dual_commutation_relative"] > 1e-12 + 64. * np.finfo(np.float64).eps for row in run["history"]):
            raise ValueError("rejected intermediate dual transport")
        with np.load(_path(run["snapshot_path"])) as data:
            entry["last_wet_trace"] = verify_trace(data, metric)
            if metric:
                entry["last_dual_transport"] = verify_dual(data)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="results/industrial_alignment/cgrid_metric_dual_reference.json")
    parser.add_argument("--negative-controls", action="store_true")
    args = parser.parse_args()
    report = json.loads((ROOT / args.report).read_text(encoding="utf-8"))
    metric = report.get("wet_trace_contract") == "latitude_arc_wet_traces_paired_interior_flux_frozen_geometry_not_velocity"
    print(json.dumps({"scope": "inventory_geometry_and_last_actual_q_wet_trace_snapshots_not_every_step_replay",
                      "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "verified": verify_wet(report)}, indent=2))
    if args.negative_controls:
        altered = copy.deepcopy(report)
        altered["runs"][0]["history"][0]["wet_trace_valid"] = False
        try:
            verify_wet(altered)
        except (ValueError, AssertionError):
            print("negative wet trace stage: rejected")
        else:
            raise ValueError("wet trace stage corruption accepted")
        with np.load(_path(report["runs"][1]["snapshot_path"])) as data:
            original = {name: data[name] for name in data.files}
        for label in ("density", "interval", "net_flux", "primitive"):
            altered = {name: field.copy() for name, field in original.items()}
            if label == "density":
                altered["last_trace_side_density"] += 1.
            elif label == "interval":
                altered["last_trace_side_bottom"] += .01
            elif label == "net_flux":
                altered["last_trace_net_flux"] += 1.
            else:
                altered["last_trace_evaluated_middle_vertical"] += 1.
            try:
                verify_trace(altered, metric)
            except (ValueError, AssertionError):
                print(f"negative wet trace {label}: rejected")
            else:
                raise ValueError(f"wet trace corruption accepted: {label}")
        if metric:
            altered = copy.deepcopy(report)
            altered["runs"][0]["history"][0]["dual_transport_valid"] = False
            try:
                verify_wet(altered)
            except (ValueError, AssertionError):
                print("negative dual stage: rejected")
            else:
                raise ValueError("dual stage corruption accepted")
            for label, field in (("metric", "last_trace_south_latitude"), ("mass", "last_dual_north_volume"),
                                 ("east_flux", "last_dual_north_east_flux"), ("north_flux", "last_dual_north_north_flux")):
                altered = {name: value.copy() for name, value in original.items()}
                altered[field] += .01 if label == "metric" else 1e8
                try:
                    verify_trace(altered, True)
                    verify_dual(altered)
                except (ValueError, AssertionError):
                    print(f"negative dual {label}: rejected")
                else:
                    raise ValueError(f"metric/dual corruption accepted: {label}")


if __name__ == "__main__":
    main()
