"""Independent host frame/metric audit of last actual-Q snapshots, not ALE replay."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from verify_reference import _path
from verify_wet_trace_reference import adjacent, verify_wet

FLOOR = 1e-12 + 64. * np.finfo(np.float64).eps
CONTRACT = "mean_q_frozen_eulerian_lift_signed_top_boundary_source_not_full_ale"


def equal(actual, expected, scale):
    if actual.shape != expected.shape or not np.all(np.isfinite(actual)):
        raise ValueError("physical field shape or finiteness mismatch")
    if not np.all(np.abs(actual - expected) <= FLOOR * scale):
        raise ValueError("physical field differs from actual-Q spherical frame")


def verify_physical(data, dt):
    area, volume = data["area"], data["last_previous_volume"]
    height = volume / area[..., None]
    latitude = np.radians(data["latitude_edges"])
    delta_lat = np.diff(latitude)
    sine_width = np.diff(np.sin(latitude))
    delta_lon = np.radians(np.diff(data["longitude_edges"]))
    radius_squared = area / (delta_lon[:, None] * sine_width[None, :])
    if not np.all(np.isfinite(radius_squared)) or np.any(radius_squared <= 0.):
        raise ValueError("invalid independent spherical metric")
    radius = np.sqrt(radius_squared[0, 0])
    equal(radius_squared, np.full_like(area, radius ** 2), radius_squared)
    if latitude[0] <= -.5 * np.pi or latitude[-1] >= .5 * np.pi:
        raise ValueError("unsupported pole")
    expected_metrics = {"area": area[..., None],
                        "meridional_width": np.broadcast_to(radius * sine_width[None, :, None], area.shape + (1,)),
                        "zonal_arc": np.broadcast_to(radius * delta_lon[:, None, None], area.shape + (1,))}
    for name, expected in expected_metrics.items():
        actual = data[f"last_physical_{name}"]
        if actual.dtype != np.float64:
            raise ValueError("physical metric precision changed")
        equal(actual, expected, np.abs(expected))
    top = np.broadcast_to(data["interfaces"][:-1], height.shape).copy()
    top[..., 0] = data["thickness"][..., 0] - height[..., 0]
    east, north, vertical = (data[f"last_{name}_flux"] for name in ("east", "north", "vertical"))
    net = east - adjacent(east, 0, -1) + north - adjacent(north, 1, -1) + vertical[..., 1:] - vertical[..., :-1]
    scale_flux = (np.abs(east) + np.abs(adjacent(east, 0, -1)) + np.abs(north)
                  + np.abs(adjacent(north, 1, -1)) + np.abs(vertical[..., 1:]) + np.abs(vertical[..., :-1]))
    source = data["last_volume_source"]
    if (source.shape != volume.shape or source.dtype != np.float64 or not np.all(np.isfinite(source))
            or np.any(source[..., 1:] != 0.) or np.any(source[height <= 0.] != 0.)):
        raise ValueError("unsupported or malformed physical source")
    delta = data["final_volume"] - volume
    residual = delta + dt * (net - source)
    tolerance = 1e-12 * dt * (scale_flux + np.abs(source)) + 64. * np.finfo(np.float64).eps * (np.abs(volume) + np.abs(data["final_volume"]))
    if np.any(np.abs(residual) > tolerance) or np.any(np.abs(net[..., 1:]) > FLOOR * scale_flux[..., 1:]):
        raise ValueError("actual local continuity or lower bulk incompressibility failed")
    lift = np.zeros_like(volume)
    lift[..., 0] = net[..., 0] / area
    source_speed = source / area[..., None]
    surface = lift - source_speed
    for name, expected, scale in (("absolute_lift", lift, scale_flux / area[..., None]),
                                  ("source_speed", source_speed, np.abs(source_speed)),
                                  ("surface_downward", surface, (scale_flux + np.abs(source)) / area[..., None])):
        actual = data[f"last_physical_{name}"]
        if actual.dtype != np.float64:
            raise ValueError("physical lift/source precision changed")
        equal(actual, expected, scale)
    wet = height > 0.
    safe_height = np.where(wet, height, 1.)
    side_top, side_bottom, density = (data[f"last_trace_{name}"] for name in ("side_top", "side_bottom", "side_density"))
    fraction, longitude = .61, .37
    phi = np.arcsin(np.sin(latitude[:-1]) + fraction * sine_width)[None, :, None]
    arc_weight = sine_width[None, :, None] / (delta_lat[None, :, None] * np.cos(phi))
    primitive_latitude = (phi - latitude[:-1][None, :, None]) / delta_lat[None, :, None]
    worst = 0.
    for location, alpha in (("top", 0.), ("bottom", 1.), ("middle", .38123)):
        depth = top + alpha * height
        traces = np.where((depth[..., None] >= side_top) & (depth[..., None] <= side_bottom) & (side_bottom > side_top), density, 0.)
        primitives = density * np.clip(depth[..., None] - side_top, 0., side_bottom - side_top)
        mapped_vertical = (vertical[..., :-1] + alpha * net - primitives[..., 1] + primitives[..., 0]
                           - primitives[..., 3] + primitives[..., 2])
        east_velocity = ((1. - longitude) * traces[..., 0] + longitude * traces[..., 1]) / (radius * delta_lat[None, :, None])
        north_velocity = ((1. - fraction) * traces[..., 2] + fraction * traces[..., 3]
                          + (traces[..., 1] - traces[..., 0]) * (fraction - primitive_latitude)) / (radius * delta_lon[:, None, None] * np.cos(phi))
        fields = {"east": east_velocity, "north": north_velocity,
                  "downward": mapped_vertical / area[..., None] + lift * (1. - alpha),
                  "grid_downward": surface * (1. - alpha),
                  "relative_downward": mapped_vertical / area[..., None] + source_speed * (1. - alpha),
                  "divergence": net / (safe_height * area[..., None]) - lift / safe_height}
        velocity_scale = (np.sum(np.abs(traces), axis=-1) / np.minimum(radius * delta_lat[None, :, None], radius * delta_lon[:, None, None] * np.cos(phi))
                          + (np.abs(vertical[..., :-1]) + scale_flux + np.abs(source)) / area[..., None])
        divergence_scale = (scale_flux + np.abs(net)) / (safe_height * area[..., None])
        for name, expected in fields.items():
            expected = np.where(wet, expected, 0.)
            actual = data[f"last_physical_{location}_{name}"]
            scale = np.where(wet, divergence_scale if name == "divergence" else velocity_scale, 0.)
            if actual.dtype != np.float64:
                raise ValueError("physical point precision changed")
            equal(actual, expected, scale)
        flags = data[f"last_physical_{location}_valid"]
        if flags.dtype != np.bool_ or not np.array_equal(flags, wet):
            raise ValueError("accepted physical point mask differs from actual wet geometry")
        east_div = (traces[..., 1] - traces[..., 0]) / (radius ** 2 * delta_lat[None, :, None] * delta_lon[:, None, None] * np.cos(phi))
        north_div = (traces[..., 3] - traces[..., 2] + (traces[..., 1] - traces[..., 0]) * (1. - arc_weight)) / area[..., None]
        down_div = (net / safe_height - traces[..., 1] + traces[..., 0] - traces[..., 3] + traces[..., 2]) / area[..., None] - lift / safe_height
        divergence = np.where(wet, east_div + north_div + down_div, 0.)
        divisor = np.abs(east_div) + np.abs(north_div) + np.abs(down_div) + divergence_scale
        if np.any(np.abs(divergence) > FLOOR * divisor):
            raise ValueError("independent physical derivative divergence failed")
        worst = max(worst, float(np.max(np.abs(divergence) / np.where(divisor > 0., divisor, 1.))))
    return {"maximum_spherical_divergence_relative": worst, "actual_source_m3_s": float(np.sum(source)),
            "scope": "last_mean_q_on_frozen_geometry_not_endpoint_velocity_or_all_step_replay"}


def verify_physical_report(report):
    if report.get("physical_velocity_contract") != CONTRACT:
        raise ValueError("physical velocity contract absent")
    required = {"src/physical_velocity.py", "tests/test_physical_velocity.py",
                "research/experiments/cgrid_hydrostatic_momentum/physical_velocity_protocol.md"}
    if not required.issubset(report["provenance"]["source_sha256"]):
        raise ValueError("physical frame runtime manifest absent")
    result = verify_wet(report)
    for run, entry in zip(report["runs"], result):
        for row in run["history"]:
            if (row.get("physical_velocity_valid") is not True or not np.isfinite(row.get("physical_bulk_continuity_relative", np.nan))
                    or row["physical_bulk_continuity_relative"] > FLOOR):
                raise ValueError("rejected intermediate physical frame")
        with np.load(_path(run["snapshot_path"]), allow_pickle=False) as data:
            entry["last_physical_frame"] = verify_physical(data, run["dt_seconds"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="results/industrial_alignment/cgrid_physical_velocity_reference.json")
    parser.add_argument("--negative-controls", action="store_true")
    args = parser.parse_args()
    report = json.loads(_path(args.report).read_text(encoding="utf-8"))
    print(json.dumps({"verified": verify_physical_report(report), "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
    if args.negative_controls:
        changed = copy.deepcopy(report)
        changed["runs"][0]["history"][0]["physical_velocity_valid"] = False
        try:
            verify_physical_report(changed)
        except (ValueError, AssertionError):
            print("negative physical stage: rejected")
        else:
            raise ValueError("physical stage corruption accepted")
        with np.load(_path(report["runs"][1]["snapshot_path"]), allow_pickle=False) as data:
            original = {name: data[name].copy() for name in data.files}
        for field in ("last_physical_area", "last_physical_meridional_width", "last_physical_zonal_arc",
                      "last_volume_source", "last_physical_absolute_lift", "last_physical_surface_downward",
                      "last_physical_top_downward", "last_physical_middle_relative_downward",
                      "last_physical_middle_grid_downward", "last_physical_middle_divergence", "last_physical_middle_valid"):
            changed = {name: value.copy() for name, value in original.items()}
            if field.endswith("valid"):
                changed[field] = ~changed[field]
            elif field == "last_volume_source":
                changed[field][..., 0] += 1e5
            elif field in ("last_physical_area", "last_physical_meridional_width", "last_physical_zonal_arc"):
                changed[field] *= 1.001
            else:
                changed[field] += 1e-3
            try:
                verify_physical(changed, report["runs"][1]["dt_seconds"])
            except (ValueError, AssertionError):
                print(f"negative physical {field}: rejected")
            else:
                raise ValueError(f"physical corruption accepted: {field}")


if __name__ == "__main__":
    main()
