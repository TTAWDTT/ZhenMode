"""NumPy-only independent last-step physical mass/time/Q/work audit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_physical_kinetic import angular_gram

ROOT = Path(__file__).resolve().parents[3]
OMEGA = 7.2921e-5


def adjacent(field, axis, direction):
    if axis == 0:
        return np.roll(field, -direction, axis=axis)
    return np.concatenate((field[:, 1:], np.zeros_like(field[:, :1])), axis=1) if direction > 0 else np.concatenate((np.zeros_like(field[:, :1]), field[:, :-1]), axis=1)


def matrices(data, volume):
    area, base = data["area"], data["thickness"]
    lat, lon = np.radians(data["latitude_edges"]), np.radians(data["longitude_edges"])
    dphi, dlon, dsin = np.diff(lat), np.diff(lon), np.diff(np.sin(lat))
    radius = np.sqrt(area[0, 0] / (dlon[0] * dsin[0]))
    if not np.allclose(area, radius ** 2 * dlon[:, None] * dsin[None, :], rtol=1e-12, atol=0.):
        raise ValueError("spherical area/coordinate identity")
    height = volume / area[..., None]
    if np.max(np.abs(height[..., 1:] - base[..., 1:]) / np.maximum(base[..., 1:], 1.)) > 1e-12:
        raise ValueError("lower layer geometry")
    top = np.broadcast_to(data["interfaces"][:-1], height.shape).copy()
    top[..., 0] = base[..., 0] - height[..., 0]
    bottom = top + height
    face_tops, face_bottoms, face_areas = [], [], []
    for axis in (0, 1):
        opened = (base > 0.) & (adjacent(base, axis, 1) > 0.)
        shared_top = np.maximum(top, adjacent(top, axis, 1))
        shared_bottom = np.minimum(bottom, adjacent(bottom, axis, 1))
        common = np.where(opened, np.maximum(shared_bottom - shared_top, 0.), 0.)
        width = radius * np.broadcast_to(dphi[None, :], area.shape) if axis == 0 else radius * dlon[:, None] * np.cos(lat[1:])[None, :]
        face_tops.append(shared_top)
        face_bottoms.append(shared_bottom)
        face_areas.append(width[..., None] * common)
    tops = np.stack((adjacent(face_tops[0], 0, -1), face_tops[0], adjacent(face_tops[1], 1, -1), face_tops[1]), axis=-1)
    bottoms = np.stack((adjacent(face_bottoms[0], 0, -1), face_bottoms[0], adjacent(face_bottoms[1], 1, -1), face_bottoms[1]), axis=-1)
    opened = np.stack((adjacent(face_areas[0], 0, -1), face_areas[0], adjacent(face_areas[1], 1, -1), face_areas[1]), axis=-1) > 0.
    overlap = np.maximum(np.minimum(bottoms[..., :, None], bottoms[..., None, :]) - np.maximum(tops[..., :, None], tops[..., None, :]), 0.)
    overlap = np.where(opened[..., :, None] & opened[..., None, :], overlap, 0.)
    angular_mass, angular_spin = np.zeros(area.shape + (4, 4)), np.zeros(area.shape + (4, 4))
    nodes, weights = np.polynomial.legendre.leggauss(48)
    mu, weights = (nodes + 1.) * .5, weights * .5
    for position in np.ndindex(area.shape):
        lon_index, lat_index = position
        south, north = lat[lat_index:lat_index + 2]
        phi = np.arcsin(np.sin(south) + mu * (np.sin(north) - np.sin(south)))
        beta = dphi[lat_index] / dlon[lon_index] * (mu - (phi - south) / dphi[lat_index]) / np.cos(phi)
        basis = np.stack((-beta, beta, (1. - mu) * np.cos(south) / np.cos(phi), mu * np.cos(north) / np.cos(phi)), axis=-1)
        mean = np.sum(weights[:, None] * (2. * OMEGA * np.sin(phi))[:, None] * basis, axis=0)
        cross = np.outer([.5, .5, 0., 0.], mean)
        angular_spin[position] = cross - cross.T
        angular_mass[position] = angular_gram(south, dphi[lat_index], dlon[lon_index], 48)
    factor = area[..., None, None, None] * overlap
    return factor * angular_mass[..., None, :, :], factor * angular_spin[..., None, :, :], face_areas


def action(matrix, velocity, areas):
    east, north = (np.where(area > 0., field, 0.) for area, field in zip(areas, velocity))
    local = np.stack((adjacent(east, 0, -1), east, adjacent(north, 1, -1), north), axis=-1)
    applied = np.einsum("...ab,...b->...a", matrix, local)
    return applied[..., 1] + adjacent(applied[..., 0], 0, 1), applied[..., 3] + adjacent(applied[..., 2], 1, 1)


def dot(left, right):
    return sum(np.sum(first * second) for first, second in zip(left, right))


def norm(fields):
    return np.sqrt(dot(fields, fields))


def kinetic(matrix, velocity, areas):
    return .5 * dot(velocity, action(matrix, velocity, areas))


def audit(data, dt=60.):
    mass, spin, areas = matrices(data, data["previous_volume"])
    previous = (data["previous_east_velocity"].astype(np.float64), data["previous_north_velocity"].astype(np.float64))
    uncast = (data["uncast_east_velocity"], data["uncast_north_velocity"])
    stored = (data["final_east_velocity"].astype(np.float64), data["final_north_velocity"].astype(np.float64))
    mean = tuple(data[name] / np.where(area > 0., area, 1.) for name, area in zip(("mean_east_flux", "mean_north_flux"), areas))
    host_height = data["previous_volume"][..., 0] / data["area"]
    host_eta = host_height - data["thickness"][..., 0]
    eta0 = data["initial_eta"]
    coordinate_floor = 64. * np.finfo(float).eps * (np.abs(host_height) + np.abs(data["thickness"][..., 0]))
    if np.any(np.abs(eta0 - host_eta) > coordinate_floor):
        raise ValueError("actual starting eta geometry representation")
    eta1 = data["surface_eta"]
    mean_eta = data["mean_eta"]
    surface_gate = 1e-12 + 1e-12 * max(np.max(np.abs(eta0)), np.max(np.abs(eta1)))
    mean_eta_error = abs(np.sum(data["area"] * (mean_eta - eta0))) / np.sum(data["area"])
    if mean_eta_error > surface_gate:
        raise ValueError("zero-source global substep-mean eta")
    force = (data["force_east"], data["force_north"])
    for velocity in (previous, uncast, stored, mean):
        if any(np.any(field[area == 0.] != 0.) for field, area in zip(velocity, areas)):
            raise ValueError("closed normal velocity")
    increment = action(mass, tuple(final - initial for final, initial in zip(uncast, previous)), areas)
    rotation = action(spin, mean, areas)
    pressure = tuple(9.81 * area * (mean_eta - adjacent(mean_eta, axis, 1))[..., None] for axis, area in enumerate(areas))
    rhs_terms = (rotation, pressure, force)
    rhs = tuple(dt * sum(fields[axis] for fields in rhs_terms) for axis in (0, 1))
    momentum_residual = norm(tuple(left - right for left, right in zip(increment, rhs)))
    scale = norm(increment) + dt * sum(norm(fields) for fields in rhs_terms)
    state_scale = norm(action(mass, previous, areas)) + norm(action(mass, uncast, areas))
    momentum_gate = 1e-11 * scale + 64. * np.finfo(float).eps * state_scale
    if not np.isfinite(momentum_residual) or momentum_residual > momentum_gate:
        raise ValueError(f"aggregate midpoint momentum: {momentum_residual} > {momentum_gate}")
    horizontal = data["mean_east_flux"] - adjacent(data["mean_east_flux"], 0, -1) + data["mean_north_flux"] - adjacent(data["mean_north_flux"], 1, -1)
    surface_error = np.max(np.abs(eta1 - eta0 + dt * np.sum(horizontal, axis=-1) / data["area"]))
    if surface_error > surface_gate:
        raise ValueError("aggregate midpoint surface/Q equation")
    net = horizontal + data["vertical_flux"][..., 1:] - data["vertical_flux"][..., :-1]
    volume_error = np.max(np.abs((data["final_volume"] - data["previous_volume"] + dt * net) / data["area"][..., None]))
    if volume_error > surface_gate:
        raise ValueError("actual inventory shared Q")
    initial_energy = kinetic(mass, previous, areas) + .5 * 9.81 * np.sum(data["area"] * eta0 ** 2)
    final_energy = kinetic(mass, uncast, areas) + .5 * 9.81 * np.sum(data["area"] * eta1 ** 2)
    work = dt * dot(force, mean)
    energy_residual = final_energy - initial_energy - work
    energy_scale = abs(initial_energy) + abs(final_energy) + abs(work)
    if not np.isfinite(energy_residual) or abs(energy_residual) > 1e-11 * energy_scale:
        raise ValueError("independent frozen energy/work")
    cast_work = kinetic(mass, stored, areas) - kinetic(mass, uncast, areas)
    endpoint_mass, unused_spin, endpoint_areas = matrices(data, data["final_volume"])
    moving_work = kinetic(endpoint_mass, stored, endpoint_areas) - kinetic(mass, stored, areas)
    for name, expected in (("external_work", work), ("cast_work", cast_work), ("moving_mass_energy_change", moving_work)):
        tolerance = 1e-11 * abs(expected) + 64. * np.finfo(float).eps * energy_scale
        if abs(float(data[name]) - expected) > tolerance:
            raise ValueError(f"independent {name}")
    content_budget = np.abs(np.sum(data["final_content"] - data["initial_content"], axis=(0, 1, 2))) / np.sum(np.abs(data["initial_content"]), axis=(0, 1, 2))
    if np.max(content_budget) > 1e-12:
        raise ValueError("independent content inventory")
    return {"starting_eta_geometry_error_m": float(np.max(np.abs(eta0 - host_eta))),
            "mean_eta_global_error_m": float(mean_eta_error),
            "momentum_relative": float(momentum_residual / scale if scale > 0. else 0.),
            "surface_error_m": float(surface_error), "volume_equation_error_m": float(volume_error),
            "energy_work_relative": float(abs(energy_residual) / energy_scale if energy_scale > 0. else 0.),
            "cast_work_per_rho0": float(cast_work), "moving_mass_energy_per_rho0": float(moving_work)}


def verify(path, negative=False):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["status"] != "PASS" or len(report["runs"]) != 8 or not report.get("source_hashes_unchanged"):
        raise ValueError("paired report not complete/qualified")
    if report["kinetic_norm"] != "physical_wet_contact_horizontal_field_L2" or report["time_scheme"] != "simultaneous_implicit_midpoint_rotation_pressure_surface_mean_q":
        raise ValueError("paired method contract")
    provenance = report["provenance"]
    if provenance["git_status"]:
        raise ValueError("paired reference did not start at clean revision")
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value for name, value in provenance["source_sha256"].items()):
        raise ValueError("current source differs from paired runtime manifest")
    input_report = ROOT / provenance["input_report"]
    if hashlib.sha256(input_report.read_bytes()).hexdigest() != provenance["input_report_sha256"]:
        raise ValueError("paired real input report hash")
    combinations = {(run["geometry"], run["velocity_dtype"], run["disturbed"]) for run in report["runs"]}
    expected_combinations = {(geometry, dtype, disturbed) for geometry in ("prior_smoothed", "unsmoothed")
                             for dtype in ("float64", "float32") for disturbed in (False, True)}
    if combinations != expected_combinations:
        raise ValueError("paired real group coverage")
    evidence, negatives = [], []
    for run in report["runs"]:
        endpoint = ROOT / run["snapshot_path"]
        if hashlib.sha256(endpoint.read_bytes()).hexdigest() != run["snapshot_sha256"]:
            raise ValueError("paired snapshot hash")
        if run["status"] != "PASS" or run["completed_steps"] != run["requested_steps"] or len(run["history"]) != run["completed_steps"] or any(not row["accepted"] for row in run["history"]):
            raise ValueError("paired recorded history not qualified")
        immutable_input = ROOT / run["input_snapshot"]
        if hashlib.sha256(immutable_input.read_bytes()).hexdigest() != run["input_snapshot_sha256"]:
            raise ValueError("paired immutable real input hash")
        for index, row in enumerate(run["history"]):
            if row["step"] != index + 1 or any(value is None or not np.isfinite(value) for name, value in row.items() if name not in ("step", "accepted")):
                raise ValueError("paired finite ordered history")
            limits = {"solve_relative": 1e-12, "frozen_energy_work_relative": 1e-11,
                      "surface_error_m": 1e-12 + 1e-12 * row["eta_max_m"], "outflow_fraction": 1. + 1e-12,
                      "wet_bottom_relative": 1e-12 + 64. * np.finfo(float).eps,
                      "dual_commutation_relative": 1e-12 + 64. * np.finfo(float).eps,
                      "bulk_continuity_relative": 1e-12 + 64. * np.finfo(float).eps,
                      "constant_error": 1e-12, "bound_excursion": 1e-12}
            if any(not 0. <= row[name] <= limit for name, limit in limits.items()):
                raise ValueError("paired recorded gate violation")
        with np.load(endpoint, allow_pickle=False) as snapshot:
            data = {name: snapshot[name] for name in snapshot.files}
        if any(not np.all(np.isfinite(array)) for array in data.values()):
            raise ValueError("nonfinite paired snapshot")
        evidence.append({"geometry": run["geometry"], "dtype": run["velocity_dtype"], "disturbed": run["disturbed"], **audit(data)})
        if negative and run["disturbed"] and run["velocity_dtype"] == "float64" and not negatives:
            for name in ("uncast_east_velocity", "mean_east_flux", "initial_eta", "mean_eta", "force_east", "final_volume", "final_content", "external_work", "latitude_edges"):
                corrupted = {key: value.copy() for key, value in data.items()}
                corrupted[name] = corrupted[name] + .01 if name in ("mean_eta", "initial_eta") else corrupted[name] * 1.01
                try:
                    audit(corrupted)
                except ValueError:
                    negatives.append(name)
                else:
                    raise ValueError(f"corruption accepted: {name}")
    return {"status": "PASS", "scope": "independent_last_step_aggregate_not_every_substep_or_full_ocean",
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "quadrature_helper_sha256": hashlib.sha256(Path(__file__).with_name("analyze_physical_kinetic.py").read_bytes()).hexdigest(),
            "runs": evidence, "negative_controls_rejected": negatives}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = verify(ROOT / args.report, args.negative_controls)
    (ROOT / args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
