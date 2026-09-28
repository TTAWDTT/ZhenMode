"""Offline physical half-prism kinematics; NOT migrated nonlinear momentum."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from verify_reference import verify

ROOT = Path(__file__).resolve().parents[3]
EPS = np.finfo(np.float64).eps


def half_fractions(latitude_edges):
    latitude = np.radians(latitude_edges)
    middle = .5 * (latitude[:-1] + latitude[1:])
    south = (np.sin(middle) - np.sin(latitude[:-1])) / np.diff(np.sin(latitude))
    return south, 1. - south


def dual_map(field, component, south, north):
    if component == "east":
        return .5 * (field + np.roll(field, -1, axis=0))
    shape = (1, len(south)) + (1,) * (field.ndim - 2)
    output = np.zeros((field.shape[0], field.shape[1] + 1) + field.shape[2:])
    output[:, :-1] += field * south.reshape(shape)
    output[:, 1:] += field * north.reshape(shape)
    return output


def dual_flux(east, north_flux, vertical, component, south, north):
    north_full = np.concatenate((np.zeros_like(north_flux[:, :1]), north_flux), axis=1)
    if component == "east":
        return tuple(dual_map(field, component, south, north) for field in (east, north_full, vertical))
    shape = (1, len(south), 1)
    center_flux = north.reshape(shape) * north_full[:, :-1] + south.reshape(shape) * north_full[:, 1:]
    return dual_map(east, component, south, north), np.pad(center_flux, ((0, 0), (1, 1), (0, 0))), dual_map(vertical, component, south, north)


def divergence(east, north_full, vertical):
    return east - np.roll(east, 1, axis=0) + north_full[:, 1:] - north_full[:, :-1] + vertical[..., 1:] - vertical[..., :-1]


def divergence_size(east, north_full, vertical):
    return np.abs(east) + np.abs(np.roll(east, 1, axis=0)) + np.abs(north_full[:, 1:]) + np.abs(north_full[:, :-1]) + np.abs(vertical[..., 1:]) + np.abs(vertical[..., :-1])


def check(actual, expected, scale, floor):
    residual = np.abs(actual - expected)
    tolerance = 1e-12 * scale + floor
    return {"pass": bool(np.all(residual <= tolerance)), "maximum_residual": float(np.max(residual)),
            "maximum_tolerance_fraction": float(np.max(residual / np.where(tolerance > 0., tolerance, 1.)))}


def physical_oracle(volume, area, longitude_edges, latitude_edges):
    longitude, latitude = np.radians(longitude_edges), np.radians(latitude_edges)
    east = np.zeros_like(volume)
    north = np.zeros((volume.shape[0], volume.shape[1] + 1, volume.shape[2]))
    for column in range(volume.shape[0]):
        longitude_width = longitude[column + 1] - longitude[column]
        for row in range(volume.shape[1]):
            middle = .5 * (latitude[row] + latitude[row + 1])
            full_area = 6.371e6 ** 2 * longitude_width * (np.sin(latitude[row + 1]) - np.sin(latitude[row]))
            south_area = 6.371e6 ** 2 * longitude_width * (np.sin(middle) - np.sin(latitude[row]))
            north_area = 6.371e6 ** 2 * longitude_width * (np.sin(latitude[row + 1]) - np.sin(middle))
            for level in range(volume.shape[2]):
                height = volume[column, row, level] / area[column, row]
                east[column, row, level] += .5 * full_area * height
                east[(column - 1) % volume.shape[0], row, level] += .5 * full_area * height
                north[column, row, level] += south_area * height
                north[column, row + 1, level] += north_area * height
    return east, north


def contact_areas(volume, area, thickness, interfaces, longitude_edges, latitude_edges):
    height = volume / area[..., None]
    top = np.broadcast_to(interfaces[:-1], height.shape).copy()
    top[..., 0] = thickness[..., 0] - height[..., 0]
    bottom = top + height
    east_height = np.maximum(np.minimum(bottom, np.roll(bottom, -1, axis=0)) - np.maximum(top, np.roll(top, -1, axis=0)), 0.)
    north_height = np.maximum(np.minimum(bottom[:, :-1], bottom[:, 1:]) - np.maximum(top[:, :-1], top[:, 1:]), 0.)
    east_height = np.where((thickness > 0.) & (np.roll(thickness, -1, axis=0) > 0.), east_height, 0.)
    north_height = np.where((thickness[:, :-1] > 0.) & (thickness[:, 1:] > 0.), north_height, 0.)
    longitude, latitude = np.radians(np.diff(longitude_edges)), np.radians(latitude_edges)
    east = 6.371e6 * np.diff(latitude)[None, :, None] * east_height
    north = np.zeros((volume.shape[0], volume.shape[1] + 1, volume.shape[2]))
    north[:, 1:-1] = 6.371e6 * longitude[:, None, None] * np.cos(latitude[1:-1])[None, :, None] * north_height
    return (east, north), (east_height, north_height), height


def column_pairing(mass, contact, component):
    level = np.arange(mass.shape[-1])[None, None, :]
    phase = np.arange(mass.shape[0])[:, None, None] * .7 + np.arange(mass.shape[1])[None, :, None] * .3
    velocity = np.where(contact > 0., .04 * np.cos(phase) + .013 * level, 0.)
    mode = contact / np.where(mass > 0., mass, 1.)
    mobility = np.sum(contact * mode, axis=-1)
    transport = np.sum(contact * velocity, axis=-1)
    fast = mode * (transport / np.where(mobility > 0., mobility, 1.))[..., None]
    slow = velocity - fast
    cross = np.sum(mass * fast * slow, axis=-1)
    cross_scale = np.sum(np.abs(mass * fast * slow), axis=-1)
    energy = .5 * np.sum(mass * velocity ** 2, axis=-1)
    split_energy = .5 * np.sum(mass * (fast ** 2 + slow ** 2), axis=-1)
    orthogonal = check(cross, np.zeros_like(cross), cross_scale + energy, 64. * EPS * energy)
    decomposition = check(energy, split_energy, energy + split_energy, 64. * EPS * energy)
    uniform = transport / np.where(np.sum(contact, axis=-1) > 0., np.sum(contact, axis=-1), 1.)
    old_cross = np.sum(mass * uniform[..., None] * (velocity - uniform[..., None]), axis=-1)
    old_relative = np.abs(old_cross) / np.where(energy > 0., energy, 1.)
    primary_rows = mass.shape[1] - int(component == "north")
    head = np.sin(np.arange(mass.shape[0])[:, None] * .5 + np.arange(primary_rows)[None, :] * .8)
    head_difference = np.roll(head, -1, axis=0) - head if component == "east" else np.diff(head, axis=1)
    if component == "north":
        head_difference = np.pad(head_difference, ((0, 0), (1, 1)))
    layer_force = -mode * head_difference[..., None]
    actual_transport_force = np.sum(contact * layer_force, axis=-1)
    expected_transport_force = -mobility * head_difference
    force_check = check(actual_transport_force, expected_transport_force,
                        np.abs(actual_transport_force) + np.abs(expected_transport_force),
                        64. * EPS * (np.abs(actual_transport_force) + np.abs(expected_transport_force)))
    layer_work = np.sum(mass * velocity * layer_force, axis=-1)
    conjugate_work = -transport * head_difference
    work_check = check(layer_work, conjugate_work, np.abs(layer_work) + np.abs(conjugate_work),
                       64. * EPS * np.sum(np.abs(mass * velocity * layer_force), axis=-1))
    return {"orthogonality": orthogonal, "kinetic_decomposition": decomposition,
            "transport_force": force_check, "conjugate_pressure_work": work_check,
            "old_uniform_mode_maximum_energy_cross_relative": float(np.max(old_relative)),
            "mobility_minimum_open_m": float(np.min(mobility[mobility > 0.])) if np.any(mobility > 0.) else 0.}


def audit(data, dt, label):
    old_volume = data["last_previous_volume"] if "last_previous_volume" in data else data["initial_volume"]
    final_volume, area = data["final_volume"], data["area"]
    fluxes = tuple(data[("last_" if "last_previous_volume" in data else "") + direction + "_flux"] for direction in ("east", "north", "vertical"))
    source = data["volume_source"] if "volume_source" in data else np.zeros_like(old_volume)
    south, north = half_fractions(data["latitude_edges"])
    north_full = np.concatenate((np.zeros_like(fluxes[1][:, :1]), fluxes[1]), axis=1)
    primary_divergence = divergence(fluxes[0], north_full, fluxes[2])
    oracle_old = physical_oracle(old_volume, area, data["longitude_edges"], data["latitude_edges"])
    oracle_new = physical_oracle(final_volume, area, data["longitude_edges"], data["latitude_edges"])
    contacts, contact_height, height = contact_areas(old_volume, area, data["thickness"], data["interfaces"], data["longitude_edges"], data["latitude_edges"])
    directions, rejected_corruptions = {}, 0
    for position, component in enumerate(("east", "north")):
        old_mass, final_mass = (dual_map(field, component, south, north) for field in (old_volume, final_volume))
        np.testing.assert_allclose(old_mass, oracle_old[position], rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(final_mass, oracle_new[position], rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(np.sum(old_mass), np.sum(old_volume), rtol=1e-12)
        mapped_flux = dual_flux(*fluxes, component, south, north)
        mapped_divergence = divergence(*mapped_flux)
        expected_divergence = dual_map(primary_divergence, component, south, north)
        flux_scale = divergence_size(*mapped_flux) + dual_map(divergence_size(fluxes[0], north_full, fluxes[2]), component, south, north)
        commutation = check(mapped_divergence, expected_divergence, flux_scale, 64. * EPS * flux_scale)
        expected_change = dt * (dual_map(source, component, south, north) - mapped_divergence)
        amount = dt * (dual_map(np.abs(source), component, south, north) + divergence_size(*mapped_flux))
        mass_check = check(final_mass - old_mass, expected_change, amount, 64. * EPS * np.maximum(old_mass, final_mass))
        corrupted = [field.copy() for field in mapped_flux]
        corrupted[1][0, 1, 0] += 1e8
        if check(divergence(*corrupted), expected_divergence, flux_scale, 64. * EPS * flux_scale)["pass"]:
            raise ValueError("corrupted dual flux accepted")
        rejected_corruptions += 1
        directions[component] = {"geometry_oracle_pass": True, "all_dual_stock_fraction": float(np.sum(old_mass) / np.sum(old_volume)),
                                 "flux_commutation": commutation, "actual_mass_increment": mass_check,
                                 "column_pairing": column_pairing(old_mass, contacts[position], component)}
    east_fraction_left = contact_height[0] / np.where(height > 0., height, 1.)
    east_fraction_right = contact_height[0] / np.where(np.roll(height, -1, axis=0) > 0., np.roll(height, -1, axis=0), 1.)
    north_fraction_lower = contact_height[1] / np.where(height[:, :-1] > 0., height[:, :-1], 1.)
    north_fraction_upper = contact_height[1] / np.where(height[:, 1:] > 0., height[:, 1:], 1.)
    leakage = [np.abs(fluxes[0]) * np.maximum(1. - east_fraction_left, 1. - east_fraction_right),
               np.abs(fluxes[1][:, :-1]) * np.maximum(1. - north_fraction_lower, 1. - north_fraction_upper)]
    naive_leakage = {component: float(np.max(field)) for component, field in zip(("east", "north"), leakage)}
    pairing_pass = all(all(values["column_pairing"][gate]["pass"] for gate in ("orthogonality", "kinetic_decomposition", "transport_force", "conjugate_pressure_work")) for values in directions.values())
    kinematic_pass = pairing_pass and all(values["flux_commutation"]["pass"] and values["actual_mass_increment"]["pass"] for values in directions.values())
    return {"label": label, "directions": directions, "kinematic_and_offline_pairing_pass": kinematic_pass,
            "plain_rt0_unrepresented_wall_flux_max_m3_s": naive_leakage, "negative_flux_controls_rejected": rejected_corruptions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/industrial_alignment/staggered_support_selection.json")
    args = parser.parse_args()
    output = ROOT / args.out
    if output.exists():
        raise FileExistsError("retain old evidence; choose a new --out")
    reference_path = ROOT / "results/industrial_alignment/cgrid_momentum_flux_reference.json"
    rain_path = ROOT / "results/industrial_alignment/cgrid_moving_dual_mass_clean.json"
    reference, rain = (json.loads(path.read_text()) for path in (reference_path, rain_path))
    verify(reference)
    if rain["status"] != "FAIL" or not rain["controls_pass"] or len(rain["rows"]) != 4:
        raise ValueError("registered retained rain evidence missing")
    for relative, expected in rain["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("retained rain runtime source changed")
    rows = []
    for group in rain["rows"] + reference["runs"]:
        snapshot = ROOT / group["snapshot_path"]
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != group["snapshot_sha256"]:
            raise ValueError("input snapshot mismatch")
        label = (group["geometry"], group.get("forcing", group.get("velocity_dtype")), group.get("disturbed"))
        with np.load(snapshot) as stored:
            row = audit(stored, 60., label)
        row.update({"snapshot_path": group["snapshot_path"], "snapshot_sha256": group["snapshot_sha256"]})
        rows.append(row)
        print(label, "kinematics", row["kinematic_and_offline_pairing_pass"],
              "unqualified plain-RT0 wall flux", row["plain_rt0_unrepresented_wall_flux_max_m3_s"], flush=True)
    latitude = np.array([-60., -27., -3., 15., 56.])
    south, north = half_fractions(latitude)
    primary = np.zeros((2, 4, 1))
    primary[0, 1, 0] = 1.
    exact = dual_map(primary, "north", south, north)
    naive = dual_map(primary, "north", np.full(4, .5), np.full(4, .5))
    if check(naive, exact, np.abs(exact) + np.abs(naive), 64. * EPS * (np.abs(exact) + np.abs(naive)))["pass"]:
        raise ValueError("flat half-fraction control accepted")
    sources = [Path(__file__), Path(__file__).with_name("staggered_support_protocol.md"), Path(__file__).with_name("verify_reference.py")]
    passed = all(row["kinematic_and_offline_pairing_pass"] for row in rows)
    report = {"scope": "offline_full_half_prism_kinematics_and_column_pairing_not_implemented_momentum",
              "status": "KINEMATIC_PASS_PHYSICAL_RECONSTRUCTION_UNQUALIFIED" if passed else "FAIL",
              "plain_rt0_wall_condition_qualified": False, "nonlinear_momentum_implemented": False,
              "production_cutover": False, "rows": rows, "flat_latitude_control_rejected": True,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
              "input_report_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in (reference_path, rain_path)}}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(report["status"], flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
