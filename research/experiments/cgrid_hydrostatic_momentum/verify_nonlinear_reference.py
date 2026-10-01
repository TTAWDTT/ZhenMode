"""NumPy-only independent actual nonlinear last-step/Q/impulse/work audit.

Held baroclinic force is audited as recorded work, not full buoyancy PE closure.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EPS = 64. * np.finfo(float).eps


def shifted(field, axis, direction):
    if axis == 0:
        return np.roll(field, -direction, axis=0)
    selection = [slice(None)] * field.ndim
    selection[axis] = slice(0, 1)
    zero = np.zeros_like(field[tuple(selection)])
    selection[axis] = slice(1, None) if direction > 0 else slice(None, -1)
    parts = (field[tuple(selection)], zero) if direction > 0 else (zero, field[tuple(selection)])
    return np.concatenate(parts, axis=axis)


def norm(fields):
    return np.sqrt(sum(np.sum(field ** 2) for field in fields))


def map_mass(field, split):
    return (.5 * (field + np.roll(field, -1, axis=0)),
            np.pad(field * split, ((0, 0), (0, 1), (0, 0))) + np.pad(field * (1. - split), ((0, 0), (1, 0), (0, 0))))


def audit(data, dt=60.):
    if any(not np.all(np.isfinite(field)) for field in data.values()):
        raise ValueError("nonfinite nonlinear snapshot")
    area, base = data["area"], data["thickness"]
    latitude, longitude = np.radians(data["latitude_edges"]), np.radians(data["longitude_edges"])
    middle, width = .5 * (latitude[:-1] + latitude[1:]), np.diff(latitude)
    radius = np.sqrt(area[0, 0] / (np.diff(longitude)[0] * np.diff(np.sin(latitude))[0]))
    if not np.allclose(area, radius ** 2 * np.diff(longitude)[:, None] * np.diff(np.sin(latitude))[None, :], rtol=1e-12, atol=0.):
        raise ValueError("spherical geometry consistency")
    volume0, volume1 = data["previous_volume"], data["final_volume"]
    if np.any(np.where(base > 0., (volume0 <= 0.) | (volume1 <= 0.), (volume0 != 0.) | (volume1 != 0.))):
        raise ValueError("physical positive wet volume")
    split = ((np.sin(middle) - np.sin(latitude[:-1])) / np.diff(np.sin(latitude)))[None, :, None]
    mass0, mass1 = map_mass(volume0, split), map_mass(volume1, split)
    initial = (data["previous_east_velocity"].astype(float), np.pad(data["previous_north_velocity"].astype(float), ((0, 0), (1, 0), (0, 0))))
    final = (data["uncast_east"], data["uncast_north"])
    velocity = (data["transport_east"], data["transport_north"])
    stored = (data["final_east_velocity"].astype(float), np.pad(data["final_north_velocity"].astype(float), ((0, 0), (1, 0), (0, 0))))
    for first, last, old_speed, new_speed, stage in zip(mass0, mass1, initial, final, velocity):
        expected = (np.sqrt(last) * new_speed + np.sqrt(first) * old_speed) / np.where(first + last > 0., np.sqrt(last) + np.sqrt(first), 1.)
        if np.any(abs(stage - expected) > 1e-12 * abs(expected) + EPS * (abs(old_speed) + abs(new_speed))):
            raise ValueError("actual mass-weighted transport velocity")
    middle_volume = .5 * (volume0 + volume1)
    height = middle_volume / area[..., None]
    top = np.broadcast_to(data["interfaces"][:-1], height.shape).copy()
    top[..., 0] = base[..., 0] - height[..., 0]
    bottom = top + height
    contact_areas = []
    for axis in range(2):
        opened = (base > 0.) & (shifted(base, axis, 1) > 0.)
        common = np.where(opened, np.maximum(np.minimum(bottom, shifted(bottom, axis, 1)) - np.maximum(top, shifted(top, axis, 1)), 0.), 0.)
        length = radius * width[None, :] if axis == 0 else radius * np.diff(longitude)[:, None] * np.cos(latitude[1:])[None, :]
        contact_areas.append(common * length[..., None])
    fluxes = (data["flux_east"], data["flux_north"], data["flux_vertical"])
    for actual, expected in zip(fluxes[:2], (contact_areas[0] * velocity[0], contact_areas[1] * velocity[1][:, 1:])):
        if np.any(abs(actual - expected) > 1e-12 * abs(expected) + EPS * (abs(actual) + abs(expected))):
            raise ValueError("actual midpoint geometry and shared Q")
    horizontal_net = fluxes[0] - np.roll(fluxes[0], 1, axis=0) + fluxes[1] - shifted(fluxes[1], 1, -1)
    net = horizontal_net + fluxes[2][..., 1:] - fluxes[2][..., :-1]
    volume_error = volume1 - volume0 + dt * net
    if norm((volume_error,)) > 1e-11 * dt * norm((net,)) + EPS * norm((volume0, volume1)):
        raise ValueError("actual shared Q volume equation")
    column = np.sum(horizontal_net, axis=-1)
    if np.any(abs(net[..., 1:]) > (1e-12 + EPS) * (abs(horizontal_net[..., 1:]) + abs(fluxes[2][..., 1:-1]) + abs(fluxes[2][..., 2:]))):
        raise ValueError("fixed lower volume continuity")
    if np.any(fluxes[2][..., 0] != 0.) or np.any(fluxes[2][..., -1] != 0.):
        raise ValueError("closed material top and bottom")
    eta0, eta1, eta_mean = (data[name] for name in ("initial_eta", "final_eta", "mean_eta"))
    for eta, volume in ((eta0, volume0), (eta1, volume1)):
        host_eta = volume[..., 0] / area - base[..., 0]
        if np.any(abs(eta - host_eta) > EPS * (abs(volume[..., 0] / area) + abs(base[..., 0]))):
            raise ValueError("recorded physical surface representation")
    if np.max(abs(eta1 - (2. * eta_mean - eta0))) > 1e-12 + EPS * np.max(abs(eta1) + abs(eta0) + base[..., 0]):
        raise ValueError("actual surface time centering")
    if norm((area * (eta1 - eta0) + dt * column,)) > 1e-11 * dt * norm((column,)) + EPS * norm((volume0[..., 0], volume1[..., 0])):
        raise ValueError("surface and shared continuity")
    if abs(np.sum(area * (eta_mean - eta0))) > 1e-11 * dt * np.sum(abs(column)) + EPS * (np.sum(volume0[..., 0]) + np.sum(volume1[..., 0])):
        raise ValueError("surface mean nullspace")
    north_full = np.pad(fluxes[1], ((0, 0), (1, 0), (0, 0)))
    east_dual = tuple(.5 * (field + np.roll(field, -1, axis=0)) for field in (fluxes[0], north_full, fluxes[2]))
    north_center = ((1. - split) * north_full[:, :-1] + split * north_full[:, 1:]
                    + (split - .5) * (fluxes[0] - np.roll(fluxes[0], 1, axis=0)))
    north_dual = (map_mass(fluxes[0], .5)[1], np.pad(north_center, ((0, 0), (1, 1), (0, 0))), map_mass(fluxes[2], split)[1])
    cell_east = .5 * (velocity[0] + np.roll(velocity[0], 1, axis=0))
    cell_north = .5 * (velocity[1][:, :-1] + velocity[1][:, 1:])
    frequency = 2. * 7.2921e-5 * np.sin(middle)[None, :, None] + cell_east * np.tan(middle)[None, :, None] / radius
    cell_force_east = middle_volume * frequency * cell_north
    cell_force_north = -middle_volume * frequency * cell_east
    spin = (.5 * (cell_force_east + np.roll(cell_force_east, -1, axis=0)), map_mass(cell_force_north, .5)[1])
    pressure = (contact_areas[0] * (eta_mean - np.roll(eta_mean, -1, axis=0))[..., None],
                np.pad(contact_areas[1] * (eta_mean - shifted(eta_mean, 1, 1))[..., None], ((0, 0), (1, 0), (0, 0))))
    masks = (contact_areas[0] > 0., np.pad(contact_areas[1] > 0., ((0, 0), (1, 0), (0, 0))))
    residuals, scales, stores = [], [], []
    for component, dual in enumerate((east_dual, north_dual)):
        advection = np.zeros_like(velocity[component])
        for axis, flux in enumerate((dual[0], dual[1][:, 1:], dual[2][..., 1:])):
            transport = flux * .5 * (velocity[component] + shifted(velocity[component], axis, 1))
            advection += transport - shifted(transport, axis, -1)
        held = data[("held_east", "held_north")[component]]
        tendency = spin[component] + 9.81 * pressure[component] - advection + held
        impulse = mass1[component] * final[component] - mass0[component] * initial[component]
        row = impulse - dt * tendency
        residuals.append(np.where(masks[component], row, 0.))
        scales.append(np.where(masks[component], abs(impulse) + dt * abs(tendency), 0.))
        stores.append(np.where(masks[component], abs(mass1[component] * final[component]) + abs(mass0[component] * initial[component]), 0.))
        reaction = data[("reaction_east", "reaction_north")[component]]
        expected_reaction = np.where(masks[component], 0., row)
        if norm((reaction - expected_reaction,)) > 1e-11 * norm((expected_reaction,)) + EPS * norm((dt * tendency, impulse)):
            raise ValueError("actual wall reaction impulse")
        if np.any(np.where(masks[component], False, (initial[component] != 0.) | (final[component] != 0.))):
            raise ValueError("closed normal velocity")
        if not np.array_equal(stored[component], final[component].astype(data["final_east_velocity"].dtype).astype(float)):
            raise ValueError("explicit momentum storage cast")
    momentum_error, momentum_scale = norm(residuals), norm(scales)
    if momentum_error > 1e-11 * momentum_scale + EPS * norm(stores):
        raise ValueError("actual nonlinear momentum impulse")
    kinetic0 = .5 * sum(np.sum(mass * speed ** 2) for mass, speed in zip(mass0, initial))
    kinetic1 = .5 * sum(np.sum(mass * speed ** 2) for mass, speed in zip(mass1, final))
    surface0, surface1 = (.5 * 9.81 * np.sum(area * eta ** 2) for eta in (eta0, eta1))
    work = dt * sum(np.sum(speed * data[name]) for speed, name in zip(velocity, ("held_east", "held_north")))
    energy_error = kinetic1 - kinetic0 + surface1 - surface0 - work
    energy_scale = abs(kinetic1 - kinetic0) + abs(surface1 - surface0) + abs(work)
    if abs(energy_error) > 1e-11 * energy_scale + EPS * (kinetic0 + kinetic1 + surface0 + surface1):
        raise ValueError("actual nonlinear kinetic/surface/held work")
    recorded_work = float(data["diagnostic_held_force_work"])
    if abs(recorded_work - work) > (1e-11 + EPS) * abs(work):
        raise ValueError("recorded held work")
    constant = data["final_content"][..., 1] / np.where(volume1 > 0., volume1, 1.)
    if np.max(abs(constant[base > 0.] - 35.)) > 1e-12:
        raise ValueError("actual constant tracer transport")
    for tracer in range(data["final_content"].shape[-1]):
        change = np.sum(data["final_content"][..., tracer] - data["previous_content"][..., tracer])
        scale = np.sum(abs(data["previous_content"][..., tracer])) + np.sum(abs(data["final_content"][..., tracer]))
        if abs(change) > EPS * scale:
            raise ValueError("closed extensive tracer inventory")
    cast = .5 * sum(np.sum(mass * speed ** 2) for mass, speed in zip(mass1, stored)) - kinetic1
    if abs(cast - float(data["diagnostic_cast_work"])) > EPS * (kinetic0 + kinetic1):
        raise ValueError("recorded cast work")
    return {"momentum_relative": float(momentum_error / momentum_scale if momentum_scale else 0.),
            "energy_change_work_relative": float(abs(energy_error) / energy_scale if energy_scale else 0.),
            "momentum_error": float(momentum_error), "energy_error": float(energy_error), "cast_work": float(cast)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--out", required=True)
    parser.add_argument("--negative-controls", action="store_true")
    args = parser.parse_args()
    path = ROOT / args.report
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["status"] != "PASS" or len(report["runs"]) != 8 or not report["source_hashes_unchanged"] or report["provenance"]["git_status"]:
        raise ValueError("real nonlinear reference incomplete or unpinned")
    if report["kinetic_norm"] != "finite_volume_full_half_prism_dual_not_physical_field_L2" or not report["curvature"] or report["upwind"]:
        raise ValueError("actual nonlinear method contract")
    def sha(filename):
        return hashlib.sha256(filename.read_bytes()).hexdigest()

    if any(sha(ROOT / name) != value for name, value in report["provenance"]["source_sha256"].items()):
        raise ValueError("runtime source hashes changed")
    if sha(ROOT / report["provenance"]["input_report"]) != report["provenance"]["input_report_sha256"]:
        raise ValueError("immutable input report hash")
    expected_groups = {(geometry, dtype, disturbed) for geometry in ("prior_smoothed", "unsmoothed")
                       for dtype in ("float64", "float32") for disturbed in (False, True)}
    if {(run["geometry"], run["velocity_dtype"], run["disturbed"]) for run in report["runs"]} != expected_groups:
        raise ValueError("eight actual nonlinear group coverage")
    evidence, rejected = [], []
    for run in report["runs"]:
        snapshot = ROOT / run["snapshot_path"]
        if sha(snapshot) != run["snapshot_sha256"] or sha(ROOT / run["input_snapshot"]) != run["input_snapshot_sha256"]:
            raise ValueError("nonlinear snapshot/input hash")
        if run["status"] != "PASS" or run["completed_steps"] != run["requested_steps"] or len(run["history"]) != run["completed_steps"]:
            raise ValueError("nonlinear trajectory incomplete")
        for index, row in enumerate(run["history"]):
            if row["step"] != index + 1 or not row["accepted"] or any(value is None or not np.isfinite(value) for value in row.values()):
                raise ValueError("recorded nonlinear trajectory failure")
            if (row["momentum_residual"] > row["momentum_tolerance"] or abs(row["energy_residual"]) > row["energy_tolerance"]
                    or row["continuity_residual"] > row["continuity_tolerance"] or row["constant_error"] > 1e-12 or row["bound_excursion"] > 1e-12
                    or row["outer_change"] > 2e-13 or row["solve_relative_residual"] > 1e-12):
                raise ValueError("recorded nonlinear gate violation")
        with np.load(snapshot, allow_pickle=False) as saved:
            data = {name: saved[name] for name in saved.files}
        evidence.append({"geometry": run["geometry"], "dtype": run["velocity_dtype"], "disturbed": run["disturbed"], **audit(data, report["dt_seconds"])})
        if args.negative_controls and run["disturbed"] and run["velocity_dtype"] == "float64" and not rejected:
            for name in ("uncast_east", "transport_east", "flux_vertical", "mean_eta", "initial_eta", "held_east", "reaction_north", "final_volume", "final_content", "diagnostic_held_force_work", "latitude_edges"):
                corrupt = {key: value.copy() for key, value in data.items()}
                corrupt[name] = corrupt[name] + .01 if name in ("mean_eta", "initial_eta") else corrupt[name] * 1.01
                try:
                    audit(corrupt, report["dt_seconds"])
                except ValueError:
                    rejected.append(name)
                else:
                    raise ValueError(f"independent nonlinear control accepted: {name}")
    result = {"status": "PASS", "scope": "independent_last_nonlinear_step_recorded_held_work_not_full_thermodynamics_or_century",
              "report_sha256": sha(path), "verifier_sha256": sha(Path(__file__)), "runs": evidence, "negative_controls_rejected": rejected}
    output = ROOT / args.out
    if output.exists():
        raise FileExistsError("retain independent evidence")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
