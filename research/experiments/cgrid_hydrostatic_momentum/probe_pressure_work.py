"""Actual fast pressure work versus physical surface-potential conversion."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from ocean_solver.provenance.archives import current_source_files

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from zhenmode_research.candidates.fv.barotropic import subcycle_barotropic
from zhenmode_research.candidates.fv.momentum import hydrostatic_pressure_force, momentum_geometry
from ocean_solver.config.definitions import R_EARTH
from zhenmode_research.candidates.fv.geometry import (
    _physical_surface_height,
    build_geometry,
    horizontal_divergence,
    surface_volume,
)

jax.config.update("jax_enable_x64", True)


def spatial_mms():
    errors = []
    for nx, ny in ((24, 12), (48, 24), (96, 48)):
        longitude = np.linspace(0., 2. * np.pi, nx + 1)
        latitude = np.linspace(-np.pi / 3., np.pi / 3., ny + 1)
        geometry = build_geometry(np.degrees(longitude), np.degrees(latitude), [0., 7., 21., 50.], np.full((nx, ny), 50.))
        middle = .5 * (latitude[:-1] + latitude[1:])
        phase = .5 * (longitude[:-1] + longitude[1:])
        lon_average = np.diff(np.sin(longitude)) / np.diff(longitude)
        lat_average = np.diff(.5 * latitude + .25 * np.sin(2. * latitude)) / np.diff(np.sin(latitude))
        scalar = lon_average[:, None] * lat_average[None, :]
        analytic = (9.81 / R_EARTH * np.broadcast_to(np.sin(longitude[1:])[:, None], (nx, ny)),
                    9.81 / R_EARTH * np.cos(phase)[:, None] * np.sin(latitude[1:])[None, :])
        east_dual = geometry.area
        north_dual = np.broadcast_to(R_EARTH ** 2 * (2. * np.pi / nx) * np.diff(np.sin(middle), append=np.sin(middle[-1]))[None, :], (nx, ny))
        zero = jnp.zeros((nx, ny))
        volume = surface_volume(geometry, zero)
        pressure = hydrostatic_pressure_force(geometry, volume, jnp.asarray(np.broadcast_to(scalar[..., None], volume.shape)))
        fast = subcycle_barotropic(geometry, jnp.asarray(scalar), zero, zero, 1., 1)
        row = {"nx": nx, "ny": ny, "valid": bool(pressure.valid) and bool(fast.valid)}
        for label, observed, expected, weights in (
                ("fast", (fast.east_velocity, fast.north_velocity), analytic, (east_dual, north_dual)),
                ("hydrostatic", (pressure.east, pressure.north),
                 tuple(value[..., None] * np.array([3.5, 14., 35.5]) / 1025. for value in analytic),
                 tuple(dual[..., None] * np.array([7., 14., 29.]) for dual in (east_dual, north_dual)))):
            row[label + "_relative_rms"] = [float(np.sqrt(np.sum(weight * (np.asarray(actual) - exact) ** 2) / np.sum(weight * exact ** 2)))
                                             for weight, actual, exact in zip(weights, observed, expected)]
        errors.append(row)
    ratios = {label: (np.array([row[label + "_relative_rms"] for row in errors[:-1]])
                      / np.array([row[label + "_relative_rms"] for row in errors[1:]])).tolist()
              for label in ("fast", "hydrostatic")}
    return {"scope": "regular_smooth_pressure_not_irregular_or_full_dynamics_order", "errors": errors,
            "ratios": ratios, "ratio_gate": 3.5,
            "pass": all(row["valid"] for row in errors) and all(np.all(np.asarray(value) >= 3.5) for value in ratios.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/industrial_alignment/cgrid_pressure_work_current.json")
    args = parser.parse_args()
    output = ROOT / args.out
    if output.exists():
        raise FileExistsError("retain previous evidence; choose a new --out")
    rows = []
    for label, longitude, latitude in (
            ("regular", np.linspace(0., 360., 13), np.linspace(-60., 60., 7)),
            ("irregular", np.array([0., 37., 131., 206., 298., 360.]), np.array([-60., -27., -3., 15., 56.]))):
        depth = np.full((len(longitude) - 1, len(latitude) - 1), 50.)
        geometry = build_geometry(longitude, latitude, [0., 7., 21., 50.], depth)
        phase = 2. * np.pi * np.arange(depth.shape[0])[:, None] / depth.shape[0]
        middle = np.radians(.5 * (latitude[:-1] + latitude[1:]))
        requested_eta = .2 * np.cos(phase) * np.cos(middle)[None, :]
        volume = surface_volume(geometry, jnp.asarray(requested_eta))
        eta = np.asarray(_physical_surface_height(geometry, volume))
        faces = momentum_geometry(geometry, volume)
        active = geometry._replace(east_area=faces.east_area, north_area=faces.north_area)
        east_area = np.asarray(jnp.sum(faces.east_area, axis=-1))
        north_area = np.asarray(jnp.sum(faces.north_area, axis=-1))
        east_mass = np.asarray(jnp.sum(faces.east_volume, axis=-1))
        north_mass = np.asarray(jnp.sum(faces.north_volume, axis=-1))
        east = np.broadcast_to(.01 * np.sin(phase), depth.shape)
        north = np.broadcast_to(.02 * np.sin(np.radians(latitude[1:]))[None, :], depth.shape)
        east = np.where(east_area > 0., east, 0.)
        north = np.where(north_area > 0., north, 0.)
        volume_rate = np.asarray(-horizontal_divergence(jnp.asarray(east_area * east), jnp.asarray(north_area * north)))
        potential_power = 1025. * 9.81 * eta * volume_rate
        zero = jnp.zeros_like(jnp.asarray(eta))
        gravity_response = subcycle_barotropic(active, jnp.asarray(eta), zero, zero, 1., 1)
        east_difference = np.roll(eta, -1, axis=0) - eta
        north_difference = np.concatenate((eta[:, 1:], eta[:, -1:]), axis=1) - eta
        compatible_east = -9.81 * east_area * east_difference / np.where(east_mass > 0., east_mass, 1.)
        compatible_north = -9.81 * north_area * north_difference / np.where(north_mass > 0., north_mass, 1.)
        for method, acceleration_east, acceleration_north in (
                ("actual_current_fast_gradient", np.asarray(gravity_response.east_velocity), np.asarray(gravity_response.north_velocity)),
                ("algebraic_compatible_control_not_production_fix", compatible_east, compatible_north)):
            kinetic_east = 1025. * east_mass * east * acceleration_east
            kinetic_north = 1025. * north_mass * north * acceleration_north
            residual = float(np.sum(kinetic_east) + np.sum(kinetic_north) + np.sum(potential_power))
            scale = float(np.sum(np.abs(kinetic_east)) + np.sum(np.abs(kinetic_north)) + np.sum(np.abs(potential_power)))
            relative = abs(residual) / scale
            rows.append({"geometry": label, "method": method, "residual_w": residual,
                         "absolute_power_scale_w": scale, "relative_work_error": relative,
                         "relative_gate": 1e-12, "pass": relative <= 1e-12,
                         "gravity_response_valid": bool(gravity_response.valid),
                         "gravity_cfl_bound": float(gravity_response.gravity_cfl_bound)})
    sources = [ROOT / "src" / name for name in ("finite_volume.py", "cgrid_momentum.py", "barotropic_transport.py", "config.py")]
    sources += [Path(__file__), Path(__file__).with_name("pressure_work_protocol.md"), Path(__file__).with_name("overlap_protocol.md")]
    actual = [row for row in rows if row["method"] == "actual_current_fast_gradient"]
    mms = spatial_mms()
    report = {"scope": "frozen_geometry_gravity_continuity_work_not_full_buoyancy_or_temporal_energy",
              "status": "PASS" if all(row["pass"] and row["gravity_response_valid"] for row in actual) and mms["pass"] else "FAIL",
              "spatial_mms": mms,
              "rows": rows, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in current_source_files(ROOT, sources).items()},
              "jax_version": jax.__version__, "backend": jax.default_backend()}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
