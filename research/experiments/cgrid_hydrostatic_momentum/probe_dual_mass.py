"""Falsify ordinary-MAC dual continuity on moving common-wet supports."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from cgrid_momentum import LayerState, linear_momentum_surface_step, momentum_geometry
from finite_volume import ExtensiveState, build_geometry, surface_volume

jax.config.update("jax_enable_x64", True)


def adjacent(field, axis):
    return np.roll(field, -1, axis=0) if axis == 0 else np.concatenate((field[:, 1:], np.zeros_like(field[:, :1])), axis=1)


def physical_halves(geometry, volume, axis):
    area = np.asarray(geometry.area)
    height = volume / area[..., None]
    top = np.broadcast_to(geometry.interfaces[:-1], height.shape).copy()
    top[..., 0] = geometry.thickness[..., 0] - height[..., 0]
    bottom = top + height
    common = np.maximum(np.minimum(bottom, adjacent(bottom, axis)) - np.maximum(top, adjacent(top, axis)), 0.)
    opened = np.asarray((geometry.east_area, geometry.north_area)[axis]) > 0.
    common = np.where(opened, common, 0.)
    if axis == 0:
        half_left, half_right = .5 * area, .5 * adjacent(area, axis)
    else:
        latitude = geometry.latitude_edges
        middle = .5 * (latitude[:-1] + latitude[1:])
        denominator = np.diff(np.sin(latitude))
        half_left = area * ((np.sin(latitude[1:]) - np.sin(middle)) / denominator)[None, :]
        half_right = adjacent(area * ((np.sin(middle) - np.sin(latitude[:-1])) / denominator)[None, :], axis)
    fractions = (half_left / area, half_right / np.where(adjacent(area, axis) > 0., adjacent(area, axis), 1.))
    beta = (fractions[0][..., None] * common / np.where(height > 0., height, 1.),
            fractions[1][..., None] * common / np.where(adjacent(height, axis) > 0., adjacent(height, axis), 1.))
    return (half_left + half_right)[..., None] * common, beta, fractions


def comparison(actual, expected, initial, final):
    scale = np.abs(actual) + np.abs(expected)
    floor = 64. * np.finfo(np.float64).eps * np.maximum(np.abs(initial), np.abs(final))
    residual = actual - expected
    denominator = float(np.sum(scale))
    return {"observed_change_sum_m3": float(np.sum(actual)), "predicted_change_sum_m3": float(np.sum(expected)),
            "residual_l1_m3": float(np.sum(np.abs(residual))), "change_scale_l1_m3": denominator,
            "relative_change_residual": float(np.sum(np.abs(residual))) / denominator if denominator > 0. else 0.,
            "maximum_local_residual_m3": float(np.max(np.abs(residual))), "rounding_floor_l1_m3": float(np.sum(floor)),
            "relative_gate": 1e-12, "pass": bool(np.all(np.abs(residual) <= 1e-12 * scale + floor))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/industrial_alignment/cgrid_moving_dual_mass.json")
    args = parser.parse_args()
    output = ROOT / args.out
    if output.exists():
        raise FileExistsError("retain previous evidence; choose a new --out")
    rows = []
    for label, longitude, latitude in (
            ("regular", np.linspace(0., 360., 13), np.linspace(-60., 60., 7)),
            ("irregular_partial_land", np.array([0., 37., 131., 206., 298., 360.]), np.array([-60., -27., -3., 15., 56.]))):
        depth = np.full((len(longitude) - 1, len(latitude) - 1), 50.)
        if label != "regular":
            depth[1, 1], depth[2, 2], depth[3, 1] = 22., 3., 0.
        geometry = build_geometry(longitude, latitude, [0., 7., 21., 50.], depth)
        volume = surface_volume(geometry, jnp.zeros(depth.shape))
        initial_volume = np.asarray(volume)
        inventory = ExtensiveState(volume, volume[..., None] * jnp.asarray([12., 35.]))
        state = LayerState(inventory, jnp.zeros_like(volume), jnp.zeros_like(volume))
        for forcing in ("uniform_rain_control", "isolated_rain_witness"):
            displacement = np.where(depth > 0., .2, 0.) if forcing == "uniform_rain_control" else np.zeros(depth.shape)
            if forcing != "uniform_rain_control":
                displacement[1, 2] = .2
            source = jnp.zeros_like(volume).at[..., 0].set(jnp.asarray(geometry.area * displacement / 60.))
            result = jax.jit(lambda state: linear_momentum_surface_step(
                geometry, state, jnp.zeros_like(volume), 60., 4, gravity=0., coriolis=0.,
                volume_source=source, content_source=source[..., None] * jnp.asarray([12., 35.])))(state)
            final_volume = np.asarray(result.state.inventory.volume)
            east, north, vertical = map(np.asarray, result.fluxes)
            net = east - np.roll(east, 1, axis=0) + north
            net[:, 1:] -= north[:, :-1]
            net += vertical[..., 1:] - vertical[..., :-1]
            primal_increment = 60. * (np.asarray(source) - net)
            scalar_check = comparison(final_volume - initial_volume, primal_increment, initial_volume, final_volume)
            faces_before, faces_after = momentum_geometry(geometry, volume), momentum_geometry(geometry, result.state.inventory.volume)
            directions = {}
            arrays = {"initial_volume": initial_volume, "final_volume": final_volume, "volume_source": np.asarray(source),
                      "east_flux": east, "north_flux": north, "vertical_flux": vertical, "area": geometry.area,
                      "thickness": geometry.thickness, "interfaces": geometry.interfaces, "latitude_edges": latitude,
                      "longitude_edges": longitude, "depth": depth}
            for axis, component in enumerate(("east", "north")):
                initial_mass, beta_before, fractions = physical_halves(geometry, initial_volume, axis)
                final_mass, beta_after, unused_fractions = physical_halves(geometry, final_volume, axis)
                for observed, expected in ((np.asarray((faces_before.east_volume, faces_before.north_volume)[axis]), initial_mass),
                                           (np.asarray((faces_after.east_volume, faces_after.north_volume)[axis]), final_mass)):
                    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-6)
                opened = initial_mass > 0.
                ordinary = np.where(opened, fractions[0][..., None] * primal_increment
                                    + fractions[1][..., None] * adjacent(primal_increment, axis), 0.)
                contraction = .5 * (beta_before[0] + beta_after[0]) * primal_increment
                contraction += .5 * (beta_before[1] + beta_after[1]) * adjacent(primal_increment, axis)
                exchange = .5 * (initial_volume + final_volume) * (beta_after[0] - beta_before[0])
                exchange += .5 * adjacent(initial_volume + final_volume, axis) * (beta_after[1] - beta_before[1])
                change = final_mass - initial_mass
                directions[component] = {"ordinary_mac": comparison(change, ordinary, initial_mass, final_mass),
                                         "kinematic_product_not_momentum_fix": comparison(change, contraction + exchange, initial_mass, final_mass),
                                         "geometry_exchange_l1_m3": float(np.sum(np.abs(exchange))),
                                         "initial_represented_fraction": float(np.sum(initial_mass) / np.sum(initial_volume)),
                                         "final_represented_fraction": float(np.sum(final_mass) / np.sum(final_volume))}
                arrays.update({component + "_initial_mass": initial_mass, component + "_final_mass": final_mass,
                               component + "_ordinary_change": ordinary, component + "_contracted_change": contraction,
                               component + "_geometry_exchange": exchange})
            snapshot = output.with_name(output.stem + "_" + label + "_" + forcing + ".npz")
            if snapshot.exists():
                raise FileExistsError(snapshot)
            np.savez_compressed(snapshot, **arrays)
            row = {"geometry": label, "forcing": forcing, "step_valid": bool(result.valid), "scalar": scalar_check,
                   "source_adjusted_volume_delta_m3": float(np.sum(final_volume - initial_volume) - 60. * np.sum(source)),
                   "directions": directions, "snapshot_path": snapshot.relative_to(ROOT).as_posix(),
                   "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}
            rows.append(row)
            print(label, forcing, "actual step valid", bool(result.valid),
                  {component: values["ordinary_mac"]["relative_change_residual"] for component, values in directions.items()}, flush=True)
    sources = [ROOT / "src" / name for name in ("finite_volume.py", "bounded_transport.py", "barotropic_transport.py", "cgrid_momentum.py", "config.py")]
    sources += [Path(__file__), Path(__file__).with_name("dual_mass_protocol.md"), ROOT / "tests/test_momentum_shared_flux.py"]
    controls_pass = all(row["step_valid"] and row["scalar"]["pass"]
                        and all(values["kinematic_product_not_momentum_fix"]["pass"] for values in row["directions"].values()) for row in rows)
    report = {"scope": "ordinary_mac_hypothesis_on_actual_moving_wet_supports_not_implemented_nonlinear_momentum",
              "status": "PASS" if controls_pass and all(values["ordinary_mac"]["pass"] for row in rows for values in row["directions"].values()) else "FAIL",
              "controls_pass": controls_pass, "dt_seconds": 60., "barotropic_substeps": 4, "rows": rows,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
              "jax_version": jax.__version__, "backend": jax.default_backend()}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("status", report["status"], "controls_pass", controls_pass)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
