"""Actual rain/source-momentum counterexample, not a frozen-equation bug claim."""
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

from cgrid_momentum import LayerState
from finite_volume import ExtensiveState, build_geometry, surface_volume
from paired_dynamics import paired_momentum_surface_step

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = ROOT / args.out
    if output.exists():
        raise FileExistsError("retain previous witness; choose new output")
    sources = [Path(__file__), Path(__file__).with_name("nonlinear_source_witness.md")]
    sources.extend(ROOT / "src" / name for name in ("paired_dynamics.py", "cgrid_momentum.py", "finite_volume.py", "bounded_transport.py"))
    files = current_source_files(ROOT, sources)
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    longitude = np.array([0., 37., 131., 206., 298., 360.])
    latitude = np.array([-60., -27., -3., 15., 56.])
    radius, initial_speed, rain, dt = 2.1e6, .1, 1e-6, 600.
    geometry = build_geometry(longitude, latitude, [0., 7., 21., 50.], np.full((5, 4), 50.), radius=radius)
    volume = surface_volume(geometry, jnp.zeros((5, 4)))
    concentration = jnp.broadcast_to(jnp.array([15., 35.]), volume.shape + (2,))
    initial = LayerState(ExtensiveState(volume, volume[..., None] * concentration),
                         jnp.full(volume.shape, initial_speed), jnp.zeros_like(volume))
    source = jnp.zeros_like(volume).at[..., 0].set(geometry.area * rain)
    result = jax.jit(lambda state: paired_momentum_surface_step(
        geometry, state, jnp.zeros_like(volume), dt, 4, gravity=0., coriolis=0.,
        volume_source=source, content_source=source[..., None] * concentration))(initial)
    result.valid.block_until_ready()
    np.testing.assert_allclose(result.state.east_velocity, initial_speed, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(result.state.north_velocity, 0., rtol=0., atol=1e-15)
    lat = np.radians(latitude)
    arc_weight = .5 * np.diff(lat) + .25 * np.diff(np.sin(2. * lat))
    longitude_width = np.diff(np.radians(longitude))
    weight = radius ** 3 * longitude_width[:, None] * arc_weight[None, :]
    initial_height = np.asarray(volume) / geometry.area[..., None]
    final_height = np.asarray(result.state.inventory.volume) / geometry.area[..., None]
    angular0 = initial_speed * np.sum(weight[..., None] * initial_height)
    angular1 = initial_speed * np.sum(weight[..., None] * final_height)
    relative = (angular1 - angular0) / angular0
    kinetic0 = .5 * initial_speed ** 2 * np.sum(np.asarray(volume))
    kinetic1 = .5 * initial_speed ** 2 * np.sum(np.asarray(result.state.inventory.volume))
    recorded = float(result.moving_mass_energy_change)
    floor = 64. * np.finfo(float).eps * (abs(kinetic0) + abs(kinetic1))
    if abs(kinetic1 - kinetic0 - recorded) > 1e-11 * abs(recorded) + floor:
        raise ValueError("independent moving energy witness disagrees")
    unchanged = all(hashlib.sha256(files[name].read_bytes()).hexdigest() == value for name, value in hashes.items())
    record = {"scope": "new_zero_incoming_momentum_source_contract_not_frozen_reference_qualification",
              "provenance": {"git_head_at_launch": head, "source_sha256_at_launch": hashes,
                             "source_hashes_unchanged_at_end": unchanged},
              "inputs": {"longitude_edges": longitude.tolist(), "latitude_edges": latitude.tolist(), "radius_m": radius,
                         "uniform_depth_m": 50., "interfaces_m": [0., 7., 21., 50.], "rain_m_s": rain, "dt_s": dt,
                         "initial_east_speed_m_s": initial_speed, "incoming_east_speed_m_s": 0.},
              "uniform_speed_verified": True, "runtime_frozen_valid": bool(result.valid),
              "frozen_energy_work_relative": float(result.surface.energy_work_relative),
              "actual_top_speed_m_s": float(result.state.east_velocity[0, 0, 0]),
              "isolated_source_conserving_top_speed_m_s": initial_speed * 7. / (7. + rain * dt),
              "axial_per_rho0_initial": float(angular0), "axial_per_rho0_final": float(angular1),
              "axial_relative_change": float(relative), "assumed_incoming_axial_momentum": 0.,
              "source_momentum_qualification": "FAIL" if abs(relative) > 1e-12 + 64. * np.finfo(float).eps else "PASS",
              "independent_kinetic_change_per_rho0": float(kinetic1 - kinetic0), "recorded_moving_mass_energy_per_rho0": recorded}
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    if not unchanged or record["source_momentum_qualification"] != "FAIL" or not record["runtime_frozen_valid"]:
        raise SystemExit("registered counterexample not reproduced")


if __name__ == "__main__":
    main()
