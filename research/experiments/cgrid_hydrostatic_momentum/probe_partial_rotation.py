"""Uniform velocity on common wet depths distinguishes physical from skew rotation."""
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

from cgrid_momentum import coriolis_tendency, rotate_coriolis
from finite_volume import build_geometry

jax.config.update("jax_enable_x64", True)


def main():
    depth = np.full((8, 8), 50.)
    depth[3, :] = 21.
    geometry = build_geometry(np.linspace(0., 360., 9), np.linspace(-.001, .001, 9),
                              [0., 5., 20., 50.], depth)
    east = jnp.zeros(geometry.thickness.shape)
    north = jnp.where(geometry.north_area > 0., 2., 0.)
    acceleration, unused_north = coriolis_tendency(geometry, east, north, coriolis=.001)
    selected = np.asarray(acceleration)[2:4, 2:6, 2]
    expected = .002
    error = float(np.max(np.abs(selected / expected - 1.)))
    rotated = rotate_coriolis(geometry, east, north, 60., coriolis=.001)
    passed = error <= 1e-8 and bool(rotated.valid)
    sources = [ROOT / "src" / name for name in ("cgrid_momentum.py", "finite_volume.py")]
    sources += [Path(__file__), Path(__file__).with_name("partial_rotation_protocol.md")]
    report = {"scope": "local_common_wet_depth_coriolis_consistency_not_whole_model",
              "status": "PASS" if passed else "FAIL", "expected_acceleration_m_s2": expected,
              "measured_acceleration_m_s2": selected.tolist(), "maximum_relative_error": error,
              "relative_gate": 1e-8, "rotation_valid": bool(rotated.valid),
              "rotation_energy_relative": float(rotated.energy_relative_change),
              "rotation_solve_relative": float(rotated.solve_relative_residual),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
              "jax_version": jax.__version__, "backend": jax.default_backend()}
    output = ROOT / "results/industrial_alignment/cgrid_partial_rotation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
