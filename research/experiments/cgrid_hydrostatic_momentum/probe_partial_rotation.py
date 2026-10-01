"""Uniform velocity on common wet depths distinguishes physical from skew rotation."""
import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

import cgrid_momentum
from finite_volume import build_geometry

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision")
    parser.add_argument("--out", default="results/industrial_alignment/cgrid_partial_rotation_current.json")
    args = parser.parse_args()
    output = ROOT / args.out
    if output.exists():
        raise FileExistsError("retain previous evidence; choose a new --out")
    operator = cgrid_momentum
    operator_source = (ROOT / "src/cgrid_momentum.py").read_bytes()
    frozen_revision = None
    if args.revision:
        frozen_revision = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options", f"{args.revision}^{{commit}}"], cwd=ROOT, text=True).strip()
        for dependency in ("config.py", "finite_volume.py", "barotropic_transport.py", "bounded_transport.py"):
            frozen = subprocess.check_output(["git", "show", f"{frozen_revision}:src/{dependency}"], cwd=ROOT)
            if frozen != (ROOT / "src" / dependency).read_bytes():
                raise ValueError(f"frozen replay requires isolated dependency: {dependency}")
        operator_source = subprocess.check_output(["git", "show", f"{frozen_revision}:src/cgrid_momentum.py"], cwd=ROOT)
        operator = types.ModuleType("frozen_cgrid_momentum")
        exec(compile(operator_source, "frozen_cgrid_momentum", "exec"), operator.__dict__)
    depth = np.full((8, 8), 50.)
    depth[3, :] = 21.
    geometry = build_geometry(np.linspace(0., 360., 9), np.linspace(-.001, .001, 9),
                              [0., 5., 20., 50.], depth)
    east = jnp.zeros(geometry.thickness.shape)
    north = jnp.where(geometry.north_area > 0., 2., 0.)
    acceleration, unused_north = operator.coriolis_tendency(geometry, east, north, coriolis=.001)
    selected = np.asarray(acceleration)[2:4, 2:6, 2]
    expected = .002
    error = float(np.max(np.abs(selected / expected - 1.)))
    rotated = operator.rotate_coriolis(geometry, east, north, 60., coriolis=.001)
    passed = error <= 1e-8 and bool(rotated.valid)
    sources = [ROOT / "src" / name for name in ("cgrid_momentum.py", "finite_volume.py")]
    sources += [Path(__file__), Path(__file__).with_name("partial_rotation_protocol.md")]
    source_hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    source_hashes["src/cgrid_momentum.py"] = hashlib.sha256(operator_source).hexdigest()
    report = {"scope": "local_common_wet_depth_coriolis_consistency_not_whole_model",
              "status": "PASS" if passed else "FAIL", "expected_acceleration_m_s2": expected,
              "measured_acceleration_m_s2": selected.tolist(), "maximum_relative_error": error,
              "relative_gate": 1e-8, "rotation_valid": bool(rotated.valid),
              "rotation_energy_relative": float(rotated.energy_relative_change),
              "rotation_solve_relative": float(rotated.solve_relative_residual),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": source_hashes, "frozen_core_revision": frozen_revision,
              "jax_version": jax.__version__, "backend": jax.default_backend()}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
