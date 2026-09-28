"""Frozen/current native projection comparison; not a moving-volume budget test."""
import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from test_horizontal_tracer_diffusion import _parameters

import jax_solver_global as current


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/industrial_alignment/column_projection_comparison.json")
    args = parser.parse_args()
    frozen_source = subprocess.check_output(["git", "show", "ab56075:src/jax_solver_global.py"], cwd=ROOT)
    frozen = types.ModuleType("frozen_column_projection")
    frozen.__file__ = str(ROOT / "src/jax_solver_global.py")
    sys.modules[frozen.__name__] = frozen
    exec(compile(frozen_source, frozen.__file__, "exec"), frozen.__dict__)
    results = []
    for land in (False, True):
        for seed in (2718, 3141, 1618):
            _, params, volume = _parameters(65., land=land)
            random = np.random.default_rng(seed)
            velocities = tuple(jnp.asarray(random.normal(size=volume.shape)) * params.wet_mask_z for _ in range(2))
            area = np.asarray(params.dx_2d) * params.dy * np.asarray(params.wet_mask)
            top_before = np.asarray(current._vertical_transport_iface(*velocities, params)[..., 0])
            energy_before = np.sum(sum(np.asarray(velocity) ** 2 for velocity in velocities) * volume)
            for label, solver in (("frozen_ab56075", frozen), ("corrected", current)):
                constraint = np.asarray(solver._column_divergence(*velocities, params))
                for iterations in (150, 1000):
                    corrected = jax.jit(lambda velocity_x, velocity_y: solver._project_column_divergence(
                        velocity_x, velocity_y, params, params.dt, n_iter=iterations))(*velocities)
                    top_after = np.asarray(current._vertical_transport_iface(*corrected, params)[..., 0])
                    energy_after = np.sum(sum(np.asarray(velocity) ** 2 for velocity in corrected) * volume)
                    ratio = np.sqrt(np.sum(top_after ** 2 * area) / np.sum(top_before ** 2 * area))
                    results.append({"land_and_dry_bottom": land, "seed": seed, "kernel": label,
                                    "iteration_cap": iterations,
                                    "constraint_relative_linf_error": finite_number(np.max(np.abs(constraint - top_before)) / np.max(np.abs(top_before))),
                                    "native_transport_relative_area_l2_residual": finite_number(ratio),
                                    "wet_energy_ratio": finite_number(energy_after / energy_before),
                                    "finite": bool(np.all(np.isfinite(top_after)) and np.isfinite(energy_after))})
    report = {"scope": "native_projection_not_complete_free_surface_tracer_coupling",
              "frozen_kernel": "ab56075", "frozen_git_blob_sha256": hashlib.sha256(frozen_source).hexdigest(),
              "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [ROOT / "src/jax_solver_global.py", ROOT / "src/stage_budgets.py",
                                             ROOT / "tests/test_horizontal_tracer_diffusion.py", Path(__file__).resolve()]},
              "results": results}
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
