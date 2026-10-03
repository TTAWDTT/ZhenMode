"""Separate CG primal-norm absolute tolerance from initialization in the VJP."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

import zhenmode_research.candidates.fv.momentum as cgrid_momentum
from zhenmode_research.candidates.fv.momentum import _rotation_system
from zhenmode_research.candidates.fv.geometry import build_geometry

jax.config.update("jax_enable_x64", True)


def main():
    geometry = build_geometry(np.linspace(0., 360., 13), np.linspace(-30., 30., 7),
                              [0., 5., 20., 50.], np.full((12, 6), 50.))
    random = np.random.default_rng(945)
    east = jnp.asarray(random.normal(size=geometry.thickness.shape))
    north = jnp.asarray(random.normal(size=geometry.thickness.shape))
    cotangent = jnp.asarray(random.normal(size=geometry.thickness.shape))
    rows = []
    for absolute in ("primal_norm", "zero"):
        for initial_guess in ("primal", "zero"):
            def update(east_field):
                root_east, root_north, clean_east, clean_north, cross, transpose, unused_valid = _rotation_system(geometry, east_field, north, .001)
                initial_east = root_east * clean_east
                initial_north = root_north * clean_north
                half = 30.
                rhs = initial_east - half ** 2 * cross(transpose(initial_east)) + 2. * half * cross(initial_north)

                def matrix(field):
                    return field + half ** 2 * cross(transpose(field))

                atol = 1e-13 * jnp.linalg.norm(rhs) if absolute == "primal_norm" else 0.
                start = initial_east if initial_guess == "primal" else None
                solution, unused_info = cg(matrix, rhs, x0=start, tol=1e-13, atol=atol, maxiter=100)
                return solution / jnp.where(root_east > 0., root_east, 1.)

            _, derivative = jax.jvp(update, (east,), (north,))
            _, pullback = jax.vjp(update, east)
            forward = float(jnp.vdot(cotangent, derivative))
            backward = float(jnp.vdot(pullback(cotangent)[0], north))
            relative = abs(forward - backward) / max(abs(forward), abs(backward), 1e-300)
            rows.append({"atol": absolute, "initial_guess": initial_guess,
                         "jvp_dot": forward, "vjp_dot": backward,
                         "relative_dot_error": relative, "pass": relative <= 1e-12})
            print(rows[-1], flush=True)
    report = {"scope": "same_physical_rotation_matrix_solver_parameter_controls_not_full_adjoint",
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "operator_source_sha256": hashlib.sha256(Path(cgrid_momentum.__file__).read_bytes()).hexdigest(),
              "jax_version": jax.__version__, "backend": jax.default_backend(), "rows": rows}
    output = ROOT / "results/industrial_alignment/cgrid_rotation_adjoint_controls_rechecked.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not any(row["atol"] == "primal_norm" and not row["pass"] for row in rows):
        raise ValueError("old dot-product failure not reproduced")
    selected = [row for row in rows if row["atol"] == "zero" and row["initial_guess"] == "zero"]
    rejected = [row for row in rows if row not in selected]
    if len(selected) != 1 or not selected[0]["pass"] or any(row["pass"] for row in rejected):
        raise ValueError("both zero tolerance and zero initial guess selection not reproduced")


if __name__ == "__main__":
    main()
