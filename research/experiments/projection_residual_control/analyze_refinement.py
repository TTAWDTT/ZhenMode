"""Frozen predictors: true matrix residual versus native velocity refinement."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from config import DEFAULT_CONFIG, G_EARTH
from jax_solver_global import (
    _column_divergence,
    _gradient_conservative_3d,
    _project_column_divergence,
    _vertical_transport_iface,
    make_solver_global,
    projection_config,
)
from verify_debug_integration import make_smoke_fixture


def pressure_problem(params):
    area = params.dx_2d * params.dy * params.wet_mask

    def operator(potential):
        gradient = _gradient_conservative_3d(potential[..., None], params)
        return -area * _column_divergence(*gradient, params)

    def right_hand_side(velocities):
        return -(_column_divergence(*velocities, params) / (params.dt * G_EARTH)) * area

    def correct(velocities, potential):
        gradient = _gradient_conservative_3d(potential[..., None], params)
        return tuple(velocity - (params.dt * G_EARTH) * derivative
                     for velocity, derivative in zip(velocities, gradient, strict=True))

    def preconditioner(residual):
        return residual * params.projection_inv_diagonal

    return operator, right_hand_side, correct, preconditioner


def recursive_cg(operator, right_hand_side, preconditioner, tolerance, cap):
    def inner(first, second):
        return jnp.vdot(first, second, precision=jax.lax.Precision.HIGHEST)

    threshold = tolerance ** 2 * inner(right_hand_side, right_hand_side)
    residual = right_hand_side - operator(jnp.zeros_like(right_hand_side))
    direction = preconditioner(residual)

    def active(carry):
        return (inner(carry[1], carry[1]) > threshold) & (carry[4] < cap)

    def advance(carry):
        solution, remainder, product, direction, count = carry
        applied = operator(direction)
        alpha = product / inner(direction, applied)
        solution = solution + alpha * direction
        remainder = remainder - alpha * applied
        scaled = preconditioner(remainder)
        next_product = inner(remainder, scaled)
        direction = scaled + (next_product / product) * direction
        return solution, remainder, next_product, direction, count + 1

    return jax.lax.while_loop(active, advance, (
        jnp.zeros_like(right_hand_side), residual, inner(residual, direction), direction, jnp.asarray(0)))


def make_candidate(params, kind, refinements, absolute_floor=True):
    operator, right_hand_side, correct, preconditioner = pressure_problem(params)
    tolerance = projection_config(params)["rtol"]

    @jax.jit
    def project(velocity_x, velocity_y):
        velocities = (velocity_x, velocity_y)
        original_rhs = right_hand_side(velocities)
        floor = tolerance * jnp.linalg.norm(original_rhs) if absolute_floor else 0.
        potential, _ = jax.scipy.sparse.linalg.cg(
            operator, original_rhs, tol=tolerance, maxiter=params.projection_niter, M=preconditioner)
        corrected = correct(velocities, potential)
        for refinement in range(refinements):
            residual = (original_rhs - operator(potential) if kind == "pressure"
                        else right_hand_side(corrected))
            delta, _ = jax.scipy.sparse.linalg.cg(
                operator, residual, tol=tolerance, atol=floor,
                maxiter=params.projection_niter, M=preconditioner)
            if kind == "pressure":
                potential = potential + delta
                corrected = correct(velocities, potential)
            else:
                corrected = correct(corrected, delta)
        return corrected

    return project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/industrial_alignment/projection_refinement.json")
    parser.add_argument("--bathy", default=DEFAULT_CONFIG.bathymetry_file)
    args = parser.parse_args()
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifests = [ROOT / "results/industrial_alignment/projection_convergence.json",
                 ROOT / "results/industrial_alignment/projection_precision.json"]
    convergence, precision = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    if any(manifest["status"] != "complete" for manifest in (convergence, precision)):
        raise ValueError("completed registered input manifests required")
    inputs = [{**item, "kind": "coarse", "fields": ["velocity_x", "velocity_y"]}
              for item in convergence["inputs"]]
    inputs.extend({"dtype": "float32", "case": item["case"], "step": item["worst_step"],
                   "path": item["raw_path"], "sha256": item["raw_sha256"], "kind": "worst",
                   "fields": ["initial_x", "initial_y"]} for item in precision["cases"])
    for item in inputs:
        if hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"input hash mismatch: {item['path']}")
    grid, physics, initial_temperature, initial_salinity, atmosphere, forcing = make_smoke_fixture(2., args.bathy, 2e14)
    settings = dict(forcing=forcing, T_atm=atmosphere, lambda_bulk=80., T_init=initial_temperature,
                    S_init=initial_salinity, mode_split=True, dt_bt=50., nu_nsub="cfl", use_scan=True,
                    conservative_kv=True, localize_conv=True, project_adv_vel=True, fct_adv=True,
                    projection_niter=600, projection_preconditioner="jacobi", return_params=True)
    sources = sorted((ROOT / "src").glob("*.py")) + [ROOT / "scripts/verify_debug_integration.py",
              Path(__file__).resolve(), Path(__file__).with_name("protocol.md")]
    report = {"scope": "fixed_rhs_not_bitwise_full_step_or_physical_budget_qualification", "status": "running",
              "inputs": inputs, "diagnoses": [], "results": [], "provenance": {
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
                  "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                  "manifest_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in manifests},
                  "bathymetry_sha256": hashlib.sha256(Path(args.bathy).read_bytes()).hexdigest(),
                  "jax_version": jax.__version__, "backend": jax.default_backend()}}

    def save():
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    for dtype in ("float64", "float32"):
        params = make_solver_global(grid, physics, 600., dtype=dtype, **settings)[3]
        params64 = make_solver_global(grid, physics, 600., dtype="float64", **settings)[3]
        candidates = {f"{kind}_{count}": make_candidate(params, kind, count)
                      for kind, count in (("pressure", 0), ("pressure", 1), ("pressure", 2), ("velocity", 1), ("velocity", 2))}
        candidates["float64_quantized"] = jax.jit(lambda velocity_x, velocity_y: tuple(
            field.astype(velocity_x.dtype) for field in _project_column_divergence(
                velocity_x.astype(jnp.float64), velocity_y.astype(jnp.float64), params64, params64.dt)))
        if dtype == "float32":
            candidates["velocity_1_no_absolute_floor"] = make_candidate(params, "velocity", 1, False)
        area = np.asarray(params.dx_2d, dtype=np.float64) * params.dy * np.asarray(params.wet_mask)
        volume = area[..., None] * np.asarray(params.dz_node, dtype=np.float64) * np.asarray(params.wet_mask_z)
        operator, right_hand_side, correct, preconditioner = pressure_problem(params)

        @jax.jit
        def diagnose(velocity_x, velocity_y):
            rhs = right_hand_side((velocity_x, velocity_y))
            tolerance = projection_config(params)["rtol"]
            potential, _ = jax.scipy.sparse.linalg.cg(operator, rhs, tol=tolerance,
                maxiter=params.projection_niter, M=preconditioner)
            mirrored = recursive_cg(operator, rhs, preconditioner, tolerance, params.projection_niter)
            return (rhs, rhs - operator(potential), mirrored[1], rhs - operator(mirrored[0]), mirrored[4],
                    jnp.max(jnp.abs(mirrored[0] - potential)), correct((velocity_x, velocity_y), potential),
                    _project_column_divergence(velocity_x, velocity_y, params, params.dt))

        for item in (entry for entry in inputs if entry["dtype"] == dtype):
            with np.load(ROOT / item["path"]) as raw:
                velocities = tuple(jnp.asarray(raw[name]) for name in item["fields"])
            before = np.asarray(_vertical_transport_iface(*velocities, params)[..., 0], dtype=np.float64)
            norm_before = np.sum(before ** 2 * area)
            if norm_before <= 0.:
                raise ValueError("nonzero right hand side required")
            energy_before = sum(np.sum(np.asarray(field, dtype=np.float64) ** 2 * volume) for field in velocities)
            rhs, true_residual, recursive_residual, mirror_true, iterations, mirror_difference, reconstructed, original = diagnose(*velocities)
            norm_rhs = np.linalg.norm(np.asarray(rhs, dtype=np.float64))
            diagnosis = {**{name: item[name] for name in ("dtype", "case", "kind", "step")},
                         "rtol": projection_config(params)["rtol"], "iterations": int(iterations),
                         "public_true_matrix_relative_residual": float(np.linalg.norm(np.asarray(true_residual, dtype=np.float64)) / norm_rhs),
                         "mirror_recursive_relative_residual": float(np.linalg.norm(np.asarray(recursive_residual, dtype=np.float64)) / norm_rhs),
                         "mirror_true_matrix_relative_residual": float(np.linalg.norm(np.asarray(mirror_true, dtype=np.float64)) / norm_rhs),
                         "mirror_pressure_max_difference": float(mirror_difference),
                         "reconstruction_velocity_max_differences": [float(jnp.max(jnp.abs(first - second)))
                              for first, second in zip(reconstructed, original, strict=True)]}
            report["diagnoses"].append(diagnosis)
            print(json.dumps({"diagnosis": diagnosis}), flush=True)
            for name, project in candidates.items():
                if name.endswith("no_absolute_floor") and item["kind"] != "worst":
                    continue
                corrected = project(*velocities)
                corrected[0].block_until_ready()
                timings = []
                for repeat in range(3):
                    started = time.perf_counter()
                    corrected = project(*velocities)
                    corrected[0].block_until_ready()
                    timings.append(time.perf_counter() - started)
                after = np.asarray(_vertical_transport_iface(*corrected, params)[..., 0], dtype=np.float64)
                ratio = float(np.sqrt(np.sum(after ** 2 * area) / norm_before))
                energy_ratio = float(sum(np.sum(np.asarray(field, dtype=np.float64) ** 2 * volume) for field in corrected) / energy_before)
                finite = bool(np.isfinite(ratio) and np.isfinite(energy_ratio) and all(np.all(np.isfinite(field)) for field in corrected))
                threshold = 1e-9 if dtype == "float64" else 5e-5
                margin = 1e-12 if dtype == "float64" else 5e-6
                result = {**{key: item[key] for key in ("dtype", "case", "kind", "step")}, "candidate": name,
                          "native_area_l2_relative_residual": ratio if finite else None,
                          "wet_energy_ratio": energy_ratio if finite else None, "finite": finite,
                          "output_dtype": str(corrected[0].dtype), "compiled_cpu_median_seconds": float(np.median(timings)),
                          "gate_pass": bool(finite and ratio <= threshold and energy_ratio <= 1. + margin)}
                report["results"].append(result)
                print(json.dumps(result, allow_nan=False), flush=True)
            save()
        jax.clear_caches()
    report["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
