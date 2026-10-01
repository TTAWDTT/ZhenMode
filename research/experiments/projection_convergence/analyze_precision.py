"""Capture failing actual predictors, distinguish solve and transport arithmetic."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from config import DEFAULT_CONFIG
from jax_solver_global import _project_column_divergence, _step_impl, _vertical_transport_iface, make_solver_global
from stage_budgets import _StageRecorder, make_budget_step
from verify_debug_integration import make_smoke_fixture


class PrecisionRecorder(_StageRecorder):
    def column_projection(self, velocity_x, velocity_y, corrected_x, corrected_y):
        super().column_projection(velocity_x, velocity_y, corrected_x, corrected_y)
        self.captured = (velocity_x, velocity_y, corrected_x, corrected_y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/industrial_alignment/projection_precision.json")
    parser.add_argument("--bathy", default=DEFAULT_CONFIG.bathymetry_file)
    args = parser.parse_args()
    grid, physics, initial_temperature, initial_salinity, atmosphere, forcing = make_smoke_fixture(2., args.bathy, 2e14)
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "src").glob("*.py")) + [ROOT / "scripts/verify_debug_integration.py", Path(__file__).resolve()]
    report = {"scope": "precision_attribution_not_physical_budget_or_climate_qualification", "status": "running",
              "threshold": 5e-5, "cases": [], "provenance": {
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
                  "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                  "bathymetry_sha256": hashlib.sha256(Path(args.bathy).read_bytes()).hexdigest(),
                  "jax_version": jax.__version__, "backend": jax.default_backend(),
              }}

    def save():
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    for case in ("baseline", "ice"):
        settings = dict(forcing=forcing, T_atm=atmosphere, lambda_bulk=80., T_init=initial_temperature, S_init=initial_salinity,
                        mode_split=True, dt_bt=50., nu_nsub="cfl", use_scan=True, dtype="float32", conservative_kv=True,
                        localize_conv=True, project_adv_vel=True, fct_adv=True, projection_niter=600,
                        projection_preconditioner="jacobi", mixed_layer_depth_m=20. if case == "ice" else None,
                        dynamic_ice=case == "ice", return_params=True)
        _, initialize, _, params, _ = make_solver_global(grid, physics, 600., **settings)
        reference_step = make_budget_step(params)
        state = initialize(T_init=initial_temperature, S_init=initial_salinity)

        @jax.jit
        def capture(previous):
            recorder = PrecisionRecorder(params)
            updated = _step_impl(previous, params, budget=recorder)
            return updated, recorder.captured, recorder.projection_relative_residual_max

        first, _, _ = capture(state)
        expected, _ = reference_step(state)
        differences = [float(jnp.max(jnp.abs(actual - reference))) for actual, reference in zip(first, expected, strict=True)]
        maximum_differences = np.asarray(differences)
        worst = -1.
        failures = []
        captured_worst = None
        worst_step = None
        for step in range(1, 145):
            captured_state, captured, capture_ratio = capture(state)
            state, ledger = reference_step(state)
            maximum_differences = np.maximum(maximum_differences, [float(jnp.max(jnp.abs(actual - reference)))
                                                for actual, reference in zip(captured_state, state, strict=True)])
            ratio = float(ledger["projection_relative_residual_max"])
            if not np.isfinite(ratio):
                raise ValueError("nonfinite native residual")
            if ratio > report["threshold"]:
                failures.append({"step": step, "relative_residual": ratio})
            if ratio > worst:
                worst = ratio
                worst_step = step
                captured_worst = tuple(np.asarray(field) for field in captured)
                worst_capture_ratio = float(capture_ratio)
        quantized_metric64 = params._replace(**{name: value.astype(jnp.float64)
                                     for name, value in params._asdict().items()
                                     if isinstance(value, jnp.ndarray) and jnp.issubdtype(value.dtype, jnp.floating)})
        settings64 = {**settings, "dtype": "float64", "projection_niter": 1200}
        params64 = make_solver_global(grid, physics, 600., **settings64)[3]
        initial = tuple(jnp.asarray(field) for field in captured_worst[:2])
        original = tuple(jnp.asarray(field) for field in captured_worst[2:])
        precise = jax.jit(lambda velocity_x, velocity_y: _project_column_divergence(
            velocity_x, velocity_y, params64, params64.dt))(*(field.astype(jnp.float64) for field in initial))
        quantized = tuple(field.astype(jnp.float32) for field in precise)
        area = np.asarray(params.dx_2d, dtype=np.float64) * params.dy * np.asarray(params.wet_mask)
        initial_top = np.asarray(_vertical_transport_iface(*initial, params)[..., 0], dtype=np.float64)
        denominator = np.sum(initial_top ** 2 * area)
        if denominator <= 0.:
            raise ValueError("nonzero failing predictor required")
        results = []
        for name, velocities in (("captured_float32", original), ("float64_solve", precise), ("float64_solve_quantized_float32", quantized)):
            native = np.asarray(_vertical_transport_iface(*(field.astype(jnp.float32) for field in velocities), params)[..., 0], dtype=np.float64)
            reference = np.asarray(_vertical_transport_iface(*(field.astype(jnp.float64) for field in velocities), quantized_metric64)[..., 0])
            consistent = np.asarray(_vertical_transport_iface(*(field.astype(jnp.float64) for field in velocities), params64)[..., 0])
            results.append({"output": name, "stored_dtype": str(velocities[0].dtype),
                            "float32_transport_relative_residual": float(np.sqrt(np.sum(native ** 2 * area) / denominator)),
                            "float64_transport_relative_residual": float(np.sqrt(np.sum(reference ** 2 * area) / denominator)),
                            "consistent_float64_metric_transport_relative_residual": float(np.sqrt(np.sum(consistent ** 2 * area) / denominator)),
                            "transport_arithmetic_difference_relative_norm": float(np.sqrt(np.sum((native - reference) ** 2 * area) / denominator)),
                            "metric_quantization_difference_relative_norm": float(np.sqrt(np.sum((reference - consistent) ** 2 * area) / denominator))})
        raw_path = destination.with_name(f"projection_precision_worst_{case}.npz")
        np.savez_compressed(raw_path, initial_x=captured_worst[0], initial_y=captured_worst[1],
                            original_x=captured_worst[2], original_y=captured_worst[3],
                            precise_x=np.asarray(precise[0]), precise_y=np.asarray(precise[1]),
                            quantized_x=np.asarray(quantized[0]), quantized_y=np.asarray(quantized[1]))
        evidence = {"case": case, "steps": 144, "worst_step": worst_step, "worst_native_relative_residual": worst,
                    "failure_count": len(failures), "failures": failures, "first_capture_state_max_differences": differences,
                    "capture_bitwise_state_preserving": bool(np.all(maximum_differences == 0.)),
                    "maximum_capture_state_differences": maximum_differences.tolist(),
                    "captured_residual_at_reference_worst_step": worst_capture_ratio,
                    "trajectory_scope": "unaltered_audit_single_step_not_bitwise_batch_replay",
                    "raw_path": str(raw_path.relative_to(ROOT)), "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    "comparison": results}
        report["cases"].append(evidence)
        save()
        print(json.dumps(evidence, allow_nan=False), flush=True)
        jax.clear_caches()
    report["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
