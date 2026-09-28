"""Same frozen real-grid predictors, different native Poisson solve settings."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import jax_solver_global as current
from config import DEFAULT_CONFIG
from stage_budgets import _StageRecorder
from verify_debug_integration import make_smoke_fixture


class PredictorRecorder(_StageRecorder):
    def column_projection(self, velocity_x, velocity_y, corrected_x, corrected_y):
        self.predictor = (velocity_x, velocity_y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/industrial_alignment/projection_convergence.json")
    parser.add_argument("--bathy", default=DEFAULT_CONFIG.bathymetry_file)
    args = parser.parse_args()
    frozen_source = subprocess.check_output(["git", "show", "cce8f87:src/jax_solver_global.py"], cwd=ROOT)
    frozen = types.ModuleType("frozen_projection_convergence")
    frozen.__file__ = str(ROOT / "src/jax_solver_global.py")
    sys.modules[frozen.__name__] = frozen
    exec(compile(frozen_source, frozen.__file__, "exec"), frozen.__dict__)
    inherited_cap = os.environ.get("OCEAN_PAV_NITER")
    os.environ["OCEAN_PAV_NITER"] = "150"
    grid, physics, initial_temperature, initial_salinity, atmosphere, forcing = make_smoke_fixture(
        2., args.bathy, 2e14)
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "src").glob("*.py")) + [ROOT / "scripts/verify_debug_integration.py", Path(__file__).resolve()]
    report = {
        "scope": "native_projection_not_moving_volume_conservation_or_climate_qualification",
        "status": "running", "frozen_kernel": "cce8f87", "grid": [grid.nx, grid.ny, grid.nz],
        "forcing": "explicitly_synthetic", "frozen_projection_environment": {"OCEAN_PAV_NITER": "150"},
        "inherited_environment_not_used": inherited_cap,
        "provenance": {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
            "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            "frozen_git_blob_sha256": hashlib.sha256(frozen_source).hexdigest(),
            "bathymetry_sha256": hashlib.sha256(Path(args.bathy).read_bytes()).hexdigest(),
            "jax_version": jax.__version__, "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        }, "inputs": [], "results": [],
    }

    def save():
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    for dtype in ("float64", "float32"):
        for case in ("baseline", "ice"):
            settings = dict(forcing=forcing, T_atm=atmosphere, lambda_bulk=80.,
                            T_init=initial_temperature, S_init=initial_salinity,
                            mode_split=True, dt_bt=50., nu_nsub="cfl", use_scan=True, dtype=dtype,
                            conservative_kv=True, localize_conv=True, project_adv_vel=True, fct_adv=True,
                            mixed_layer_depth_m=20. if case == "ice" else None,
                            dynamic_ice=case == "ice", return_params=True)
            plain_step, initialize, _, frozen_params, _ = frozen.make_solver_global(grid, physics, 600., **settings)
            state = initialize(T_init=initial_temperature, S_init=initial_salinity)

            @jax.jit
            def capture(previous):
                recorder = PredictorRecorder(frozen_params)
                updated = frozen._step_impl(previous, frozen_params, budget=recorder)
                return updated, recorder.predictor

            @jax.jit
            def advance(previous, count):
                def body(index, carry):
                    updated, predictor = capture(carry[0])
                    return updated, predictor
                return jax.lax.fori_loop(0, count, body, (previous, (previous.u, previous.v)))

            audited_first, _ = capture(state)
            ordinary_first = plain_step(state)
            differences = [float(jnp.max(jnp.abs(first - second))) for first, second in zip(audited_first, ordinary_first, strict=True)]
            if any(differences):
                raise ValueError(f"frozen capture changed state: {differences}")
            params_by_preconditioner = {name: current.make_solver_global(
                grid, physics, 600., projection_niter=150, projection_preconditioner=name, **settings)[3]
                for name in ("none", "jacobi")}
            solvers = {name: jax.jit(lambda velocity_x, velocity_y, cap, params=params: current._project_column_divergence(
                velocity_x, velocity_y, params, params.dt, n_iter=cap)) for name, params in params_by_preconditioner.items()}
            area = np.asarray(frozen_params.dx_2d, dtype=np.float64) * frozen_params.dy * np.asarray(frozen_params.wet_mask)
            volume = area[..., None] * np.asarray(frozen_params.dz_node, dtype=np.float64) * np.asarray(frozen_params.wet_mask_z)
            completed = 0
            for sample_step in (1, 72, 144):
                state, predictor = advance(state, sample_step - completed)
                state.T.block_until_ready()
                completed = sample_step
                if not all(bool(jnp.all(jnp.isfinite(field))) for field in state):
                    raise ValueError("nonfinite frozen trajectory")
                input_path = destination.with_name(f"projection_predictor_{dtype}_{case}_{sample_step}.npz")
                np.savez_compressed(input_path, velocity_x=np.asarray(predictor[0]), velocity_y=np.asarray(predictor[1]))
                report["inputs"].append({"dtype": dtype, "case": case, "step": sample_step,
                                         "path": str(input_path.relative_to(ROOT)),
                                         "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                                         "first_capture_state_max_differences": differences})
                top_before = np.asarray(current._vertical_transport_iface(*predictor, frozen_params)[..., 0], dtype=np.float64)
                norm_before = np.sum(top_before ** 2 * area)
                energy_before = sum(np.sum(np.asarray(field, dtype=np.float64) ** 2 * volume) for field in predictor)
                for name, project in solvers.items():
                    params = params_by_preconditioner[name]
                    for cap in (150, 300, 600, 1200):
                        corrected = project(*predictor, cap)
                        corrected[0].block_until_ready()
                        started = time.perf_counter()
                        for repeat in range(3):
                            corrected = project(*predictor, cap)
                            corrected[0].block_until_ready()
                        elapsed = (time.perf_counter() - started) / 3.
                        top_after = np.asarray(current._vertical_transport_iface(*corrected, params)[..., 0], dtype=np.float64)
                        ratio = float(np.sqrt(np.sum(top_after ** 2 * area) / norm_before)) if norm_before > 0. else 0.
                        energy_after = sum(np.sum(np.asarray(field, dtype=np.float64) ** 2 * volume) for field in corrected)
                        energy_ratio = float(energy_after / energy_before) if energy_before > 0. else 1.
                        finite = bool(np.all(np.isfinite(top_after)) and np.isfinite(energy_ratio))
                        threshold = 1e-9 if dtype == "float64" else 5e-5
                        energy_margin = 1e-12 if dtype == "float64" else 5e-6
                        result = {"dtype": dtype, "case": case, "step": sample_step,
                                  "preconditioner": name, "cap": cap, "rtol": current.projection_config(params)["rtol"],
                                  "native_area_l2_relative_residual": ratio if np.isfinite(ratio) else None,
                                  "wet_energy_ratio": energy_ratio if np.isfinite(energy_ratio) else None,
                                  "compiled_cpu_seconds": elapsed, "finite": finite,
                                  "gate_pass": bool(finite and ratio <= threshold and energy_ratio <= 1. + energy_margin)}
                        report["results"].append(result)
                        print(json.dumps(result, allow_nan=False), flush=True)
                save()
            jax.clear_caches()
    report["eligible_configurations"] = [
        {"preconditioner": name, "cap": cap}
        for name in ("none", "jacobi") for cap in (150, 300, 600, 1200)
        if all(result["gate_pass"] for result in report["results"] if result["preconditioner"] == name and result["cap"] == cap)]
    report["status"] = "complete"
    save()
    print(json.dumps({"eligible_configurations": report["eligible_configurations"]}), flush=True)


if __name__ == "__main__":
    main()
