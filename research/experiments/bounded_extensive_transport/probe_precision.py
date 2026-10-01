"""Fixed-Q storage/arithmetic controls; not replay of the barotropic trajectory."""
import hashlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
FROZEN_REF = "cb38ba7"
SOURCE = subprocess.check_output(["git", "show", f"{FROZEN_REF}:src/finite_volume.py"], cwd=ROOT)
frozen = types.ModuleType("frozen_finite_volume")
exec(compile(SOURCE, "frozen_finite_volume", "exec"), frozen.__dict__)


def main():
    nx, ny = 64, 16
    depth = np.full((nx, ny), 50.)
    depth[10:25, 4:10] = 15.
    depth[35:40, 8:12] = 0.
    geometry = frozen.build_geometry(np.linspace(0., 360., nx + 1),
                                     np.linspace(-66., 66., ny + 1), [0., 5., 20., 50.], depth)
    geometry = jax.tree_util.tree_map(lambda value: jnp.asarray(value, jnp.float32), geometry)
    phase = 2. * np.pi * np.arange(nx)[:, None] / nx
    eta = jnp.asarray(np.broadcast_to(.2 * np.cos(phase), (nx, ny)) * (depth > 0.), jnp.float32)
    initial_volume = frozen.surface_volume(geometry, eta)
    fluxes = frozen.closed_surface_fluxes(geometry.east_area * .02, geometry.north_area * -.01)
    output = ROOT / "results/industrial_alignment/bounded_transport_precision.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "fixed_prescribed_Q_not_barotropic_replay_or_gpu_performance",
              "frozen_ref": FROZEN_REF, "kernel_sha256": hashlib.sha256(SOURCE).hexdigest(),
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol_sha256": hashlib.sha256(Path(__file__).with_name("protocol.md").read_bytes()).hexdigest(),
              "backend": jax.default_backend(), "jax_version": jax.__version__,
              "input_sha256": hashlib.sha256(depth.tobytes() + np.asarray(initial_volume).tobytes()
                                              + b"".join(np.asarray(field).tobytes() for field in fluxes)).hexdigest(),
              "shape": list(initial_volume.shape), "steps": 100, "dt_seconds": 60., "rows": []}
    wet = np.asarray(initial_volume) > 0.
    modes = ("native32", "arithmetic64_store32", "V32_N64", "V64_N32", "V64_N64", "expansion32")
    for constant in (True, False):
        concentration = np.ones(initial_volume.shape + (2,)) * [12., 35.]
        if not constant:
            concentration[..., 0] += np.sin(phase)[..., None] * np.exp(-np.asarray(geometry.center_depth) / 500.)
            concentration[..., 1] += .1 * np.sin(np.linspace(-62., 62., ny))[None, :, None]
        initial_content = initial_volume[..., None] * jnp.asarray(concentration, jnp.float32)
        initial_concentration = np.asarray(initial_content, dtype=np.float64)[wet] / np.asarray(initial_volume, dtype=np.float64)[wet, None]
        for mode in modes:
            volume_dtype = jnp.float64 if mode in {"V64_N64", "V64_N32"} else jnp.float32
            content_dtype = jnp.float64 if mode in {"V64_N64", "V32_N64"} else jnp.float32
            initial = (initial_volume.astype(volume_dtype), initial_content.astype(content_dtype))
            if mode == "expansion32":
                initial += (jnp.zeros_like(initial_volume), jnp.zeros_like(initial_content))

            @jax.jit
            def step(fields):
                if mode == "native32":
                    state = frozen.ExtensiveState(*fields)
                elif mode == "expansion32":
                    state = frozen.ExtensiveState(fields[0].astype(jnp.float64) + fields[2].astype(jnp.float64),
                                                   fields[1].astype(jnp.float64) + fields[3].astype(jnp.float64))
                else:
                    state = frozen.ExtensiveState(fields[0].astype(jnp.float64), fields[1].astype(jnp.float64))
                result = frozen.advance_contents(geometry, state, fluxes, 60.)
                if mode == "expansion32":
                    high_volume = result.state.volume.astype(jnp.float32)
                    high_content = result.state.content.astype(jnp.float32)
                    next_fields = (high_volume, high_content,
                                   (result.state.volume - high_volume.astype(jnp.float64)).astype(jnp.float32),
                                   (result.state.content - high_content.astype(jnp.float64)).astype(jnp.float32))
                else:
                    next_fields = (result.state.volume.astype(volume_dtype), result.state.content.astype(content_dtype))
                return next_fields, result.valid

            warmed, valid = step(initial)
            jax.block_until_ready(warmed)
            fields = initial
            started = time.perf_counter()
            completed = 0
            for index in range(100):
                fields, valid = step(fields)
                if not bool(valid):
                    break
                completed = index + 1
            elapsed = time.perf_counter() - started
            volume, content = (np.asarray(field, dtype=np.float64) for field in fields[:2])
            if mode == "expansion32":
                volume += np.asarray(fields[2], dtype=np.float64)
                content += np.asarray(fields[3], dtype=np.float64)
            final_concentration = content[wet] / volume[wet, None]
            under = initial_concentration.min(axis=0) - final_concentration.min(axis=0)
            over = final_concentration.max(axis=0) - initial_concentration.max(axis=0)
            bound_error = float(max(0., float(under.max()), float(over.max())))
            relative_constant = float(np.max(np.abs(final_concentration - initial_concentration) / np.maximum(np.abs(initial_concentration), 1.)))
            content_budget = (np.sum(content, axis=(0, 1, 2)) - np.sum(np.asarray(initial_content, dtype=np.float64), axis=(0, 1, 2))) / np.sum(np.abs(np.asarray(initial_content, dtype=np.float64)), axis=(0, 1, 2))
            row = {"mode": mode, "constant": constant, "completed_steps": completed,
                   "maximum_absolute_bound_excursion": bound_error,
                   "minimum_undershoot": under.tolist(), "maximum_overshoot": over.tolist(),
                   "maximum_relative_constant_change": relative_constant if constant else None,
                   "content_budget_relative_error": content_budget.tolist(),
                   "volume_budget_m3": float(np.sum(volume - np.asarray(initial_volume, dtype=np.float64))),
                   "stored_bytes": sum(field.size * field.dtype.itemsize for field in fields),
                   "warmed_cpu_gate_and_update_seconds": elapsed,
                   "bound_pass": completed == 100 and bound_error <= 2e-6}
            report["rows"].append(row)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"{mode}/constant={constant}: steps={completed}, bound={bound_error:.3e}, bytes={row['stored_bytes']}, seconds={elapsed:.3f}", flush=True)
    reproduced = any(row["mode"] == "native32" and not row["constant"] and not row["bound_pass"] for row in report["rows"])
    report["native_failure_reproduced"] = reproduced
    report["status"] = "controls_completed" if reproduced else "failure_not_reproduced"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not reproduced:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
