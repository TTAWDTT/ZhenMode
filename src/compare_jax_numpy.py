"""
Correctness comparison: JAX solver vs numpy/scipy solver.

Runs both solvers for N steps from identical initial conditions
(perturbed rest state) and compares the prognostic variables.

Expected: max abs diff < 1e-10 (float64, same algorithm).
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ── numpy solver ──
from config import DEFAULT_CONFIG
from grid import make_grid
from state import ModelState
from integrator import step as numpy_step

# ── JAX solver ──
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax_solver import make_solver, JaxState


def main():
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    dt = 300.0

    print("=== JAX vs Numpy Correctness Comparison ===")
    print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}")
    print(f"dt: {dt}s")
    print()

    # ── Create identical initial conditions ──
    # Use numpy RNG for both — generate perturbation, pass to JAX
    rng = np.random.default_rng(42)
    shape = (grid.nx, grid.ny, grid.nz)

    u_init = rng.normal(0, 0.01, shape)
    v_init = rng.normal(0, 0.01, shape)
    T_init = np.full(shape, physics.T_ref) + rng.normal(0, 0.01, shape)
    S_init = np.full(shape, physics.S_ref)
    eta_init = np.zeros((grid.nx, grid.ny))

    # numpy state
    np_state = ModelState(
        u=u_init.copy(), v=v_init.copy(),
        T=T_init.copy(), S=S_init.copy(),
        eta=eta_init.copy(),
    )

    # JAX state (same data)
    jax_state = JaxState(
        u=jnp.array(u_init), v=jnp.array(v_init),
        T=jnp.array(T_init), S=jnp.array(S_init),
        eta=jnp.array(eta_init),
    )

    # ── Build JAX solver ──
    step_fn, _, _ = make_solver(grid, physics, dt)

    # Warmup JIT
    _ = step_fn(jax_state)
    jax.block_until_ready(jax_state.u)

    # ── Run N steps ──
    n_steps = 5

    # numpy
    t0 = time.perf_counter()
    for _ in range(n_steps):
        np_state = numpy_step(np_state, grid, physics, DEFAULT_CONFIG.time, dt)
    t_np = time.perf_counter() - t0

    # JAX
    t0 = time.perf_counter()
    for _ in range(n_steps):
        jax_state = step_fn(jax_state)
    jax.block_until_ready(jax_state.u)
    t_jax = time.perf_counter() - t0

    # ── Compare ──
    np_u = np_state.u
    np_v = np_state.v
    np_T = np_state.T
    np_S = np_state.S

    jax_u = np.array(jax_state.u)
    jax_v = np.array(jax_state.v)
    jax_T = np.array(jax_state.T)
    jax_S = np.array(jax_state.S)

    print(f"Steps: {n_steps}")
    print()
    print(f"{'Var':<6} {'Max Abs Diff':<18} {'Max Rel Diff':<18} {'numpy max':<14} {'jax max':<14}")
    print("-" * 70)

    for name, a, b in [("u", np_u, jax_u), ("v", np_v, jax_v),
                        ("T", np_T, jax_T), ("S", np_S, jax_S)]:
        abs_diff = np.abs(a - b)
        max_abs = np.max(abs_diff)
        # relative diff: |a-b| / max(|a|, |b|, 1e-30)
        scale = np.maximum(np.abs(a), np.abs(b))
        scale = np.maximum(scale, 1e-30)
        max_rel = np.max(abs_diff / scale)
        print(f"{name:<6} {max_abs:<18.6e} {max_rel:<18.6e} {np.max(np.abs(a)):<14.6e} {np.max(np.abs(b)):<14.6e}")

    print()
    print(f"numpy time: {t_np:.3f}s ({t_np/n_steps*1000:.1f} ms/step)")
    print(f"JAX   time: {t_jax:.3f}s ({t_jax/n_steps*1000:.1f} ms/step)")
    print(f"Speedup: {t_np/t_jax:.2f}x")

    # Overall pass/fail
    all_diff = max(
        np.max(np.abs(np_u - jax_u)),
        np.max(np.abs(np_v - jax_v)),
        np.max(np.abs(np_T - jax_T)),
        np.max(np.abs(np_S - jax_S)),
    )
    print()
    if all_diff < 1e-10:
        print(f"PASS: max diff = {all_diff:.2e} < 1e-10")
    elif all_diff < 1e-6:
        print(f"WARNING: max diff = {all_diff:.2e} (acceptable but not exact)")
    else:
        print(f"FAIL: max diff = {all_diff:.2e} > 1e-6")


if __name__ == "__main__":
    main()
