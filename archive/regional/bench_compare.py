"""
Head-to-head benchmark: numpy integrator vs JAX solver.

Tests:
  1. Per-step wall time (numpy vs JAX steady-state, stable config)
  2. Speedup ratio
  3. JAX solver accuracy: wave speed, mass conservation, dt convergence
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ── numpy solver imports ──
from config import DEFAULT_CONFIG, G_EARTH, RHO_0
from grid import make_grid
from state import initialize_state
from integrator import step as np_step

# ── JAX solver imports ──
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax_solver import make_solver, JaxState


def bench_timing():
    """Timing comparison: numpy vs JAX, stable no-forcing config."""
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    dt = 300.0
    n_steps = 20  # short enough for numpy to stay stable with small perturbation

    print("=" * 70)
    print("  Timing Benchmark: numpy vs JAX")
    print("=" * 70)
    print(f"  Grid:   {grid.nx} x {grid.ny} x {grid.nz}")
    print(f"  dt:     {dt} s")
    print(f"  Steps:  {n_steps}")
    print(f"  Config: small perturbation, no forcing (stable for both)")
    print()

    # ── numpy solver ──
    np_state = initialize_state(grid, physics)
    rng = np.random.RandomState(42)
    np_state.u += rng.normal(scale=0.001, size=np_state.u.shape)
    np_state.v += rng.normal(scale=0.001, size=np_state.v.shape)
    np_state.T += rng.normal(scale=0.001, size=np_state.T.shape)

    # Warmup (1 step)
    np_step(np_state, grid, physics, DEFAULT_CONFIG.time, dt=dt)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        np_step(np_state, grid, physics, DEFAULT_CONFIG.time, dt=dt)
    t_np = time.perf_counter() - t0
    ms_np = t_np / n_steps * 1000

    print(f"  numpy:  {ms_np:.1f} ms/step  ({t_np:.2f}s total)")
    print()

    # ── JAX solver ──
    jax_step_fn, jax_init_state, _ = make_solver(grid, physics, dt)

    # Same initial condition
    np_state2 = initialize_state(grid, physics)
    rng2 = np.random.RandomState(42)
    np_state2.u += rng2.normal(scale=0.001, size=np_state2.u.shape)
    np_state2.v += rng2.normal(scale=0.001, size=np_state2.v.shape)
    np_state2.T += rng2.normal(scale=0.001, size=np_state2.T.shape)

    jax_state = JaxState(
        u=jnp.array(np_state2.u),
        v=jnp.array(np_state2.v),
        T=jnp.array(np_state2.T),
        S=jnp.array(np_state2.S),
        eta=jnp.array(np_state2.eta),
    )

    # Warmup (JIT compile + 1 step)
    t0 = time.perf_counter()
    jax_state = jax_step_fn(jax_state)
    jax.block_until_ready(jax_state.u)
    t_compile = time.perf_counter() - t0

    # Timed
    t0 = time.perf_counter()
    for _ in range(n_steps):
        jax_state = jax_step_fn(jax_state)
    jax.block_until_ready(jax_state.u)
    t_jax = time.perf_counter() - t0
    ms_jax = t_jax / n_steps * 1000

    print(f"  JAX:    {ms_jax:.1f} ms/step  ({t_jax:.2f}s total, compile={t_compile:.2f}s)")
    print()

    speedup = ms_np / ms_jax
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  numpy sim/wall: {n_steps*dt/t_np:.0f}x")
    print(f"  JAX   sim/wall: {n_steps*dt/t_jax:.0f}x")
    print()

    return ms_np, ms_jax, speedup


def bench_accuracy_wave_speed():
    """Accuracy: surface gravity wave speed vs analytical theory."""
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    dt = 150.0

    H_sw = float(np.sum(grid.dz))
    c_theory = np.sqrt(G_EARTH * H_sw)
    dx = float(grid.dx)
    nx = grid.nx
    k = 2.0 * np.pi / (nx * dx)
    omega_theory = c_theory * k

    # Traveling wave IC
    A = 0.01
    ix = np.arange(nx, dtype=np.float64)
    cos_kx = np.cos(k * ix * dx)
    eta_init = A * np.broadcast_to(cos_kx[:, None], (nx, grid.ny)).copy()
    u_bt_target = (c_theory / H_sw) * A * cos_kx
    u_init = np.broadcast_to(u_bt_target[:, None, None], (nx, grid.ny, grid.nz)).copy()

    jax_state = JaxState(
        u=jnp.array(u_init), v=jnp.zeros((nx, grid.ny, grid.nz)),
        T=jnp.full((nx, grid.ny, grid.nz), physics.T_ref),
        S=jnp.full((nx, grid.ny, grid.nz), physics.S_ref),
        eta=jnp.array(eta_init),
    )

    step_fn, _, _ = make_solver(grid, physics, dt)
    _ = step_fn(jax_state)  # warmup

    n_steps = 30
    state = jax_state
    phases, times = [], []
    for i in range(n_steps + 1):
        eta_np = np.array(state.eta)
        eta_fft = np.fft.fft(eta_np, axis=0)
        mode1 = np.mean(eta_fft[1, :])
        phases.append(np.angle(mode1))
        times.append(i * dt)
        if i < n_steps:
            state = step_fn(state)

    times = np.array(times)
    phases = np.array(phases)
    phases_unwrapped = np.unwrap(phases)
    slope, _ = np.polyfit(times, phases_unwrapped, 1)
    omega_measured = -slope
    c_measured = omega_measured / k
    rel_err = abs(c_measured - c_theory) / c_theory

    # Mass conservation
    eta_final = np.array(state.eta)
    mass_drift = abs(np.sum(eta_final) - np.sum(eta_init))

    print("=" * 70)
    print("  Accuracy: Surface Gravity Wave Speed")
    print("=" * 70)
    print(f"  c_theory   = sqrt(g*H) = {c_theory:.2f} m/s")
    print(f"  c_measured = {c_measured:.2f} m/s")
    print(f"  Error:     {rel_err*100:.2f}%")
    print(f"  Mass conservation drift: {mass_drift:.2e}")
    print()

    return rel_err, mass_drift


def bench_accuracy_dt_convergence():
    """Accuracy: dt convergence — compare coarse dt vs fine dt reference."""
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    t_total = 1800.0  # 30 min simulated

    # Initial condition: small smooth perturbation
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    T_pert = 0.01 * np.sin(2 * np.pi * X / (nx * dx)) * np.sin(2 * np.pi * Y / (ny * dy))
    T_pert_3d = np.broadcast_to(T_pert[:, :, None], (nx, ny, nz)).copy()

    state0 = JaxState(
        u=jnp.zeros((nx, ny, nz)), v=jnp.zeros((nx, ny, nz)),
        T=jnp.array(physics.T_ref + T_pert_3d),
        S=jnp.full((nx, ny, nz), physics.S_ref),
        eta=jnp.zeros((nx, ny)),
    )

    # Reference: very fine dt
    dt_ref = 10.0
    n_ref = int(t_total / dt_ref)
    step_ref, _, _ = make_solver(grid, physics, dt_ref)
    state_ref = state0
    for _ in range(n_ref):
        state_ref = step_ref(state_ref)

    print("=" * 70)
    print("  Accuracy: dt Convergence (reference dt=10s, t=1800s)")
    print("=" * 70)
    print(f"  {'dt (s)':>8} {'steps':>6} {'max|ΔT|':>14} {'max|Δu|':>14} {'max|Δeta|':>14}")
    print(f"  {'-'*60}")

    results = []
    for dt_test in [30.0, 60.0, 120.0, 300.0]:
        n_test = int(t_total / dt_test)
        step_test, _, _ = make_solver(grid, physics, dt_test)
        state_test = state0
        for _ in range(n_test):
            state_test = step_test(state_test)

        dT = float(jnp.max(jnp.abs(state_test.T - state_ref.T)))
        du = float(jnp.max(jnp.abs(state_test.u - state_ref.u)))
        deta = float(jnp.max(jnp.abs(state_test.eta - state_ref.eta)))
        print(f"  {dt_test:>8.0f} {n_test:>6} {dT:>14.6e} {du:>14.6e} {deta:>14.6e}")
        results.append((dt_test, dT, du, deta))

    print()
    # Check convergence rate: error should decrease as dt decreases
    # For 2nd-order scheme, error ~ dt^2
    if len(results) >= 2:
        dt1, e1, _, _ = results[0]  # dt=30
        dt2, e2, _, _ = results[-1]  # dt=300
        if e2 > 0 and e1 > 0:
            rate = np.log(e2 / e1) / np.log(dt2 / dt1)
            print(f"  Empirical convergence rate (T): O(dt^{rate:.2f})")
            print(f"  (Expected: O(dt^2) for 2nd-order Strang splitting)")

    print()
    return results


def main():
    ms_np, ms_jax, speedup = bench_timing()
    print()
    wave_err, mass_drift = bench_accuracy_wave_speed()
    print()
    conv_results = bench_accuracy_dt_convergence()

    # Final summary
    print("=" * 70)
    print("  Final Summary")
    print("=" * 70)
    print(f"  Speed:     numpy {ms_np:.1f} ms/step → JAX {ms_jax:.1f} ms/step ({speedup:.2f}x)")
    print(f"  Wave speed error: {wave_err*100:.2f}%")
    print(f"  Mass drift:      {mass_drift:.2e}")
    if conv_results:
        _, dT_300, _, _ = conv_results[-1]
        _, dT_30, _, _ = conv_results[0]
        print(f"  dt=300 vs dt=10 (t=1800s): max|ΔT|={dT_300:.2e}")
        print(f"  dt=30  vs dt=10 (t=1800s): max|ΔT|={dT_30:.2e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
