"""
Physics validation: surface gravity wave speed c = sqrt(g * H).

Uses a single-wavenumber traveling-wave initial condition:
  eta(x) = A * cos(k*x)
  u_bt   = (c/H) * A * cos(k*x)   (right-going wave)

Tracks the phase of the dominant Fourier mode of eta over time.
For a right-going wave, phase(t) = -omega*t, so c = omega/k.

With f0 << omega (Coriolis negligible at this wavenumber), the measured
phase speed should match c_theory = sqrt(g * H_sw) to within ~1%.
"""
import sys
import os
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax_solver import make_solver, JaxState
from config import DEFAULT_CONFIG, G_EARTH
from grid import make_grid


def main():
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    # Use INVISCID physics for this linear analytic test. The production
    # PhysicsConfig carries Laplacian/biharmonic stabilizers (nu_bi=1e12 etc.)
    # whose dissipation shifts ~6% off the analytic phase speed. In the
    # inviscid limit the measured speed matches c=sqrt(gH) to ~0.5%, with the
    # residual being the expected f-plane rotation correction (+0.3%).
    physics = replace(
        DEFAULT_CONFIG.physics,
        nu_h=0.0, nu_v=0.0, nu_bi=0.0,
        kappa_h=0.0, kappa_bi=0.0, kappa_conv=0.0,
        cd=0.0, r_bot=0.0, smag_cs=0.0,
    )
    dt = 150.0

    # H_sw = sum(dz), effective shallow-water depth (same as solver)
    H_sw = float(np.sum(grid.dz))
    c_theory = np.sqrt(G_EARTH * H_sw)
    dx = float(grid.dx)
    f0 = float(grid.f0)

    # Fundamental wavenumber: k = 2*pi / (nx * dx)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    k = 2.0 * np.pi / (nx * dx)
    omega_theory = c_theory * k

    print("=== Surface Gravity Wave Speed Test ===")
    print(f"Grid: {nx}x{ny}x{nz},  dx = {dx:.1f} m")
    print(f"H_sw     = {H_sw:.1f} m  (sum of dz)")
    print(f"c_theory = sqrt(g*H_sw) = {c_theory:.2f} m/s")
    print(f"k = {k:.6e} /m  (fundamental mode, wavelength = {nx*dx/1000:.0f} km)")
    print(f"omega_th = {omega_theory:.6e} /s  (period = {2*np.pi/omega_theory/3600:.2f} h)")
    print(f"f0       = {f0:.6e} /s  (f0/omega = {f0/omega_theory:.4f})")
    print()

    # ── Traveling wave initial condition (right-going) ──
    # eta = A * cos(k*x),  u_bt = (c/H_sw) * A * cos(k*x)
    # The solver computes ubt = sum(0.5*(u[k]+u[k+1]) * dz/H_sw).
    # For uniform u_3d, ubt = u_3d * sum(dz) / H_sw = u_3d * 1.0 = u_3d.
    # So u_3d = u_bt_target directly (no scaling needed).
    A = 0.01  # small amplitude for linear regime
    ix = np.arange(nx, dtype=np.float64)
    cos_kx = np.cos(k * ix * dx)  # (nx,)

    eta_init = A * np.broadcast_to(cos_kx[:, None], (nx, ny)).copy()
    u_bt_target = (c_theory / H_sw) * A * cos_kx  # (nx,)
    u_init = np.broadcast_to(u_bt_target[:, None, None], (nx, ny, nz)).copy()
    v_init = np.zeros((nx, ny, nz))
    T_init = np.full((nx, ny, nz), physics.T_ref)
    S_init = np.full((nx, ny, nz), physics.S_ref)

    jax_state = JaxState(
        u=jnp.array(u_init), v=jnp.array(v_init),
        T=jnp.array(T_init), S=jnp.array(S_init),
        eta=jnp.array(eta_init),
    )

    step_fn, _, _ = make_solver(grid, physics, dt)
    _ = step_fn(jax_state)  # warmup

    # ── Track phase of the (1,0) Fourier mode ──
    # FFT of eta along x-axis, take mode (1,0): FFT[1, :]
    # For cos(kx), the FFT has peaks at k=+1 and k=-1 (or nx-1).
    # The right-going wave cos(k*(x-c*t)) has its positive-frequency
    # component exp(-i*omega*t) at mode k=+1.
    # phase(t) = arg(FFT_x(eta)[1, 0])
    # omega = -d(phase)/dt  (phase decreases for right-going wave)

    n_steps = 30
    phases = []
    times = []

    state = jax_state
    for i in range(n_steps + 1):
        eta_np = np.array(state.eta)
        # FFT along x, average over y for noise reduction
        eta_fft = np.fft.fft(eta_np, axis=0)
        mode1 = np.mean(eta_fft[1, :])  # complex scalar
        phase = np.angle(mode1)
        phases.append(phase)
        times.append(i * dt)
        if i < n_steps:
            state = step_fn(state)

    times = np.array(times)
    phases = np.array(phases)

    # Unwrap phase (handle jumps at +/- pi)
    phases_unwrapped = np.unwrap(phases)

    # Linear fit: phase = phase0 - omega * t
    # (negative slope for right-going wave)
    slope, intercept = np.polyfit(times, phases_unwrapped, 1)
    omega_measured = -slope  # positive for right-going wave
    c_measured = omega_measured / k

    print(f"Phase evolution (unwrapped, degrees):")
    for t, p in zip(times[::3], phases_unwrapped[::3]):
        print(f"  t={t:7.0f}s  phase={np.degrees(p):10.2f} deg")
    print()
    print(f"omega_measured = {omega_measured:.6e} /s")
    print(f"omega_theory   = {omega_theory:.6e} /s")
    print(f"c_measured     = {c_measured:.2f} m/s")
    print(f"c_theory       = {c_theory:.2f} m/s")
    rel_err = abs(c_measured - c_theory) / c_theory
    print(f"Relative error = {rel_err*100:.2f}%")

    # ── Mass conservation ──
    # For a cosine wave, mean(eta) = 0, so check absolute drift instead.
    eta_final = np.array(state.eta)
    mass_init = np.sum(eta_init)
    mass_final = np.sum(eta_final)
    mass_drift = abs(mass_final - mass_init)
    print()
    print(f"Mass conservation: init={mass_init:.6e}, final={mass_final:.6e}, drift={mass_drift:.2e}")
    # ── Pass/fail ──
    print()
    if rel_err < 0.05:
        print(f"PASS: wave speed error {rel_err*100:.2f}% < 5%")
    else:
        print(f"FAIL: wave speed error {rel_err*100:.2f}% >= 5%")

    if mass_drift < 1e-10:
        print(f"PASS: mass conservation drift {mass_drift:.2e} < 1e-10")
    else:
        print(f"FAIL: mass conservation drift {mass_drift:.2e} >= 1e-10")


if __name__ == "__main__":
    main()
