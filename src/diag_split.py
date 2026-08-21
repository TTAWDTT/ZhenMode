"""Diagnose WHERE instability grows: linear half-step vs nonlinear step vs splitting.

Tracks kinetic + potential energy through each sub-step of Strang splitting.
Also tests variants to isolate the unstable mode.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG, RHO_0, G_EARTH, ALPHA_T
from grid import make_grid
from jax_solver import (
    make_solver, JaxState, _compute_params, _init_state,
    _linear_half_step, _explicit_full_step, _step_impl,
    _compute_momentum_tendency, _compute_tracer_tendency,
    _compute_hydrostatic_pressure, _density_anomaly,
    _barotropic_velocity,
)

grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
physics = DEFAULT_CONFIG.physics
nx, ny, nz = grid.nx, grid.ny, grid.nz
dx, dy = grid.dx, grid.dy

# Smooth perturbation (depth-uniform)
x = np.arange(nx) * dx
y = np.arange(ny) * dy
X, Y = np.meshgrid(x, y, indexing='ij')
T_pert = 0.001 * np.sin(2 * np.pi * X / (nx * dx)) * np.sin(2 * np.pi * Y / (ny * dy))
T_pert_3d = np.broadcast_to(T_pert[:, :, None], (nx, ny, nz)).copy()

def total_energy(state, p):
    """Kinetic + potential (eta^2) energy."""
    ke = 0.5 * RHO_0 * jnp.sum(state.u**2 + state.v**2) * float(jnp.sum(jnp.array(grid.dz)))
    pe = 0.5 * RHO_0 * G_EARTH * jnp.sum(state.eta**2) * dx * dy
    return float(ke + pe)

def max_vals(state):
    return (
        float(jnp.max(jnp.abs(state.u))),
        float(jnp.max(jnp.abs(state.T - physics.T_ref))),
        float(jnp.max(jnp.abs(state.eta))),
    )

# ── Test 1: Energy through sub-steps ──
print("=" * 70)
print("Test 1: Energy tracking through Strang sub-steps (dt=300)")
print("=" * 70)
dt = 300.0
params = _compute_params(grid, physics, dt, forcing=None)
state0 = JaxState(
    u=jnp.zeros((nx, ny, nz)),
    v=jnp.zeros((nx, ny, nz)),
    T=jnp.array(physics.T_ref + T_pert_3d),
    S=jnp.full((nx, ny, nz), physics.S_ref),
    eta=jnp.zeros((nx, ny)),
)

dt_half = dt / 2.0
state = state0
for i in range(1, 81):
    s0 = state
    s_after_L1 = _linear_half_step(s0, params, dt_half)
    s_after_N = _explicit_full_step(s_after_L1, params, dt)
    s_after_L2 = _linear_half_step(s_after_N, params, dt_half)
    state = s_after_L2

    if i <= 5 or i % 10 == 0:
        mu0, mT0, me0 = max_vals(s0)
        mu1, mT1, me1 = max_vals(s_after_L1)
        mu2, mT2, me2 = max_vals(s_after_N)
        mu3, mT3, me3 = max_vals(s_after_L2)
        e0 = total_energy(s0, params)
        e1 = total_energy(s_after_L1, params)
        e2 = total_energy(s_after_N, params)
        e3 = total_energy(s_after_L2, params)
        print(f"\nStep {i:3d}:")
        print(f"  Start:     max|u|={mu0:.4e}  max|T|={mT0:.4e}  max|eta|={me0:.4e}  E={e0:.4e}")
        print(f"  After L/2: max|u|={mu1:.4e}  max|T|={mT1:.4e}  max|eta|={me1:.4e}  E={e1:.4e}  dE={e1-e0:+.4e}")
        print(f"  After N:   max|u|={mu2:.4e}  max|T|={mT2:.4e}  max|eta|={me2:.4e}  E={e2:.4e}  dE={e2-e1:+.4e}")
        print(f"  After L/2: max|u|={mu3:.4e}  max|T|={mT3:.4e}  max|eta|={me3:.4e}  E={e3:.4e}  dE={e3-e2:+.4e}")

        if mu3 > 1e6 or bool(jnp.isnan(state.u).any()):
            print(f"\n  -> BLOWUP at step {i}")
            break

# ── Test 2: Nonlinear step only (no splitting) ──
print("\n" + "=" * 70)
print("Test 2: Nonlinear step ONLY (no linear step), dt=300")
print("=" * 70)
state2 = JaxState(
    u=jnp.zeros((nx, ny, nz)),
    v=jnp.zeros((nx, ny, nz)),
    T=jnp.array(physics.T_ref + T_pert_3d),
    S=jnp.full((nx, ny, nz), physics.S_ref),
    eta=jnp.zeros((nx, ny)),
)
for i in range(1, 81):
    state2 = _explicit_full_step(state2, params, dt)
    if i <= 5 or i % 10 == 0:
        mu, mT, me = max_vals(state2)
        has_nan = bool(jnp.isnan(state2.u).any())
        print(f"  Step {i:3d}: max|u|={mu:.4e}  max|T|={mT:.4e}  max|eta|={me:.4e}  {'NaN!' if has_nan else 'OK'}")
        if has_nan or mu > 1e6:
            print(f"  -> BLOWUP at step {i}")
            break

# ── Test 3: Linear step only (no nonlinear) ──
print("\n" + "=" * 70)
print("Test 3: Linear step ONLY (no nonlinear step), dt=300")
print("=" * 70)
state3 = JaxState(
    u=jnp.zeros((nx, ny, nz)),
    v=jnp.zeros((nx, ny, nz)),
    T=jnp.array(physics.T_ref + T_pert_3d),
    S=jnp.full((nx, ny, nz), physics.S_ref),
    eta=jnp.zeros((nx, ny)),
)
for i in range(1, 81):
    state3 = _linear_half_step(state3, params, dt_half)
    if i <= 5 or i % 10 == 0:
        mu, mT, me = max_vals(state3)
        has_nan = bool(jnp.isnan(state3.u).any())
        print(f"  Step {i:3d}: max|u|={mu:.4e}  max|T|={mT:.4e}  max|eta|={me:.4e}  {'NaN!' if has_nan else 'OK'}")
        if has_nan or mu > 1e6:
            print(f"  -> BLOWUP at step {i}")
            break

# ── Test 4: Depth-varying T perturbation (first baroclinic mode) ──
print("\n" + "=" * 70)
print("Test 4: Depth-varying T perturbation (baroclinic mode), dt=300")
print("=" * 70)
z = np.array(grid.z)
z_norm = (z - z.min()) / (z.max() - z.min())  # 0 to 1
T_pert_bc = T_pert[:, :, None] * z_norm[None, None, :] * 2.0  # amplitude 2x for same mean
state4 = JaxState(
    u=jnp.zeros((nx, ny, nz)),
    v=jnp.zeros((nx, ny, nz)),
    T=jnp.array(physics.T_ref + T_pert_bc),
    S=jnp.full((nx, ny, nz), physics.S_ref),
    eta=jnp.zeros((nx, ny)),
)
step_fn, _, _ = make_solver(grid, physics, dt, forcing=None)
for i in range(1, 81):
    state4 = step_fn(state4)
    if i <= 5 or i % 10 == 0:
        mu, mT, me = max_vals(state4)
        has_nan = bool(jnp.isnan(state4.u).any())
        print(f"  Step {i:3d}: max|u|={mu:.4e}  max|T|={mT:.4e}  max|eta|={me:.4e}  {'NaN!' if has_nan else 'OK'}")
        if has_nan or mu > 1e6:
            print(f"  -> BLOWUP at step {i}")
            break

# ── Test 5: u perturbation only (no T perturbation) ──
print("\n" + "=" * 70)
print("Test 5: u perturbation only (no T pert), dt=300")
print("=" * 70)
u_pert = 0.001 * np.sin(2 * np.pi * X / (nx * dx)) * np.sin(2 * np.pi * Y / (ny * dy))
u_pert_3d = np.broadcast_to(u_pert[:, :, None], (nx, ny, nz)).copy()
state5 = JaxState(
    u=jnp.array(u_pert_3d),
    v=jnp.zeros((nx, ny, nz)),
    T=jnp.full((nx, ny, nz), physics.T_ref),
    S=jnp.full((nx, ny, nz), physics.S_ref),
    eta=jnp.zeros((nx, ny)),
)
for i in range(1, 81):
    state5 = step_fn(state5)
    if i <= 5 or i % 10 == 0:
        mu, mT, me = max_vals(state5)
        has_nan = bool(jnp.isnan(state5.u).any())
        print(f"  Step {i:3d}: max|u|={mu:.4e}  max|T|={mT:.4e}  max|eta|={me:.4e}  {'NaN!' if has_nan else 'OK'}")
        if has_nan or mu > 1e6:
            print(f"  -> BLOWUP at step {i}")
            break
