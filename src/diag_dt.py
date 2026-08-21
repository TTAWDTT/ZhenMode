"""Test: smooth perturbation + smaller dt to isolate instability source."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG
from grid import make_grid
from jax_solver import make_solver, JaxState

grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
physics = DEFAULT_CONFIG.physics

nx, ny, nz = grid.nx, grid.ny, grid.nz
dx, dy = grid.dx, grid.dy

# ── Smooth perturbation (single sine wave, NOT white noise) ──
x = np.arange(nx) * dx
y = np.arange(ny) * dy
X, Y = np.meshgrid(x, y, indexing='ij')
T_pert = 0.001 * np.sin(2 * np.pi * X / (nx * dx)) * np.sin(2 * np.pi * Y / (ny * dy))
T_pert_3d = np.broadcast_to(T_pert[:, :, None], (nx, ny, nz)).copy()

def make_perturbed_state(T_pert_3d):
    state0 = JaxState(
        u=jnp.zeros((nx, ny, nz)),
        v=jnp.zeros((nx, ny, nz)),
        T=jnp.array(physics.T_ref + T_pert_3d),
        S=jnp.full((nx, ny, nz), physics.S_ref),
        eta=jnp.zeros((nx, ny)),
    )
    return state0

def run(label, dt, n_steps, state0):
    step_fn, _, _ = make_solver(grid, physics, dt, forcing=None)
    state = state0
    print(f"\n=== {label} === (dt={dt}s)")
    for i in range(1, n_steps + 1):
        state = step_fn(state)
        if i <= 5 or i % 10 == 0:
            max_u = float(jnp.max(jnp.abs(state.u)))
            max_T = float(jnp.max(jnp.abs(state.T - physics.T_ref)))
            max_eta = float(jnp.max(jnp.abs(state.eta)))
            has_nan = bool(jnp.any(jnp.isnan(state.u)))
            print(f"  Step {i:3d}: max|u|={max_u:.4e}  max|T-Tref|={max_T:.4e}  "
                  f"max|eta|={max_eta:.4e}  {'NaN!' if has_nan else 'OK'}")
            if has_nan or max_u > 1e6:
                print(f"  -> BLOWUP at step {i}")
                return False
    return True

# Test A: smooth perturbation, dt=300
state_smooth = make_perturbed_state(T_pert_3d)
run("Smooth pert, dt=300", 300.0, 100, state_smooth)

# Test B: smooth perturbation, dt=30
state_smooth2 = make_perturbed_state(T_pert_3d)
run("Smooth pert, dt=30", 30.0, 300, state_smooth2)

# Test C: white noise, dt=30 (compare growth rate per unit time)
state_noise = JaxState(
    u=jnp.zeros((nx, ny, nz)),
    v=jnp.zeros((nx, ny, nz)),
    T=physics.T_ref + jax.random.normal(jax.random.PRNGKey(42), (nx, ny, nz)) * 0.001,
    S=jnp.full((nx, ny, nz), physics.S_ref),
    eta=jnp.zeros((nx, ny)),
)
run("White noise, dt=30", 30.0, 300, state_noise)

# Test D: smooth pert with higher viscosity
from config import PhysicsConfig
physics_visc = PhysicsConfig(nu_h=1000.0, kappa_h=1000.0)  # 10x default
step_fn_v, _, _ = make_solver(grid, physics_visc, 300.0, forcing=None)
state_visc = make_perturbed_state(T_pert_3d)
print(f"\n=== High viscosity (nu_h=1000), dt=300 ===")
for i in range(1, 101):
    state_visc = step_fn_v(state_visc)
    if i <= 5 or i % 10 == 0:
        max_u = float(jnp.max(jnp.abs(state_visc.u)))
        max_T = float(jnp.max(jnp.abs(state_visc.T - physics.T_ref)))
        has_nan = bool(jnp.any(jnp.isnan(state_visc.u)))
        print(f"  Step {i:3d}: max|u|={max_u:.4e}  max|T-Tref|={max_T:.4e}  {'NaN!' if has_nan else 'OK'}")
        if has_nan or max_u > 1e6:
            print(f"  -> BLOWUP at step {i}")
            break
