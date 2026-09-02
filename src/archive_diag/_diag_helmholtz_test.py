"""Helmholtz solver feasibility test (the gate for semi-implicit F_rho coupling).

The semi-implicit free-surface step solves
    (I - dt^2*g*H*L) eta_new = rhs
where L = div(grad) is the self-consistent conservative Laplacian (so that the
operator is the exact discrete adjoint pair, hence SPD on the wet domain). This
verifies:
  1. the operator is SPD on the real masked global grid (CG converges),
  2. CG converges in few iterations under jit (it must be fast inside the step),
  3. the solve preserves the conservative mass property (sum_eta flat),
  4. it actually balances a steady forcing (the whole point).
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax
import jax.numpy as jnp
import numpy as np
import jax.scipy.sparse.linalg as jsp
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from jax_solver_global import (make_solver_global, JaxStateG, RHO_0, G_EARTH,
                               _divergence_conservative, _gradient_conservative,
                               _compute_bt_rho_pgf)
from forcing import air_temp_profile, heat_flux_meridional
from woa_data import get_initial_fields

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
physics = replace(PhysicsConfig(), nu_h=0.0, nu_bi=0, kappa_bi=0, kappa_h=0.0, r_bot=0.0)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

wm = np.array(grid.wet_mask) > 0.5
nx, ny, nz = grid.nx, grid.ny, grid.nz
H = 4000.0
dt = 60.0
dt_half = dt / 2.0
g = float(G_EARTH)

# Self-consistent Laplacian: L(eta) = div(grad(eta)) using the conservative pair.
def lap_consistent(eta):
    gx, gy = _gradient_conservative(eta, params)
    return _divergence_conservative(gx, gy, params)

# Helmholtz operator: (I - dt_half^2 * g * H * L). SPD because -L is PSD on the
# wet domain (it is div(grad) = -grad^* grad under the area-weighted inner
# product, i.e. L = -A where A is SPD). Masked so land stays 0.
def helmholtz_op(eta):
    return params.wet_mask * (eta - (dt_half**2) * g * H * lap_consistent(eta))

# --- F_rho forcing (the steady baroclinic PGF that pumps energy explicitly) ---
eta0 = jnp.zeros((nx, ny))
u0 = jnp.zeros((nx, ny, nz)); v0 = jnp.zeros((nx, ny, nz))
state_for_rho = JaxStateG(u0, v0, jnp.array(T_init), jnp.array(S_init), eta0)
F_rho_x, F_rho_y = _compute_bt_rho_pgf(state_for_rho, params)
print(f"max|F_rho_x|={float(jnp.max(jnp.abs(F_rho_x))):.4e} "
      f"max|F_rho_y|={float(jnp.max(jnp.abs(F_rho_y))):.4e}")

# rhs = eta0 - dt_half*H*div(u=0) - dt_half^2*H*div(F_rho). With u=0, eta0=0:
# rhs = -dt_half^2 * H * div(F_rho).
div_F = _divergence_conservative(F_rho_x * params.wet_mask, F_rho_y * params.wet_mask, params)
rhs = -((dt_half**2) * H * div_F) * params.wet_mask

# --- Test 1: SPD / convergence ---
print("\n[Test 1] CG convergence on the Helmholtz system (real masked grid)...")
@jax.jit
def solve(rhs):
    eta_sol, info = jsp.cg(helmholtz_op, rhs, maxiter=200, tol=1e-10)
    return eta_sol, info
eta_sol, info = solve(rhs)
info = None if info is None else int(info)
resid = float(jnp.linalg.norm(helmholtz_op(eta_sol) - rhs))
print(f"  CG info={info}  resid={resid:.3e}  max|eta_sol|={float(jnp.max(jnp.abs(eta_sol))):.4e}")

# --- Test 2: mass conservation (sum over wet must be ~0 because rhs integrates to 0) ---
sum_wet = float(jnp.sum(np.array(eta_sol) * wm))
sum_rhs = float(jnp.sum(rhs * wm))
print(f"\n[Test 2] Mass conservation:")
print(f"  sum(rhs*wet)     = {sum_rhs:.3e}  (must be ~0: forcing div telescopes)")
print(f"  sum(eta_sol*wet) = {sum_wet:.3e}  (must be ~0: Helmholtz preserves it)")

# --- Test 3: does the balanced eta actually cancel F_rho? ---
# In the semi-implicit step, u_new = (u + dt*(-g*grad(eta_new) + F_rho)) * drag.
# The point: g*grad(eta_sol) should largely cancel F_rho at steady state.
gx, gy = _gradient_conservative(eta_sol, params)
pgf_bal_x = g * np.array(gx); pgf_bal_y = g * np.array(gy)
Fx = np.array(F_rho_x); Fy = np.array(F_rho_y)
resid_force = np.sqrt(np.mean((pgf_bal_x - Fx)**2 + (pgf_bal_y - Fy)**2))
force_mag = np.sqrt(np.mean(Fx**2 + Fy**2))
print(f"\n[Test 3] Steady balance (g*grad(eta_sol) cancels F_rho?):")
print(f"  rms|force|               = {force_mag:.4e}")
print(f"  rms|g*grad(eta) - F_rho| = {resid_force:.4e}  "
      f"(={100*resid_force/max(force_mag,1e-30):.2f}% of force)")
print(f"  -> small % means the semi-implicit eta balances the density force within-step.")

print("\nDONE.")
