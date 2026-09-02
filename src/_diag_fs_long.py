"""Re-examine the 'stable' isolated FS step over LONGER run + compare exactly to
the linear half-step. The isolated _free_surface_step_fd + r_bot=1e-3 + F_rho
was 'stable' (max|u|~0.08) at 1400 steps. But the linear half-step (which
CALLS _free_surface_step_fd) grows to max|u|~1.0. What is the linear half-step
adding that the isolated call does not?

Differences:
  1. linear step applies DIFFUSION to u/v/T/S BEFORE the FS step
  2. linear step applies Coriolis BEFORE the FS step
  3. linear step re-computes F_rho from the (diffused) T/S each step
  4. linear step applies sponge (0 here)
  5. TWO half-steps per "step" (Strang) vs one FS step

This runs the isolated FS step for 4000 steps to see if it TRULY saturates or
just grows slowly (the 1400-step 'stable' may have been too short).
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from jax_solver_global import (make_solver_global, JaxStateG, RHO_0, G_EARTH,
                               _free_surface_step_fd, _compute_bt_rho_pgf)
from forcing import air_temp_profile, heat_flux_meridional
from woa_data import get_initial_fields

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])
wm = np.array(grid.wet_mask) > 0.5
dz = np.array(grid.dz); dz_norm = dz/dz.sum(); H=4000.0

# isolated FS step params (no diffusion, r_bot=1e-3)
_, _, _, params = make_solver_global(
    grid, replace(PhysicsConfig(), nu_h=0.0, nu_bi=0, kappa_bi=0, kappa_h=0.0, r_bot=1e-3),
    60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)
nx,ny,nz = grid.nx, grid.ny, grid.nz

wm_j = jnp.array(grid.wet_mask); dz_norm_j = jnp.array(dz_norm)
@jax.jit
def energy_j(eta,u,v):
    e=eta; ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1); vbt=jnp.sum(va*dz_norm_j,-1)
    return (0.5*H*jnp.sum((ubt**2+vbt**2)*wm_j)+0.5*G_EARTH*H*jnp.sum(e**2*wm_j))
@jax.jit
def diag_j(eta,u):
    return (jnp.max(jnp.abs(eta)), jnp.max(jnp.abs(u)), jnp.isfinite(eta).all())

# CONSTANT F_rho (computed once, like the original isolated test)
eta0 = jnp.zeros((nx,ny)); u0=jnp.zeros((nx,ny,nz)); v0=jnp.zeros((nx,ny,nz))
st = JaxStateG(u0,v0,jnp.array(T_init),jnp.array(S_init),eta0)
Fx, Fy = _compute_bt_rho_pgf(st, params)

step_fs = jax.jit(lambda eta,u,v: _free_surface_step_fd(eta,u,v,params,Fx,Fy))

print("=== Isolated _free_surface_step_fd, CONSTANT F_rho, r_bot=1e-3, 4000 steps ===")
eta = jnp.zeros((nx,ny)); u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))
print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10}")
for k in range(4000):
    eta, u, v = step_fs(eta, u, v)
    if (k+1)%500==0:
        E=float(energy_j(eta,u,v)); me, mu, fin = diag_j(eta,u)
        print(f"{k+1:>5} {E:>12.4e} {float(me):>10.4e} {float(mu):>10.4e}")
        if not bool(fin): print(f"  NaN step {k+1}"); break
print("\nDONE.")
