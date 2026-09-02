"""FINAL TEST: does coupling Coriolis INTO the free-surface step (so geostrophic
balance f*kxu = F_rho establishes within-step) stop the eta drift?

The isolated _free_surface_step_fd has NO Coriolis, so the rotational F_rho
drives a NON-geostrophic flow whose divergence piles up eta (PE grows, KE
saturates). The full linear step HAS Coriolis but applies it BEFORE the FS
step as a separate rotation -- split, so balance doesn't coherently establish.

This builds a small SW step that solves Coriolis + free surface + F_rho
TOGETHER per step, via a semi-implicit solve:
  eta_new = eta - dt*H*div(u_new)                    [implicit in u_new]
  u_new   = u + dt*(f*kxu_new - g*grad(eta_new) + F) [implicit Coriolis + PGF]
This is the linear rotating SW forced by F. Eliminating eta_new gives a
2x2 (u,v) Helmholtz-like system per column... complex. Instead use the
EXACT per-column matrix exponential of the 2x2 [u,v] + eta coupling is hard
on FD.

Simpler first test: apply Coriolis as an EXACT rotation interleaved with the
FB step -- i.e. rotate u by f*dt/2, do FB, rotate again (Strang within the FS
step). If that stops the drift, the fix is Coriolis-FS coupling order.
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
                               _free_surface_step_fd, _compute_bt_rho_pgf,
                               _coriolis_rotation_2d, _barotropic_velocity)
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
g = float(G_EARTH)

_, _, _, params = make_solver_global(
    grid, replace(PhysicsConfig(), nu_h=0.0, nu_bi=0, kappa_bi=0, kappa_h=0.0, r_bot=1e-3),
    60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)
nx,ny,nz = grid.nx, grid.ny, grid.nz
dt = 60.0; dt_half = dt/2.0
f2d = params.f  # (nx, ny)

wm_j = jnp.array(grid.wet_mask); dz_norm_j = jnp.array(dz_norm)
@jax.jit
def energy_j(eta,u,v):
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1); vbt=jnp.sum(va*dz_norm_j,-1)
    return (0.5*H*jnp.sum((ubt**2+vbt**2)*wm_j)+0.5*g*H*jnp.sum(eta**2*wm_j),
            0.5*g*H*jnp.sum(eta**2*wm_j), 0.5*H*jnp.sum((ubt**2+vbt**2)*wm_j))

eta0 = jnp.zeros((nx,ny)); u0=jnp.zeros((nx,ny,nz)); v0=jnp.zeros((nx,ny,nz))
st = JaxStateG(u0,v0,jnp.array(T_init),jnp.array(S_init),eta0)
Fx, Fy = _compute_bt_rho_pgf(st, params)

# Variant 1: plain FB, no Coriolis (the drifting baseline)
step_fb = jax.jit(lambda eta,u,v: _free_surface_step_fd(eta,u,v,params,Fx,Fy))

# Variant 2: Strang-split Coriolis AROUND the FB step (f*dt/2 - FB - f*dt/2)
@jax.jit
def step_fb_cor(eta, u, v):
    # rotate u,v by f*dt/2 (3D, f broadcast)
    u, v = _coriolis_rotation_2d(u, v, f2d, dt_half)
    eta, u, v = _free_surface_step_fd(eta, u, v, params, Fx, Fy, dt_half)
    u, v = _coriolis_rotation_2d(u, v, f2d, dt_half)
    return eta, u, v

def run(label, step_fn, n=4000):
    eta = jnp.zeros((nx,ny)); u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))
    print(f"\n=== {label} ===")
    print(f"{'stp':>5} {'E':>12} {'PE':>12} {'KE':>12} {'max|eta|':>10} {'max|u|':>10}")
    for k in range(n):
        eta, u, v = step_fn(eta, u, v)
        if (k+1)%500==0:
            E,Pe,Ke=energy_j(eta,u,v)
            print(f"{k+1:>5} {float(E):>12.4e} {float(Pe):>12.4e} {float(Ke):>12.4e} "
                  f"{float(jnp.max(jnp.abs(eta))):>10.4e} {float(jnp.max(jnp.abs(u))):>10.4e}")
            if not bool(jnp.isfinite(eta).all()): print(f"  NaN step {k+1}"); break

run("FB only, NO Coriolis (drifting baseline)", step_fb)
run("Strang Coriolis around FB (f/2 - FB - f/2)", step_fb_cor)
print("\nDONE.")
