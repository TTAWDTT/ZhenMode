"""Test: is the eta PE drift caused by discrete geostrophic imbalance — F_rho's
gradient operator (_d_dx/_d_dy, centered) inconsistent with the free-surface's
conservative divergence (_divergence_conservative)?

A steady rotational F drives a geostrophic flow u_g. In the discrete, if the
gradient that produces F and the divergence that updates eta are NOT adjoint-
consistent, u_g has a spurious NET divergence => eta piles up monotonically (PE
drifts while KE saturates). The spectral solver's FFT gradient/divergence are
spectrally consistent => u_g is exactly divergence-free => no eta drift.

This rebuilds F_rho using _gradient_conservative (the exact adjoint of the
free-surface's _divergence_conservative) instead of centered _d_dx/_d_dy, and
checks if the eta drift stops.
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
                               _gradient_conservative, _density_anomaly,
                               _compute_hydrostatic_pressure)
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

wm_j = jnp.array(grid.wet_mask); dz_norm_j = jnp.array(dz_norm)
@jax.jit
def energy_j(eta,u,v):
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1); vbt=jnp.sum(va*dz_norm_j,-1)
    return (0.5*g*H*jnp.sum(eta**2*wm_j), 0.5*H*jnp.sum((ubt**2+vbt**2)*wm_j))

eta0 = jnp.zeros((nx,ny)); u0=jnp.zeros((nx,ny,nz)); v0=jnp.zeros((nx,ny,nz))
st = JaxStateG(u0,v0,jnp.array(T_init),jnp.array(S_init),eta0)

# Original F_rho (centered gradient)
Fx0, Fy0 = _compute_bt_rho_pgf(st, params)

# Rebuilt F_rho with CONSERVATIVE gradient (adjoint-consistent with the FS div)
rho_prime = _density_anomaly(st.T, st.S, params) * params.wet_mask_z
rho_avg = 0.5 * (rho_prime[..., :-1] + rho_prime[..., 1:])
dp = G_EARTH * rho_avg * params.dz_3d
p_bc = jnp.zeros_like(st.T).at[..., 1:].set(jnp.cumsum(dp, axis=-1))
p_bc_avg = jnp.sum(0.5 * (p_bc[..., :-1] + p_bc[..., 1:]) * params.dz_norm, axis=-1)
Fx_c, Fy_c = _gradient_conservative(p_bc_avg, params)
Fx_c = -Fx_c / RHO_0; Fy_c = -Fy_c / RHO_0

print(f"max|F_rho centered|   = {float(jnp.max(jnp.abs(Fx0))):.4e}")
print(f"max|F_rho conservative| = {float(jnp.max(jnp.abs(Fx_c))):.4e}")
print(f"max|diff|              = {float(jnp.max(jnp.abs(Fx0-Fx_c))):.4e}")

step_fb_orig = jax.jit(lambda e,u,v: _free_surface_step_fd(e,u,v,params,Fx0,Fy0))
step_fb_cons = jax.jit(lambda e,u,v: _free_surface_step_fd(e,u,v,params,Fx_c,Fy_c))

def run(label, step_fn, n=4000):
    eta = jnp.zeros((nx,ny)); u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))
    print(f"\n=== {label} ===")
    print(f"{'stp':>5} {'PE':>12} {'KE':>12} {'max|eta|':>10} {'max|u|':>10}")
    for k in range(n):
        eta, u, v = step_fn(eta, u, v)
        if (k+1)%500==0:
            Pe,Ke=energy_j(eta,u,v)
            print(f"{k+1:>5} {float(Pe):>12.4e} {float(Ke):>12.4e} "
                  f"{float(jnp.max(jnp.abs(eta))):>10.4e} {float(jnp.max(jnp.abs(u))):>10.4e}")
            if not bool(jnp.isfinite(eta).all()): print(f"  NaN step {k+1}"); break

run("FB + F_rho(centered gradient) [current, drifting]", step_fb_orig)
run("FB + F_rho(conservative gradient) [adjoint-consistent]", step_fb_cons)
print("\nDONE.")
