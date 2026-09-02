"""Isolate: does the LINEAR half-step alone (diffusion+Coriolis+FB free surface
WITH density F_rho) inject energy? Zero out the explicit residual entirely
(advection + depth-varying baroclinic PGF all gone — only the linear step runs).
  flat   -> linear step is energy-conserving; ALL injection is the residual PGF
  growth -> the FB free-surface step with F_rho itself injects energy
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
import jax_solver_global as G
from jax_solver_global import (make_solver_global, JaxStateG, RHO_0, G_EARTH,
                               _step_impl)
from forcing import air_temp_profile, heat_flux_meridional

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
d = np.load('src/_cache_init.npz')
T_init = d['T']; S_init = d['S']
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])
wm = np.array(grid.wet_mask) > 0.5
dz = np.array(grid.dz); dz_norm = dz/dz.sum(); H=4000.0

physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

# Zero BOTH residual components -> explicit step does nothing; only linear step runs.
def _zero_res(state, p):
    z = jnp.zeros_like(state.u); return z, z
def _zero_trac_res(state, p):
    z = jnp.zeros_like(state.T); return z, z
G._compute_momentum_residual = _zero_res
G._compute_tracer_residual = _zero_trac_res
step_lin = jax.jit(lambda s: _step_impl(s, params))

wm_j = jnp.array(grid.wet_mask); dz_norm_j = jnp.array(dz_norm)
@jax.jit
def energy_j(state):
    e=state.eta; u=state.u; v=state.v
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1); vbt=jnp.sum(va*dz_norm_j,-1)
    return (0.5*H*jnp.sum((ubt**2+vbt**2)*wm_j)+0.5*G_EARTH*H*jnp.sum(e**2*wm_j))
@jax.jit
def diag_j(state):
    return (jnp.max(jnp.abs(state.eta)), jnp.max(jnp.abs(state.u)),
            jnp.isfinite(state.eta).all())

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
print("=== LINEAR step ONLY (residual zeroed; FB free surface + F_rho + Coriolis + diffusion) ===")
print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10}")
for k in range(1400):
    state = step_lin(state)
    if (k+1) in (200,400,600,800,1000,1200,1400):
        E=float(energy_j(state)); me, mu, fin = diag_j(state)
        print(f"{k+1:>5} {E:>12.4e} {float(me):>10.4e} {float(mu):>10.4e}")
        if not bool(fin): print(f"  NaN step {k+1}"); break
print("\nDONE.")
