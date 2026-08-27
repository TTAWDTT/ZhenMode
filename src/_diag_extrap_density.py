"""Test: is the coastline injection the TOPOGRAPHIC PGF (density discontinuity at
the seafloor, where masked T_ref cells meet real deep-water density)?
Spectral solver avoids this by using flat-bottom H_sw (every column integrates to
full depth). Test: horizontally EXTRAPOLATE the density anomaly from wet cells
into masked (land/ghost) cells BEFORE the pressure integral, so there is no
density cliff at the seafloor/coast. If the no-adv injection drops => the
topographic/coast PGF is the injector; fix is density extrapolation.
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
                               _step_impl, _density_anomaly)
from forcing import air_temp_profile, heat_flux_meridional

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
d = np.load('src/_cache_init.npz')
T_init = d['T']; S_init = d['S']
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])
dz = np.array(grid.dz); dz_norm = dz/dz.sum(); H=4000.0

physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

# Horizontal extrapolation of a 3D field from wet into masked cells, per-z-slice.
# Iterative nearest-wet fill via lax.fori_loop (bounded graph).
def _extrapolate_h(field3d, wm3d, passes=40):
    def body(_, f):
        nb = (jnp.roll(f,1,0)*jnp.roll(wm3d,1,0) + jnp.roll(f,-1,0)*jnp.roll(wm3d,-1,0)
              + jnp.roll(f,1,1)*jnp.roll(wm3d,1,1) + jnp.roll(f,-1,1)*jnp.roll(wm3d,-1,1))
        cnt = (jnp.roll(wm3d,1,0)+jnp.roll(wm3d,-1,0)+jnp.roll(wm3d,1,1)+jnp.roll(wm3d,-1,1))
        fill = jnp.where(cnt>0, nb/jnp.where(cnt>0,cnt,1.0), f)
        return jnp.where(wm3d>0.5, f, fill)
    return jax.lax.fori_loop(0, passes, body, field3d)

# Patch _compute_hydrostatic_pressure + _compute_bt_rho_pgf to extrapolate density first
_orig_hydro = G._compute_hydrostatic_pressure
_orig_bt_rho = G._compute_bt_rho_pgf
def _hydro_extrap(state, p):
    rho_prime = _density_anomaly(state.T, state.S, p)
    rho_prime = _extrapolate_h(rho_prime, p.wet_mask_z)
    rho_avg = 0.5*(rho_prime[..., :-1] + rho_prime[..., 1:])
    dp = G_EARTH * rho_avg * p.dz_3d
    p_bc = jnp.zeros_like(state.T)
    p_bc = p_bc.at[..., 1:].set(jnp.cumsum(dp, axis=-1))
    p_bt = RHO_0 * G_EARTH * state.eta[:, :, None]
    return p_bt + p_bc
def _bt_rho_extrap(state, p):
    rho_prime = _density_anomaly(state.T, state.S, p)
    rho_prime = _extrapolate_h(rho_prime, p.wet_mask_z)
    rho_avg = 0.5*(rho_prime[..., :-1] + rho_prime[..., 1:])
    dp = G_EARTH * rho_avg * p.dz_3d
    p_bc = jnp.zeros_like(state.T)
    p_bc = p_bc.at[..., 1:].set(jnp.cumsum(dp, axis=-1))
    p_bc_avg = jnp.sum(0.5*(p_bc[..., :-1]+p_bc[..., 1:]) * p.dz_norm, axis=-1)
    return (-G._d_dx(p_bc_avg[:,:,None], p)[:,:,0]/RHO_0,
            -G._d_dy(p_bc_avg[:,:,None], p)[:,:,0]/RHO_0)
G._compute_hydrostatic_pressure = _hydro_extrap
G._compute_bt_rho_pgf = _bt_rho_extrap

# no-advection
def _zero_adv_u(u, v, w, p):
    return jnp.zeros_like(v), jnp.zeros_like(v)
def _zero_adv_s(T, u, v, w, p):
    return jnp.zeros_like(T)
G._advection_flux_form = _zero_adv_u
G._advection_scalar = _zero_adv_s
step_extrap = jax.jit(lambda s: _step_impl(s, params))

wm_j = jnp.array(grid.wet_mask); dz_norm_j = jnp.array(dz_norm); wm_3d = wm_j[:,:,None]
@jax.jit
def energy_j(state):
    e=state.eta; u=state.u; v=state.v
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1); vbt=jnp.sum(va*dz_norm_j,-1)
    return (0.5*H*jnp.sum((ubt**2+vbt**2)*wm_j)+0.5*G_EARTH*H*jnp.sum(e**2*wm_j))
@jax.jit
def diag_j(state):
    ua=0.5*(state.u[...,:-1]+state.u[...,1:]); va=0.5*(state.v[...,:-1]+state.v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1,keepdims=True); vbt=jnp.sum(va*dz_norm_j,-1,keepdims=True)
    kebc=0.5*H*jnp.sum(((ua-ubt)**2+(va-vbt)**2)*wm_3d)
    return (jnp.max(jnp.abs(state.eta)), jnp.max(jnp.abs(state.u)), kebc,
            jnp.isfinite(state.eta).all())

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
print("=== DENSITY EXTRAPOLATED (no coast PGF cliff), NO advection ===")
print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10} {'KE_bc':>12}")
for k in range(800):
    state = step_extrap(state)
    if (k+1) in (200,400,800):
        E=float(energy_j(state)); me, mu, kebc, fin = diag_j(state)
        print(f"{k+1:>5} {E:>12.4e} {float(me):>10.4e} {float(mu):>10.4e} {float(kebc):>12.4e}")
        if not bool(fin): print(f"  NaN step {k+1}"); break
print("\nDONE.")
