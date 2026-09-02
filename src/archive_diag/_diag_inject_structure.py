"""Characterize the LINEAR injection (no-advection run): is the growing shear
in a single horizontal wavenumber, a single vertical mode, or broadband?
Saves eta/u spectra + vertical shear profile at a few steps.
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
dz = np.array(grid.dz); dz_norm = dz/dz.sum(); H=4000.0
wm = np.array(grid.wet_mask) > 0.5

physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

def _zero_adv_u(u, v, w, p):
    z = jnp.zeros_like(v); return z, z
def _zero_adv_s(T, u, v, w, p):
    return jnp.zeros_like(T)
G._advection_flux_form = _zero_adv_u
G._advection_scalar = _zero_adv_s
step_no_adv = jax.jit(lambda s: _step_impl(s, params))

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
print("=== no-advection injection structure ===")
print(f"{'stp':>5} {'max|u|':>10} {'KE_2d':>12} {'KE_bc':>12} {'bc/total':>8}")
for k in range(800):
    state = step_no_adv(state)
    if (k+1) in (1, 100, 400, 800):
        u = np.array(state.u); v = np.array(state.v)
        # barotropic vs baroclinic split
        ua = 0.5*(u[...,:-1]+u[...,1:]); va = 0.5*(v[...,:-1]+v[...,1:])
        ubt = (ua*dz_norm[None,None,:]).sum(-1)
        vbt = (va*dz_norm[None,None,:]).sum(-1)
        ubt3 = ubt[:,:,None]; vbt3 = vbt[:,:,None]
        u_bc = ua - ubt3; v_bc = va - vbt3  # baroclinic (on interfaces)
        ke_bt = 0.5*H*np.sum((ubt**2+vbt**2)*wm)
        ke_bc = 0.5*H*np.sum((u_bc**2+v_bc**2)*wm*np.array(grid.dz_3d[:,:,0]>0,dtype=float)[...,None] if False else 0.0)
        # simpler: total KE of interface velocities
        ke_full = 0.5*H*np.sum((ua**2+va**2)*wm[...,None])
        ke_bt_full = 0.5*H*np.sum((ubt3**2+vbt3**2)*wm[...,None])
        ke_bc_full = ke_full - ke_bt_full
        print(f"{k+1:>5} {np.max(np.abs(u)):>10.4e} {ke_bt_full:>12.4e} {ke_bc_full:>12.4e} {ke_bc_full/(ke_full+1e-30):>8.3f}")
print("\nDONE.")
