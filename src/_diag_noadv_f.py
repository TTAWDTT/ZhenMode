"""Does Coriolis limit the baroclinic injection (no-advection run)?
Compare f=real vs f=0 with advection OFF. If f=0 grows the SAME => Coriolis
is irrelevant to the injection; the injection is a pure PE->KE conversion bug.
If f=0 grows FASTER => Coriolis partially balances (geostrophic) and the fix
is better Coriolis-PGF coupling.
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
from jax_solver_global import (make_solver_global, RHO_0, G_EARTH, _step_impl)
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

def _zero_adv_u(u, v, w, p):
    z = jnp.zeros_like(v); return z, z
def _zero_adv_s(T, u, v, w, p):
    return jnp.zeros_like(T)
G._advection_flux_form = _zero_adv_u
G._advection_scalar = _zero_adv_s

wm_j = jnp.array(grid.wet_mask); dz_norm_j = jnp.array(dz_norm)
wm_3d = wm_j[:, :, None]
@jax.jit
def ke_bc_j(state):
    u=state.u; v=state.v
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1,keepdims=True); vbt=jnp.sum(va*dz_norm_j,-1,keepdims=True)
    u_bc=ua-ubt; v_bc=va-vbt
    return 0.5*H*jnp.sum((u_bc**2+v_bc**2)*wm_3d)
@jax.jit
def diag_j(state):
    return (jnp.max(jnp.abs(state.u)), ke_bc_j(state), jnp.isfinite(state.eta).all())

def run(label, f_scale):
    physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
    step, init_state_fn, _, params = make_solver_global(
        grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
        T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
        T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
        return_params=True)
    # Scale Coriolis in params
    if f_scale == 0.0:
        params = params._replace(f=jnp.zeros_like(params.f))
    else:
        params = params._replace(f=params.f * f_scale)
    stepf = jax.jit(lambda s: _step_impl(s, params))
    state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    print(f"\n=== {label} (no adv) ===")
    print(f"{'stp':>5} {'max|u|':>10} {'KE_bc':>12}")
    for k in range(800):
        state = stepf(state)
        if (k+1) in (100,400,800):
            mu, kebc, fin = diag_j(state)
            print(f"{k+1:>5} {float(mu):>10.4e} {float(kebc):>12.4e}")
            if not bool(fin): print(f"  NaN step {k+1}"); break

run("f = REAL", 1.0)
run("f = ZERO", 0.0)
print("\nDONE.")
