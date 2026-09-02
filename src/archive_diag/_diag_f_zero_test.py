"""DECISIVE: is the production blowup caused by Coriolis x F_rho splitting?

Evidence so far:
- Full physics (Coriolis+nu_h+r_bot) + F_rho => NaN step 1194, max|u| 27@1000.
- Isolated FS step, r_bot=1e-3, NO Coriolis, + F_rho => STABLE (max|u|~0.08,
  eta saturating), no NaN through 1400 steps.
=> Coriolis is the discriminator. This runs the FULL step (_step_impl) with the
real params but f ZEROED, to confirm: with f=0, full physics + F_rho is stable;
with real f, it blows up. That pins the injector to the Coriolis/F_rho split.
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
                               FDPhysParams, _step_impl)
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

physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

wm_j = jnp.array(grid.wet_mask)
dz_norm_j = jnp.array(dz_norm)
@jax.jit
def energy_j(state):
    e=state.eta; u=state.u; v=state.v
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1); vbt=jnp.sum(va*dz_norm_j,-1)
    return (0.5*H*jnp.sum((ubt**2+vbt**2)*wm_j)+0.5*G_EARTH*H*jnp.sum(e**2*wm_j))
@jax.jit
def diag_j(state):
    return (jnp.max(jnp.abs(state.eta)), jnp.max(jnp.abs(state.u)),
            jnp.sum(state.eta*wm_j), jnp.isfinite(state.eta).all())

# patched params with f=0 (everything else identical)
params_f0 = params._replace(f=jnp.zeros_like(params.f))
step_f0 = jax.jit(lambda state: _step_impl(state, params_f0))

def run(step_fn, label, n=1200):
    state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    E0 = float(energy_j(state))
    print(f"\n=== {label} ===  E0={E0:.4e}")
    print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10} {'sum_eta':>11}")
    for k in range(n):
        state = step_fn(state)
        if (k+1)%200==0:
            E=float(energy_j(state))
            me, mu, se, fin = diag_j(state)
            print(f"{k+1:>5} {E:>12.4e} {float(me):>10.4e} {float(mu):>10.4e} {float(se):>11.4e}")
            if not bool(fin):
                print(f"  NaN step {k+1}"); break

run(step_f0, "FULL step, f=0 (NO Coriolis), real F_rho, nu_h=1e3, r_bot=1e-3", n=1200)
print("\nDONE.")
