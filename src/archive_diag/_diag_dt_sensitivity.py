"""DT sensitivity of the slow density-PGF growth (advection OFF).

Root cause pinned: density PGF is the sole energy source (zeroing
_density_anomaly => ocean stays at rest). The slow growth (max|u| 2.8->12.8
over 1200 steps) is the density PGF under the FD explicit split time stepping.
This tests whether halving dt halves the growth rate:
  - if growth scales with dt => first-order splitting/FB error => semi-implicit
    or exact coupling fixes it.
  - if growth is dt-independent => a structural splitting error (not dt-fixable).
Advection OFF to isolate the linear+residual density PGF path. F_rho ON (real).
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

# disable advection only (keep density PGF real)
G._advection_flux_form = lambda u, v, w, p: (jnp.zeros_like(u), jnp.zeros_like(v))
G._advection_scalar = lambda T, u, v, w, p: jnp.zeros_like(T)

def make_step(dt):
    physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
    _, init_state_fn, _, params = make_solver_global(
        grid, physics, dt, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
        T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
        T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
        return_params=True)
    return jax.jit(lambda s: _step_impl(s, params)), init_state_fn, params

# Run dt=60 for 600 steps (=36000s) vs dt=30 for 1200 steps (=36000s) vs
# dt=15 for 2400 steps (=36000s): same physical time, check max|u| at t=36000s.
for dt, nsteps in [(60, 600), (30, 1200)]:
    step_fn, init_fn, _ = make_step(dt)
    state = init_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    print(f"\n=== dt={dt}, {nsteps} steps (t={dt*nsteps}s), advection OFF, F_rho ON ===")
    for k in range(nsteps):
        state = step_fn(state)
        if (k+1) % (nsteps//3) == 0:
            me, mu, fin = diag_j(state)
            E = float(energy_j(state))
            print(f"  stp {k+1:>5} (t={dt*(k+1)}s): E={E:.4e} max|u|={float(mu):.4e} max|eta|={float(me):.4e}")
            if not bool(fin): print(f"  NaN"); break
print("\nDONE.")
