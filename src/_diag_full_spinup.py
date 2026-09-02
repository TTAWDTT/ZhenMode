"""Full step spinup: does the density-PGF-driven barotropic mode SATURATE (steady
geostrophic setup) or diverge? cap OFF, real initial T,S (=> real F_rho), no wind.
r_bot=1e-3 (regional default). Run long.
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from jax_solver_global import make_solver_global, JaxStateG, RHO_0, G_EARTH
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

def energy(state):
    e=np.array(state.eta); u=np.array(state.u); v=np.array(state.v)
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=np.sum(ua*dz_norm,-1); vbt=np.sum(va*dz_norm,-1)
    return (0.5*H*np.sum((ubt**2+vbt**2)*wm)+0.5*G_EARTH*H*np.sum(e**2*wm),
            0.5*H*np.sum((ubt**2+vbt**2)*wm), 0.5*G_EARTH*H*np.sum(e**2*wm))

for rbot, label in [(1e-3,"cap OFF, no wind, real F_rho, r_bot=1e-3"),
                    (1e-2,"cap OFF, no wind, real F_rho, r_bot=1e-2")]:
    physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=rbot)
    step, init_state_fn, _ = make_solver_global(
        grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
        T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
        T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0)
    state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    E0,_,_ = energy(state)
    print(f"\n=== {label} ===  E0={E0:.4e}")
    print(f"{'stp':>5} {'E/E0':>9} {'max|eta|':>10} {'max|u|':>10} {'sum_eta':>11}")
    for k in range(2000):
        state = step(state)
        if (k+1)%200==0:
            E,Ke,Pe=energy(state)
            e=np.array(state.eta); u=np.array(state.u)
            print(f"{k+1:>5} {E/E0:>9.3f} {float(np.max(np.abs(e))):>10.4e} "
                  f"{float(np.max(np.abs(u))):>10.4e} {float(np.sum(e[wm])):>11.4e}")
        if not np.isfinite(np.array(state.eta)).all():
            print(f"  NaN step {k+1}"); break
