"""Energy budget of the FB free-surface step: is E growing, and is it the wall?

Track E = 0.5*g*H*sum(eta^2) + 0.5*H*sum(ubt^2+vbt^2) per step. If E grows, the
scheme injects energy. Also test r_bot=0 (pure neutral) vs r_bot=1e-5.
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
H = 4000.0
dz = np.array(grid.dz); dz_norm = dz/dz.sum()

def energy(state):
    e = np.array(state.eta); u = np.array(state.u); v = np.array(state.v)
    ua = 0.5*(u[...,:-1]+u[...,1:]); va = 0.5*(v[...,:-1]+v[...,1:])
    ubt = np.sum(ua*dz_norm,-1); vbt = np.sum(va*dz_norm,-1)
    Ke = 0.5*H*np.sum((ubt**2+vbt**2)*wm)
    Pe = 0.5*G_EARTH*H*np.sum(e**2*wm)
    return Ke+Pe, Ke, Pe

for rbot in [0.0, 1e-5, 1e-4, 1e-3]:
    physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=rbot)
    step, init_state_fn, _ = make_solver_global(
        grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
        T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
        T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0)
    state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    e0 = np.array(state.eta); yy = np.arange(grid.ny)
    e0[:, yy] += 0.1*np.sin(np.pi*yy/grid.ny)
    state = JaxStateG(state.u, state.v, state.T, state.S, jnp.array(e0))
    E0,_,_ = energy(state)
    print(f"\n=== r_bot={rbot:.0e} (drag={1/(1+rbot*30):.6f}) ===")
    prev = E0
    for k in range(200):
        state = step(state)
        if (k+1) % 40 == 0:
            E,Ke,Pe = energy(state)
            print(f"  stp {k+1:>4}: E={E:.4e} (Ke={Ke:.3e} Pe={Pe:.3e}) "
                  f"growth/step={E/prev:.6f}  total {E/E0:.4e}")
            prev = E
        if not np.isfinite(np.array(state.eta)).all():
            print(f"  NaN step {k+1}"); break
