"""Isolate the post-mass-fix amplification: cap ON vs OFF, no wind, pure free wave.

mass fix closed the leak. But max|u| still grows -> NaN. Where does it nucleate,
and does the tapered cap cause or suppress it?
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from jax_solver_global import make_solver_global, JaxStateG
from forcing import air_temp_profile, heat_flux_meridional
from woa_data import get_initial_fields

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-5)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])


def build(cap_rows, cap_taper):
    step, init_state_fn, _ = make_solver_global(
        grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
        T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
        T_init=T_init, S_init=S_init, polar_cap_rows=cap_rows, polar_cap_taper=cap_taper)
    state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    e0 = np.array(state.eta)
    yy = np.arange(grid.ny); e0[:, yy] += 0.1 * np.sin(np.pi * yy / grid.ny)
    return step, JaxStateG(state.u, state.v, state.T, state.S, jnp.array(e0))


wm = np.array(grid.wet_mask) > 0.5
for label, cr, ct in [("cap OFF", 0, 0), ("cap=2+3taper", 2, 3)]:
    step, state = build(cr, ct)
    print(f"\n=== {label} (no wind, pure free wave, r_bot=1e-5) ===")
    print(f"{'stp':>4} {'max|eta|':>10} {'max|u|':>10} {'argmax(i,j)':>14}")
    for k in range(800):
        state = step(state)
        if (k+1) % 50 == 0:
            u = np.array(state.u); e = np.array(state.eta)
            au = np.abs(u); au_w = au.copy()
            au_w[~np.broadcast_to(wm[:,:,None], au.shape)] = 0
            idx = np.unravel_index(np.argmax(au_w), au.shape)
            print(f"{k+1:>4} {float(np.max(np.abs(e))):>10.4e} {float(au_w[idx]):>10.4e} "
                  f"{f'({idx[0]},{idx[1]})':>14}")
        if not np.isfinite(np.array(state.u)).all():
            print(f"  NaN at step {k+1}"); break
