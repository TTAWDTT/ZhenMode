"""Locate WHERE the free-wave velocity amplification nucleates (after mass fix).

If max|u| concentrates at coastlines => spurious PGF from centered gradient of
masked eta reaching into land zeros (symmetric twin of the div leak). If interior
=> FB scheme itself amplifying on the lat-lon grid.
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
step, init_state_fn, _ = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0)

wm = np.array(grid.wet_mask) > 0.5
nx, ny = grid.nx, grid.ny
dry = ~wm
nbr_dry = (np.roll(dry,1,0).astype(int)+np.roll(dry,-1,0).astype(int)
           +np.pad(dry[:,:-1],((0,0),(1,0))).astype(int)
           +np.pad(dry[:,1:],((0,0),(0,1))).astype(int))
is_coast = wm & (nbr_dry >= 1)

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
e0 = np.array(state.eta)
yy = np.arange(ny); e0[:, yy] += 0.1 * np.sin(np.pi * yy / ny)
state = JaxStateG(state.u, state.v, state.T, state.S, jnp.array(e0))

print(f"{'stp':>4} {'max|u|':>11} {'argmax(i,j)':>14} {'is_coast?':>10} {'coast_frac_top1%':>18}")
for k in range(700):
    state = step(state)
    if (k+1) % 50 == 0:
        u = np.array(state.u)
        au = np.abs(u)
        # argmax over wet 3D
        au_w = au.copy(); au_w[~np.broadcast_to(wm[:,:,None], au.shape)] = 0
        idx = np.unravel_index(np.argmax(au_w), au.shape)
        i, j, _ = idx
        # fraction of top-1% |u| cells that are coastline
        thr = np.percentile(au_w[au_w>0], 99)
        top = (au_w >= thr)
        top2d = np.any(top, axis=2)
        coast_frac = float(np.sum(top2d & is_coast)) / max(float(np.sum(top2d)),1.0)
        print(f"{k+1:>4} {float(au_w[idx]):>11.4e} {f'({i},{j})':>14} "
              f"{'YES' if is_coast[i,j] else 'no':>10} {coast_frac:>18.2f}")
    if not np.isfinite(np.array(state.u)).all():
        print("  NaN"); break
