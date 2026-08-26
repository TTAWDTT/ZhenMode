"""Verify the conservative-divergence fix closes the free-surface mass leak.

Pure free-wave test (no wind, no cap, no sponge): 0.1m eta seiche bump.
Before fix: sum_eta drifted +1996 -> -4607 over 200 steps (mass leak).
After fix: should stay flat (machine-zero residual).
Also checks max|eta| stays bounded (free wave should not amplify).
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
state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
e0 = np.array(state.eta)
yy = np.arange(grid.ny); e0[:, yy] += 0.1 * np.sin(np.pi * yy / grid.ny)
state = JaxStateG(state.u, state.v, state.T, state.S, jnp.array(e0))
sum0 = float(np.sum(e0[wm]))
print(f"initial sum_eta={sum0:.4f}  max|eta|={float(np.max(np.abs(e0))):.4f}")
print(f"\n{'stp':>4} {'sum_eta':>12} {'dsum':>12} {'max|eta|':>11} {'max|u|':>11}")
prev = sum0
for k in range(1000):
    state = step(state)
    e = np.array(state.eta); u = np.array(state.u)
    s = float(np.sum(e[wm]))
    if (k + 1) % 50 == 0 or k < 3:
        print(f"{k+1:>4} {s:>12.4f} {s-prev:>12.4e} {float(np.max(np.abs(e))):>11.4e} "
              f"{float(np.max(np.abs(u))):>11.4e}")
    prev = s
    if not np.isfinite(e).all():
        print("  NaN"); break
print(f"\nfinal drift = {s - sum0:+.4e}  ({(s-sum0)/sum0*100:+.3f}% of initial)")
