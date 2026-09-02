"""Verify the polar-cap-edge hypothesis: is there a sharp meridional gradient
at the j=1/j=2 (cap interior) interface that drives the j=0 uniform explosion?

Reproduce polcap=2, step 3 (before explosion gets huge), and print the
meridional T profile at a single longitude for j=0..6 to see the cap edge.
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
from jax_solver_global import make_solver_global
from forcing import air_temp_profile, heat_flux_meridional
from wind_reanalysis import real_wind_forcing
from woa_data import get_initial_fields

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
physics = replace(PhysicsConfig(), nu_h=5e6, nu_bi=0, kappa_bi=0)
Q_heat = heat_flux_meridional(grid, Q0=50.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
tau_x, tau_y = real_wind_forcing(month_idx=(2023-1948)*12, grid=grid)
T_atm = air_temp_profile(grid, T_init[:, :, 0])
step, init_state_fn, _ = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat),
    eos_type='linear', T_atm=T_atm, lambda_bulk=40.0,
    sponge_days=0.0, sponge_cells=0, T_init=T_init, S_init=S_init,
    polar_cap_rows=2)
state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
lat = grid.lat
print(f"{'stp':>3} | T at lon0, surface, j=0..7 (cap rows = j=0,1)")
for k in range(5):
    state = step(state)
    T = np.array(state.T)
    row = T[0, :8, 0]   # lon 0, surface, j=0..7
    eta = np.array(state.eta)
    erow = eta[0, :8]
    print(f"{k+1:>3} | T={[f'{v:9.3e}' for v in row]}")
    print(f"    | eta={[f'{v:9.3e}' for v in erow]}")
    if not np.all(np.isfinite(T)):
        print("  NaN"); break
