"""Isolate the step-12 j=0 blowup source with polar_cap_rows=2.

Tracks the per-step max|eta| and max|u| AT j=0 vs INTERIOR, and the
zonal-uniformity of j=0, for the first 20 steps — to see whether the
explosion is (a) the free-surface eta at j=0, (b) the cap-edge du/dy,
or (c) the metric Laplacian. No code change to the solver; pure observation.
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
month_idx = (2023 - 1948) * 12
tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid)
T_atm = air_temp_profile(grid, T_init[:, :, 0])

step, init_state_fn, _ = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat),
    eos_type='linear', T_atm=T_atm, lambda_bulk=40.0,
    sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=2)

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
print(f"{'stp':>4} {'max|u|j0':>10} {'max|u|int':>10} {'max|T|j0':>10} {'max|eta|j0':>12} {'max|eta|int':>12} {'j0_eta_std':>10} {'NaN':>4}")
for k in range(20):
    state = step(state)
    u = np.array(state.u); T = np.array(state.T); eta = np.array(state.eta)
    # j=0 row (south pole), lon-axis stats
    u_j0 = np.max(np.abs(u[:, 0, :]))
    T_j0 = np.max(np.abs(T[:, 0, :]))
    eta_j0 = np.max(np.abs(eta[:, 0]))
    eta_j0_std = np.std(eta[:, 0])
    # interior j=20..100
    u_int = np.max(np.abs(u[:, 20:100, :]))
    eta_int = np.max(np.abs(eta[:, 20:100]))
    nan = int((~np.isfinite(u)).sum() + (~np.isfinite(eta)).sum())
    print(f"{k+1:>4} {u_j0:>10.3e} {u_int:>10.3e} {T_j0:>10.3e} {eta_j0:>12.4e} {eta_int:>12.4e} {eta_j0_std:>10.3e} {nan:>4}")
    if nan > 0:
        print(f"  -> NaN at step {k+1}; j=0 row exploded. eta_j0/max_int ratio = {eta_j0/max(eta_int,1e-30):.2e}")
        break
