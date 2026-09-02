"""Production-config probe: mass fix + tapered cap + real wind + bulk heat.

Does the conservative-divergence fix + tapered polar cap keep the full production
config stable past the day-10 watchdog gate? (Previously blew up / leaked.)
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
from wind_reanalysis import real_wind_forcing

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-5)
tau_x, tau_y = real_wind_forcing(month_idx=(2023-1948)*12+(1-1), grid=grid)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
T_atm = air_temp_profile(grid, T_init[:, :, 0])
step, init_state_fn, _ = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=40.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=2, polar_cap_taper=3)

wm = np.array(grid.wet_mask) > 0.5
state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))

dt = 60.0
steps_per_day = int(86400/dt)
target_days = 12
print(f"production probe: cap=2+3taper, real wind, bulk=40, r_bot=1e-5")
print(f"{'day':>5} {'sum_eta':>12} {'max|eta|':>10} {'max|u|':>10} {'max|T|':>9} {'max|w|':>10}")
for k in range(target_days*steps_per_day):
    state = step(state)
    if (k+1) % steps_per_day == 0:
        e=np.array(state.eta); u=np.array(state.u); T=np.array(state.T)
        day = (k+1)/steps_per_day
        print(f"{day:>5.0f} {float(np.sum(e[wm])):>12.4f} {float(np.max(np.abs(e))):>10.4f} "
              f"{float(np.max(np.abs(u))):>10.4f} {float(np.max(np.abs(T))):>9.3f}", end="")
        if hasattr(state,'w'):
            print(f" {float(np.max(np.abs(np.array(state.w)))):>10.4e}")
        else:
            print()
    if not np.isfinite(np.array(state.eta)).all():
        print(f"  NaN at step {k+1} (day {(k+1)/steps_per_day:.2f})"); break
