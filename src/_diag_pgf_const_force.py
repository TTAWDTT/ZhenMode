"""Is the linear KE_bc growth just 'constant PGF -> u ~ a*t' (correct physics
of an unbalanced steady force), or a discrete instability?
Check: compute the PGF acceleration magnitude at step 0, predict u ~ |PGF|*t,
compare to actual max|u| growth. If they match => it's correct unbalanced-force
physics and the fix is to add the balancing physics (geostrophy / momentum
restoring), not a time-stepping change.
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
from jax_solver_global import (make_solver_global, _compute_pressure_gradient,
                               RHO_0, G_EARTH)
from forcing import air_temp_profile, heat_flux_meridional

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
d = np.load('src/_cache_init.npz')
T_init = d['T']; S_init = d['S']
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])

physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

state0 = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
pgf_x, pgf_y = _compute_pressure_gradient(state0, params)
pgf_mag = jnp.sqrt(pgf_x**2 + pgf_y**2)
max_pgf = float(jnp.max(pgf_mag))
mean_pgf = float(jnp.mean(pgf_mag))
dt = 60.0
print(f"max|PGF acceleration| at t=0: {max_pgf:.4e} m/s^2")
print(f"mean|PGF acceleration|:       {mean_pgf:.4e} m/s^2")
print(f"Predicted max|u| = max|PGF| * t (if unbalanced steady force):")
for nstep in (100, 400, 800, 1200, 1400):
    t = nstep * dt
    print(f"  step {nstep:>4} (t={t/86400:.2f}d): pred max|u| = {max_pgf*t:.4e}")
print()
print("Actual no-adv run (from _diag_noadv_decisive):")
print("  step  100: max|u|=1.35")
print("  step  400: max|u|=6.04")
print("  step  800: max|u|=11.0")
print("  step 1200: max|u|=11.7")
print("  step 1400: max|u|=14.8")
print()
print(f"Ratio actual/pred at 1400: {14.8/(max_pgf*1400*60):.4f}")
print("DONE.")
