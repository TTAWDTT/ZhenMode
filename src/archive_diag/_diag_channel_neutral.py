"""Is the FB free-surface scheme energy-neutral on a CLEAN all-wet periodic channel?

If yes -> the amplification is mask/wall-induced. If no -> the FB scheme itself
is non-neutral on the lat-lon (cos-varying dx) grid, even without land.

All-wet channel: wet_mask all 1, no land, no cap, no wind. Same seiche bump.
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

# Force ALL-WET channel: override wet_mask to all-open.
import jax_solver_global as JSG
_orig_make = JSG.make_solver_global

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
# We can't easily flip wet_mask after build; instead test the REAL masked grid
# but with a SMOOTH initial eta (no coastline-crossing signal): a zonally-uniform
# seiche (constant in lon) so no coastline gradient at all.
e0 = np.array(state.eta)
yy = np.arange(grid.ny)
e0[:] = 0.1 * np.sin(np.pi * yy / grid.ny)   # zonally uniform: eta(i,j) depends on j only
state = JaxStateG(state.u, state.v, state.T, state.S, jnp.array(e0))

wm = np.array(grid.wet_mask) > 0.5
print("zonally-uniform seiche (no coastline-crossing eta signal):")
print(f"{'stp':>4} {'sum_eta':>12} {'max|eta|':>10} {'max|u|':>10}")
for k in range(600):
    state = step(state)
    if (k+1) % 50 == 0:
        e=np.array(state.eta); u=np.array(state.u)
        print(f"{k+1:>4} {float(np.sum(e[wm])):>12.4f} {float(np.max(np.abs(e))):>10.4e} "
              f"{float(np.max(np.abs(u))):>10.4e}")
    if not np.isfinite(np.array(state.u)).all():
        print(f"  NaN at step {k+1}"); break
