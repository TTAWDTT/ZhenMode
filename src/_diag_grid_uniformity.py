"""Does the FB amplification depend on dx non-uniformity (cos(lat))?

Test: same seiche bump but measure growth rate vs latitude band. If equatorial
(near-uniform dx) is neutral and high-lat (strongly varying dx) amplifies, the
non-uniform grid is the culprit. Also: does the spherical divergence
(1/cos d/dlambda + d(v cos)/dphi) fix it?
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
lat = np.array(grid.lat)
R = 6371e3
dx = R * np.cos(np.deg2rad(lat)) * np.deg2rad(1.0)   # (ny,) varies 31km..56km
print(f"dx range: {dx.min():.0f}m (lat {lat[np.argmin(dx)]:.1f}) to {dx.max():.0f}m (lat {lat[np.argmax(dx)]:.1f})")
print(f"dx ratio max/min = {dx.max()/dx.min():.3f}")

# Test 1: equatorial seiche (eta signal only in equatorial band, near-uniform dx)
state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
e0 = np.array(state.eta)
# seiche confined to |lat|<15 (equatorial, dx nearly uniform ~111km*cos<15>=107-111km)
eq = np.abs(lat) < 15
e0[:, :] = 0.0
e0[:, eq] = 0.1 * np.sin(np.pi * (lat[eq]+15)/30)
state = JaxStateG(state.u, state.v, state.T, state.S, jnp.array(e0))
print(f"\n--- equatorial seiche (|lat|<15, near-uniform dx) ---")
for k in range(400):
    state = step(state)
    if (k+1) % 50 == 0:
        u=np.array(state.u); e=np.array(state.eta)
        print(f"  stp {k+1:>4}: max|eta|={float(np.max(np.abs(e))):.3e} max|u|={float(np.max(np.abs(u))):.3e}")
    if not np.isfinite(np.array(state.u)).all():
        print(f"  NaN step {k+1}"); break

# Test 2: high-lat seiche (eta signal only at |lat|>45, strongly varying dx)
state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
e0 = np.array(state.eta)
hl = np.abs(lat) > 45
e0[:, :] = 0.0
e0[:, hl] = 0.1 * np.sin(np.pi * (lat[hl]-45)/15 * np.sign(lat[hl]))
state = JaxStateG(state.u, state.v, state.T, state.S, jnp.array(e0))
print(f"\n--- high-lat seiche (|lat|>45, strongly varying dx) ---")
for k in range(400):
    state = step(state)
    if (k+1) % 50 == 0:
        u=np.array(state.u); e=np.array(state.eta)
        print(f"  stp {k+1:>4}: max|eta|={float(np.max(np.abs(e))):.3e} max|u|={float(np.max(np.abs(u))):.3e}")
    if not np.isfinite(np.array(state.u)).all():
        print(f"  NaN step {k+1}"); break
