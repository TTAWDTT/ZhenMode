"""Find the first step + field where NaN appears with polar_cap_rows=2.

Mirrors the g10d_fulldealias_polcap config exactly. Steps one-at-a-time,
checks each state field for finiteness, locates the exact step + cell where
NaN nucleates (before the polar-cap zonal mean smears it everywhere).
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
month_idx = (2023 - 1948) * 12 + (1 - 1)
tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid)
T_atm = air_temp_profile(grid, T_init[:, :, 0])

step, init_state_fn, _ = make_solver_global(
    grid, physics, 60.0,
    forcing=(tau_x, tau_y, Q_heat),
    eos_type='linear',
    T_atm=T_atm, lambda_bulk=40.0,
    sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init,
    polar_cap_rows=2)

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))

for k in range(400):
    state = step(state)
    fields = {'u': state.u, 'v': state.v, 'T': state.T, 'S': state.S, 'eta': state.eta}
    bad = [n for n, f in fields.items() if not bool(jnp.all(jnp.isfinite(f)))]
    if bad:
        print(f"step {k+1}: NaN/Inf first appears in {bad}")
        for n in bad:
            f = np.array(fields[n])
            nf = ~np.isfinite(f)
            ij = np.unravel_index(np.argmax(nf), f.shape)
            print(f"  {n}: first non-finite at {n}{tuple(ij)} = {f[ij]} ; total non-finite={nf.sum()}/{f.size}")
        # also report max magnitudes of the still-finite fields
        for n in ['u','v','T','eta']:
            f = np.array(fields[n])
            fin = f[np.isfinite(f)]
            if fin.size:
                print(f"  {n} (finite only): max|.|={np.max(np.abs(fin)):.4g}")
        break
else:
    print(f"ran 400 steps clean, no NaN")
    print(f"  final max|u|={float(jnp.max(jnp.abs(state.u))):.4f} max|T|={float(jnp.max(state.T)):.3f} max|eta|={float(jnp.max(jnp.abs(state.eta))):.4f}")
