"""Where does the amplifying free-wave energy concentrate: grid-scale or basin-scale?

If grid-scale (2dx), a barotropic Laplacian viscosity can sink it. If basin-scale,
needs the adjoint-consistent reformulation.
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
e0 = np.array(state.eta); yy = np.arange(grid.ny)
e0[:, yy] += 0.1*np.sin(np.pi*yy/grid.ny)
state = JaxStateG(state.u, state.v, state.T, state.S, jnp.array(e0))

# Step to where amplitude is large but pre-NaN.
for k in range(200):
    state = step(state)
e = np.array(state.eta)
print(f"step 200: max|eta|={float(np.max(np.abs(e))):.3e}")

# Zonal power spectrum of eta at a wet row (lon-axis FFT) -> is energy at 2dx?
# pick a row with lots of wet cells
for j in [30, 60, 90]:
    row = e[:, j] * wm[:, j]
    if wm[:,j].sum() < 100: continue
    # FFT along lon (periodic)
    fh = np.fft.fft(row)
    pw = np.abs(fh)**2
    pw = pw[:len(pw)//2]
    total = pw.sum()
    # grid-scale = top 10% of wavenumbers (2dx-4dx)
    n = len(pw)
    gs = pw[int(n*0.9):].sum()
    print(f"  row j={j} (lat {grid.lat[j]:.1f}): grid-scale(2-4dx) frac = {gs/total:.3f}  "
          f"argmax k={np.argmax(pw[1:])+1}/{n}")

# Meridional profile of |eta| (is it basin-scale seiche or localized?)
print("\nmeridional max|eta| profile (lon-max at each lat):")
prof = np.max(np.abs(e) * wm, axis=0)
for j in range(0, grid.ny, 10):
    bar = '#' * int(prof[j]/max(prof.max(),1)*50)
    print(f"  j={j:>3} lat={grid.lat[j]:>6.1f}: {prof[j]:>10.3e} {bar}")
