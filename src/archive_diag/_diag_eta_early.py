"""Early-time eta growth diagnostic (iteration 2).

Production config: 360x120x14, dx=111km, nu_h=5e6, dt=60, advection+wind+bulk+sponge+polcap.
Sample every 20 steps for 400 steps (~0.28 day). Track:
  max|eta|, mean|eta|, max|ubt|, max|div(ubt)|, KE_bt, wind power input.

Goal: distinguish monotone injection (eta grows every step) from oscillatory
adjustment (eta oscillates but envelope grows) from steady spin-up (eta -> plateau).
This pins WHERE the 27m/day blowup originates.
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.92')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.path.insert(0, 'src')
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import replace

from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig, G_EARTH, RHO_0
from grid import make_global_grid
from jax_solver_global import (make_solver_global, _barotropic_velocity,
                                _divergence_conservative, JaxStateG)
from woa_data import get_initial_fields
from wind_reanalysis import real_wind_forcing
from run_long_integration_global import heat_flux_meridional, air_temp_profile

# ── build production global grid (mirror run_long_integration_global) ──
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
print("Building global FD grid ...")
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file,
                        smooth_passes=30, min_depth=100.0)
ocean = np.asarray(grid.ocean_mask, dtype=bool)
nx, ny, nz = grid.nx, grid.ny, grid.nz
H_sw = float(np.sum(grid.dz))
print(f"  grid {nx}x{ny}x{nz}, ocean {float(grid.wet_mask.mean()):.1%}, "
      f"lat[{grid.lat[0]:.1f},{grid.lat[-1]:.1f}], H_sw={H_sw:.0f}m")

physics = replace(PhysicsConfig(), nu_h=5e6, nu_bi=0.0, kappa_bi=0.0)
Q_heat = heat_flux_meridional(grid, Q0=50.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
print(f"  T_init range=[{T_init.min():.2f}, {T_init.max():.2f}] C")

month_idx = (2023 - 1948) * 12  # January 2023
tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid)
T_sst = T_init[:, :, 0]
T_atm = air_temp_profile(grid, T_sst)
lambda_bulk = 40.0

step, init_state_global, _ = make_solver_global(
    grid, physics, 60.0,
    forcing=(tau_x, tau_y, Q_heat),
    eos_type='linear',
    T_atm=T_atm, lambda_bulk=lambda_bulk,
    sponge_days=3.0, sponge_cells=16,
    T_init=T_init, S_init=S_init,
    polar_cap_rows=2, polar_cap_taper=3)

state = init_state_global(T_init, S_init)
step = jax.jit(step)
# need params for diagnostics — rebuild minimal via make_solver_global return_params
step_p, init2, diag, p = make_solver_global(
    grid, physics, 60.0,
    forcing=(tau_x, tau_y, Q_heat),
    eos_type='linear',
    T_atm=T_atm, lambda_bulk=lambda_bulk,
    sponge_days=3.0, sponge_cells=16,
    T_init=T_init, S_init=S_init,
    polar_cap_rows=2, polar_cap_taper=3, return_params=True)
step = jax.jit(step_p)
state = init2(T_init, S_init)

wet_2d = np.asarray(p.wet_mask)
dt = 60.0
tx = np.asarray(p.tau_x_2d); ty = np.asarray(p.tau_y_2d)
print(f"  tau max: {np.abs(tx).max():.3f} {np.abs(ty).max():.3f} N/m^2  r_bot={physics.r_bot}")
print()
print(f"{'step':>5} {'day':>7} {'max|eta|':>10} {'rms|eta|':>9} {'max|ubt|':>9} "
      f"{'max|divbt|':>11} {'KE_bt':>11} {'P_wind':>11}")

@jax.jit
def diag_bt(state):
    ubt, vbt = _barotropic_velocity(state.u, state.v, p)
    divbt = _divergence_conservative(ubt * p.wet_mask, vbt * p.wet_mask, p)
    return ubt, vbt, divbt

for k in range(401):
    state = step(state)
    if k < 60:
        sample = (k % 2 == 0)
    elif k < 200:
        sample = (k % 10 == 0)
    else:
        sample = (k % 40 == 0)
    if not sample:
        continue
    eta = np.asarray(state.eta)
    ubt_j, vbt_j, divbt_j = diag_bt(state)
    ubt = np.asarray(ubt_j); vbt = np.asarray(vbt_j)
    divbt = np.asarray(divbt_j)
    ke_bt = 0.5 * np.sum((ubt**2 + vbt**2) * ocean * H_sw)
    pwind = np.sum((tx * ubt + ty * vbt) * ocean)
    day = (k + 1) * dt / 86400.0
    rms_eta = np.sqrt(np.mean((eta * ocean) ** 2))
    print(f"{k+1:5d} {day:7.4f} {np.abs(eta).max():10.3f} {rms_eta:9.4f} "
          f"{np.abs(ubt).max():9.4f} {np.abs(divbt).max():11.2e} {ke_bt:11.3e} {pwind:11.3e}", flush=True)
