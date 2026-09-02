"""Verify the _compute_bt_rho_pgf conservative-gradient fix (day-45 eta blowup).

The day-50 locator found eta blows up at day 45-49 in the equatorial Atlantic
(Gulf of Guinea), a barotropic/free-surface instability. Root cause (code
analysis): _compute_bt_rho_pgf used the bare centered _d_dx/_d_dy (no wet/dry
face masking, no cos(lat) weighting) for the density PGF, while the eta PGF
in the SAME free-surface FB step used _gradient_conservative (masked, cos-
weighted, adjoint-consistent). The non-adjoint rho gradient injects barotropic
energy at coastlines, unarrested near the equator (f->0).

This script runs the production config to day 55 with the FIXED solver and
snapshots daily from day 40. PASS = eta stays bounded (< 8m) through day 55
and no T/u spike. Compare against the pre-fix blowup (eta 5->18m, T->78 by day 49).
"""
import os, sys, time
os.environ.setdefault('JAX_ENABLE_X64', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.92')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.path.insert(0, 'src')
import jax, jax.numpy as jnp, numpy as np
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from woa_data import get_initial_fields
from wind_reanalysis import real_wind_forcing
from forcing import heat_flux_meridional, air_temp_profile, BULK_LAMBDA_DEFAULT
import jax_solver_global as G
from jax_solver_global import make_solver_global

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
nx, ny, nz = grid.nx, grid.ny, grid.nz
lon = np.asarray(grid.lon); lat = np.asarray(grid.lat); z = np.asarray(grid.z)
depth = np.asarray(grid.depth)

T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init, dtype=np.float64); S_init = np.array(S_init, dtype=np.float64)
month_idx = (2023 - 1948) * 12 + (1 - 1)
tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid)
Q_heat = heat_flux_meridional(grid, Q0=50.0)
T_sst = T_init[:, :, 0]
T_atm = air_temp_profile(grid, T_sst)
physics = replace(PhysicsConfig(), nu_h=5e6, nu_bi=0.0, kappa_bi=0.0, r_bot=1e-3)
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=BULK_LAMBDA_DEFAULT, sponge_days=3.0, sponge_cells=16,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0, return_params=True)
stepf = jax.jit(lambda s: G._step_impl(s, params))

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
steps_per_day = int(86400 // 60)

def report(day, state):
    T = np.asarray(state.T); eta = np.asarray(state.eta); u = np.asarray(state.u)
    maxT = float(np.max(T)); maxeta = float(np.nanmax(np.abs(eta))); maxu = float(np.max(np.abs(u)))
    nan = int(np.sum(~np.isfinite(T)))
    ie, je = np.unravel_index(np.argmax(np.abs(eta)), eta.shape)
    print(f"  day{day:3d} max|u|={maxu:7.3f} max|T|={maxT:8.2f} max|eta|={maxeta:7.3f} "
          f"@({lon[ie]:.1f}E,{lat[je]:.1f}N) NaN={nan}", flush=True)

print(f"FIX-VERIFY: conservative rho-PGF gradient. grid {nx}x{ny}x{nz}", flush=True)
print(f"running to day 55 (snap daily from day 40)...", flush=True)
report(0, state)
t0 = time.time()
for day in range(1, 56):
    for _ in range(steps_per_day):
        state = stepf(state)
    if day >= 40 or day % 5 == 0:
        report(day, state)
    if not np.all(np.isfinite(np.asarray(state.T))):
        print(f"  NaN/Inf at day {day} — FAIL (instability)", flush=True)
        break
    if float(np.nanmax(np.abs(np.asarray(state.eta)))) > 15.0:
        print(f"  eta>15m at day {day} — FAIL (blowup)", flush=True)
        break
    if float(np.max(np.asarray(state.T))) > 100.0:
        print(f"  T>100 at day {day} — FAIL (thermal blowup)", flush=True)
        break
maxeta_final = float(np.nanmax(np.abs(np.asarray(state.eta))))
if maxeta_final < 8.0:
    print(f"PASS: max|eta|={maxeta_final:.2f}m at day 55 (was 17.9m pre-fix). "
          f"Barotropic blowup arrested. wall {time.time()-t0:.0f}s", flush=True)
else:
    print(f"FAIL: max|eta|={maxeta_final:.2f}m at day 55 (still blowing up). "
          f"wall {time.time()-t0:.0f}s", flush=True)
