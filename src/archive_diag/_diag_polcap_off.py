"""polcap OFF long probe: does eta stay bounded to 500 steps? does pole metric stay stable?
Full production config (wind+rhoPGF+bulk), polcap OFF, 500 steps.
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
import jax_solver_global as G
from jax_solver_global import make_solver_global, _barotropic_velocity, JaxStateG
from woa_data import get_initial_fields
from wind_reanalysis import real_wind_forcing
from run_long_integration_global import heat_flux_meridional, air_temp_profile

gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
ocean = np.asarray(grid.ocean_mask, dtype=bool)
H_sw = float(np.sum(grid.dz))

physics = replace(PhysicsConfig(), nu_h=5e6, nu_bi=0.0, kappa_bi=0.0)
Q_heat = heat_flux_meridional(grid, Q0=50.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
tau_x, tau_y = real_wind_forcing(month_idx=(2023-1948)*12, grid=grid)
T_atm = air_temp_profile(grid, T_init[:, :, 0])

def run(label, polcap_rows, polcap_taper, n=500):
    # restore real _compute_bt_rho_pgf (full production)
    step, init_state, _, p = make_solver_global(
        grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat),
        eos_type='linear', T_atm=T_atm, lambda_bulk=40.0,
        sponge_days=3.0, sponge_cells=16, T_init=T_init, S_init=S_init,
        polar_cap_rows=polcap_rows, polar_cap_taper=polcap_taper, return_params=True)
    step = jax.jit(step)
    state = init_state(T_init, S_init)
    print(f"\n=== {label} (polcap {polcap_rows}+{polcap_taper}) FULL prod, {n} steps ===")
    print(f"{'step':>5} {'day':>7} {'max|eta|':>9} {'rms|eta|':>8} {'max|u|':>8} {'max|T|':>7} {'KEbt':>10}")
    for k in range(n):
        state = step(state)
        if k % 50 == 49:
            eta = np.asarray(state.eta)
            u = np.asarray(state.u)
            T = np.asarray(state.T)
            ubt = np.asarray(_barotropic_velocity(state.u, state.v, p)[0])
            vbt = np.asarray(_barotropic_velocity(state.u, state.v, p)[1])
            kebt = 0.5*np.sum((ubt**2+vbt**2)*ocean*H_sw)
            rms = np.sqrt(np.mean((eta*ocean)**2))
            day = (k+1)*60/86400.0
            print(f"{k+1:5d} {day:7.3f} {np.abs(eta).max():9.2f} {rms:8.3f} "
                  f"{np.abs(u).max():8.3f} {np.abs(T*ocean[:,:,None]).max():7.2f} {kebt:10.2e}", flush=True)

run("polcap OFF", 0, 0, n=500)
run("polcap ON (baseline 2+3)", 2, 3, n=500)
