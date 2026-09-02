"""Isolate the eta-buildup source: wind vs barotropic density PGF vs both.

Run 3 configs for 100 steps, sample every 10:
  A) wind ON, density PGF ON  (full production)
  B) wind ON, density PGF OFF
  C) wind OFF, density PGF ON
  D) wind OFF, density PGF OFF  (pure residual — should be ~0)

The free-surface step gets F_rho from _compute_bt_rho_pgf. To turn it OFF
without editing the solver, monkeypatch _compute_bt_rho_pgf to return zeros.
Wind off: zero tau in forcing.
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
from jax_solver_global import make_solver_global, _barotropic_velocity, _divergence_conservative, JaxStateG
from woa_data import get_initial_fields
from wind_reanalysis import real_wind_forcing
from run_long_integration_global import heat_flux_meridional, air_temp_profile

gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
ocean = np.asarray(grid.ocean_mask, dtype=bool)
nx, ny, nz = grid.nx, grid.ny, grid.nz
H_sw = float(np.sum(grid.dz))

physics = replace(PhysicsConfig(), nu_h=5e6, nu_bi=0.0, kappa_bi=0.0)
Q_heat = heat_flux_meridional(grid, Q0=50.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
month_idx = (2023 - 1948) * 12
tau_x, tau_y = real_wind_forcing(month_idx=month_idx, grid=grid)
T_sst = T_init[:, :, 0]
T_atm = air_temp_profile(grid, T_sst)

# save original
_orig_bt_rho_pgf = G._compute_bt_rho_pgf

def run(label, wind_on, rho_pgf_on, n=100):
    # monkeypatch density PGF
    if rho_pgf_on:
        G._compute_bt_rho_pgf = _orig_bt_rho_pgf
    else:
        G._compute_bt_rho_pgf = lambda state, p: (jnp.zeros((p.nx, p.ny)), jnp.zeros((p.nx, p.ny)))
    tau = (tau_x, tau_y) if wind_on else (np.zeros_like(tau_x), np.zeros_like(tau_y))
    step, init_state, _, p = make_solver_global(
        grid, physics, 60.0, forcing=(tau[0], tau[1], Q_heat),
        eos_type='linear', T_atm=T_atm, lambda_bulk=40.0,
        sponge_days=3.0, sponge_cells=16, T_init=T_init, S_init=S_init,
        polar_cap_rows=2, polar_cap_taper=3, return_params=True)
    step = jax.jit(step)
    state = init_state(T_init, S_init)
    tx = np.asarray(p.tau_x_2d); ty = np.asarray(p.tau_y_2d)
    print(f"\n=== {label} (wind={wind_on}, rho_pgf={rho_pgf_on}) ===")
    print(f"{'step':>5} {'max|eta|':>10} {'rms|eta|':>9} {'max|ubt|':>9} {'KE_bt':>11} {'P_wind':>11}")
    for k in range(n):
        state = step(state)
        if k % 10 == 9 or k < 5:
            eta = np.asarray(state.eta)
            ubt_j, vbt_j = _barotropic_velocity(state.u, state.v, p)
            ubt = np.asarray(ubt_j); vbt = np.asarray(vbt_j)
            ke_bt = 0.5*np.sum((ubt**2+vbt**2)*ocean*H_sw)
            pwind = np.sum((tx*ubt+ty*vbt)*ocean)
            rms_eta = np.sqrt(np.mean((eta*ocean)**2))
            print(f"{k+1:5d} {np.abs(eta).max():10.3f} {rms_eta:9.4f} {np.abs(ubt).max():9.4f} {ke_bt:11.3e} {pwind:11.3e}", flush=True)

run("A full", True, True)
run("B wind-only", True, False)
run("C rhoPGF-only", False, True)
run("D neither", False, False)
G._compute_bt_rho_pgf = _orig_bt_rho_pgf
