"""Test what arrests the 13m eta hotspot: sponge/polcap vs nu_h vs dt.

Config D (no wind, no F_rho) baseline builds eta to 13.8m by step 100.
Sweep levers to find what arrests it:
  1) sponge OFF, polcap OFF
  2) sponge OFF, polcap ON
  3) sponge ON, polcap OFF
  4) nu_h=5e6 -> 5e7 (10x stronger)
  5) dt 60 -> 30 (halve)
  6) r_bot 1e-3 -> 1e-2 (10x stronger bottom drag)
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
from run_long_integration_global import heat_flux_meridional, air_temp_profile

gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
ocean = np.asarray(grid.ocean_mask, dtype=bool)
H_sw = float(np.sum(grid.dz))
Q_heat = heat_flux_meridional(grid, Q0=50.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
T_atm = air_temp_profile(grid, T_init[:, :, 0])

def run(label, nu_h=5e6, dt=60.0, sponge_days=3.0, sponge_cells=16,
        polcap_rows=2, polcap_taper=3, r_bot=1e-3, n=100):
    physics = replace(PhysicsConfig(), nu_h=nu_h, nu_bi=0.0, kappa_bi=0.0, r_bot=r_bot)
    G._compute_bt_rho_pgf = lambda state, p: (jnp.zeros((p.nx, p.ny)), jnp.zeros((p.nx, p.ny)))
    step, init_state, _, p = make_solver_global(
        grid, physics, dt, forcing=(np.zeros((grid.nx,grid.ny)), np.zeros((grid.nx,grid.ny)), Q_heat),
        eos_type='linear', T_atm=T_atm, lambda_bulk=40.0,
        sponge_days=sponge_days, sponge_cells=sponge_cells,
        T_init=T_init, S_init=S_init,
        polar_cap_rows=polcap_rows, polar_cap_taper=polcap_taper, return_params=True)
    step = jax.jit(step)
    state = init_state(T_init, S_init)
    vals = []
    for k in range(n):
        state = step(state)
        if k % 20 == 19:
            eta = np.asarray(state.eta)
            ubt = np.asarray(_barotropic_velocity(state.u, state.v, p)[0])
            vals.append((k+1, np.abs(eta).max(), np.sqrt(np.mean((eta*ocean)**2)), np.abs(ubt).max()))
    print(f"{label:35s} | " + " | ".join(f"s{v[0]}:{v[1]:.1f}m(rms{v[2]:.2f},u{v[3]:.3f})" for v in vals), flush=True)

print("Config D baseline (no wind, no F_rho) — eta at steps 20/40/60/80/100:")
print(f"{'lever':35s} | trajectory")
print("-"*120)
run("baseline (sponge3d/16,pcap2+3,nu5e6,dt60)", n=100)
run("sponge OFF", sponge_days=0.0, sponge_cells=0)
run("polcap OFF", polcap_rows=0, polcap_taper=0)
run("sponge+polcap OFF", sponge_days=0.0, sponge_cells=0, polcap_rows=0, polcap_taper=0)
run("nu_h 5e7 (10x)", nu_h=5e7)
run("dt 30 (half)", dt=30.0)
run("r_bot 1e-2 (10x)", r_bot=1e-2)
