"""Test: does initializing eta to BALANCE the baroclinic PGF prevent the spinup
amplification?  eta_setup = -p_bc_baroclinic / (rho_0 * g)  so that
g*grad(eta) + grad(p_bc)/rho_0 ~= 0  (no net barotropic PGF at t=0).

If the amplification is unbalanced geostrophic adjustment, this should kill it.
If it persists, F_rho is being re-injected each step independent of eta balance.
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from jax_solver_global import (make_solver_global, JaxStateG, RHO_0, G_EARTH,
                               _compute_bt_rho_pgf, _density_anomaly)
from forcing import air_temp_profile, heat_flux_meridional
from woa_data import get_initial_fields

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])
wm = np.array(grid.wet_mask) > 0.5
step, init_state_fn, _ = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0)

state0 = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))

# Use make_solver_global with return_params to get the params object.
step2, init2, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)
rho_prime = _density_anomaly(state0.T, state0.S, params) * params.wet_mask_z
rho_avg = 0.5*(rho_prime[...,:-1]+rho_prime[...,1:])
dp = G_EARTH * rho_avg * params.dz_3d
p_bc = jnp.zeros_like(state0.T)
p_bc = p_bc.at[...,1:].set(jnp.cumsum(dp,axis=-1))
p_bc_avg = jnp.sum(0.5*(p_bc[...,:-1]+p_bc[...,1:])*params.dz_norm, axis=-1)  # (nx,ny)
eta_balance = (-p_bc_avg / (RHO_0 * G_EARTH)) * params.wet_mask
print(f"eta_balance: max|={float(jnp.max(jnp.abs(eta_balance))):.4f} mean={float(jnp.mean(eta_balance)):.4f}")

# Run with balanced eta vs zero eta.
def run(eta_init, label):
    state = JaxStateG(state0.u, state0.v, state0.T, state0.S, eta_init)
    print(f"\n=== {label} ===")
    print(f"{'stp':>5} {'max|eta|':>10} {'max|u|':>10} {'sum_eta':>11}")
    for k in range(1500):
        state = step(state)
        if (k+1)%200==0:
            e=np.array(state.eta); u=np.array(state.u)
            print(f"{k+1:>5} {float(np.max(np.abs(e))):>10.4e} {float(np.max(np.abs(u))):>10.4e} {float(np.sum(e[wm])):>11.4e}")
        if not np.isfinite(np.array(state.eta)).all():
            print(f"  NaN step {k+1}"); break

run(jnp.zeros((grid.nx,grid.ny)), "zero eta (unbalanced)")
run(eta_balance, "balanced eta (eta = -p_bc/rho_0/g)")
