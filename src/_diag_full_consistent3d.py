"""Test full step with ALL horizontal pressure gradients using the conservative
gradient (adjoint-consistent with the free-surface divergence). This patches
both _compute_bt_rho_pgf AND _compute_pressure_gradient (the full 3D PGF) to
use _gradient_conservative. If the blowup was discrete-gradient/divergence
inconsistency, this fixes it; if it still blows, the issue is the explicit
time stepping of the 3D baroclinic PGF (architectural).
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
import jax_solver_global as G
from jax_solver_global import (make_solver_global, JaxStateG, RHO_0, G_EARTH,
                               _step_impl, _gradient_conservative,
                               _density_anomaly, _compute_hydrostatic_pressure,
                               _d_dx, _d_dy)
from forcing import air_temp_profile, heat_flux_meridional
from woa_data import get_initial_fields

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])
wm = np.array(grid.wet_mask) > 0.5
dz = np.array(grid.dz); dz_norm = dz/dz.sum(); H=4000.0

physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

wm_j = jnp.array(grid.wet_mask); dz_norm_j = jnp.array(dz_norm)
@jax.jit
def energy_j(state):
    e=state.eta; u=state.u; v=state.v
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1); vbt=jnp.sum(va*dz_norm_j,-1)
    return (0.5*H*jnp.sum((ubt**2+vbt**2)*wm_j)+0.5*G_EARTH*H*jnp.sum(e**2*wm_j))
@jax.jit
def diag_j(state):
    return (jnp.max(jnp.abs(state.eta)), jnp.max(jnp.abs(state.u)),
            jnp.sum(state.eta*wm_j), jnp.isfinite(state.eta).all())

# Conservative gradient for a 3D field: apply per-level
def _grad3d_consistent(field3d, p):
    # field3d (nx,ny,nz): apply conservative gradient on each level
    gx = jnp.zeros_like(field3d); gy = jnp.zeros_like(field3d)
    for k in range(field3d.shape[-1]):
        gxa, gya = _gradient_conservative(field3d[..., k], p)
        gx = gx.at[..., k].set(gxa); gy = gy.at[..., k].set(gya)
    return gx, gy

# Patch _compute_pressure_gradient to use conservative gradient
def _pressure_gradient_consistent(state, p):
    pressure = _compute_hydrostatic_pressure(state, p)   # (nx,ny,nz)
    pgf_x, pgf_y = _grad3d_consistent(pressure, p)
    return -pgf_x / RHO_0, -pgf_y / RHO_0
G._compute_pressure_gradient = _pressure_gradient_consistent

# Patch _compute_bt_rho_pgf too (barotropic density PGF)
def _bt_rho_consistent(state, p):
    rho_prime = _density_anomaly(state.T, state.S, p) * p.wet_mask_z
    rho_avg = 0.5 * (rho_prime[..., :-1] + rho_prime[..., 1:])
    dp = G_EARTH * rho_avg * p.dz_3d
    p_bc = jnp.zeros_like(state.T).at[..., 1:].set(jnp.cumsum(dp, axis=-1))
    p_bc_avg = jnp.sum(0.5 * (p_bc[..., :-1] + p_bc[..., 1:]) * p.dz_norm, axis=-1)
    gx, gy = _gradient_conservative(p_bc_avg, p)
    return -gx / RHO_0, -gy / RHO_0
G._compute_bt_rho_pgf = _bt_rho_consistent

step_cons = jax.jit(lambda s: _step_impl(s, params))

def run(label, step_fn, n=1400):
    state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    print(f"\n=== {label} ===")
    print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10} {'sum_eta':>11}")
    for k in range(n):
        state = step_fn(state)
        if (k+1)%200==0:
            E=float(energy_j(state)); me, mu, se, fin = diag_j(state)
            print(f"{k+1:>5} {E:>12.4e} {float(me):>10.4e} {float(mu):>10.4e} {float(se):>11.4e}")
            if not bool(fin): print(f"  NaN step {k+1}"); break

run("FULL step, ALL PGF conservative-gradient, real Coriolis+nu_h+r_bot", step_cons)
print("\nDONE.")
