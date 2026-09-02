"""Isolate the free-surface step: does _free_surface_step_fd ALONE conserve energy?

Calls _free_surface_step_fd directly (no linear half-step, no Coriolis, no
advection, no diffusion) on the real masked global grid. If energy grows here,
the free-surface step itself injects. If energy is flat here, the injection is
in the linear half-step / Coriolis / 3D coupling.
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from jax_solver_global import make_solver_global, JaxStateG, RHO_0, G_EARTH, _free_surface_step_fd
from forcing import air_temp_profile, heat_flux_meridional
from woa_data import get_initial_fields

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
physics = replace(PhysicsConfig(), nu_h=0.0, nu_bi=0, kappa_bi=0, kappa_h=0.0, r_bot=0.0)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

wm = np.array(grid.wet_mask) > 0.5
nx, ny, nz = grid.nx, grid.ny, grid.nz
H = 4000.0
dz = np.array(grid.dz); dz_norm = dz/dz.sum()

def energy(eta, u, v):
    e = np.array(eta)
    ua = 0.5*(np.array(u)[...,:-1]+np.array(u)[...,1:])
    va = 0.5*(np.array(v)[...,:-1]+np.array(v)[...,1:])
    ubt = np.sum(ua*dz_norm,-1); vbt = np.sum(va*dz_norm,-1)
    Ke = 0.5*H*np.sum((ubt**2+vbt**2)*wm)
    Pe = 0.5*G_EARTH*H*np.sum(e**2*wm)
    return Ke+Pe, Ke, Pe

# seiche bump, zero 3D velocity
eta = jnp.zeros((nx, ny))
yy = jnp.arange(ny); eta = eta.at[:, :].set(0.1*jnp.sin(np.pi*yy/ny))
u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))

# Build a state to compute the density PGF (the full step passes F_rho).
from jax_solver_global import _compute_bt_rho_pgf
state_for_rho = JaxStateG(u, v, jnp.array(T_init), jnp.array(S_init), eta)
F_rho_x, F_rho_y = _compute_bt_rho_pgf(state_for_rho, params)
print(f"max|F_rho_x|={float(jnp.max(jnp.abs(F_rho_x))):.4e} max|F_rho_y|={float(jnp.max(jnp.abs(F_rho_y))):.4e}")

E0,_,_ = energy(eta,u,v)
print(f"\nisolated _free_surface_step_fd WITH density PGF F_rho (no Coriolis/diffusion, r_bot=0, cap off)")
print(f"{'stp':>4} {'E/E0':>10} {'Ke':>11} {'Pe':>11} {'max|eta|':>10} {'max|ubt|':>10}")
for k in range(400):
    eta, u, v = _free_surface_step_fd(eta, u, v, params, F_rho_x, F_rho_y)
    if (k+1)%40==0:
        E,Ke,Pe = energy(eta,u,v)
        ue = np.array(u)
        print(f"{k+1:>4} {E/E0:>10.5f} {Ke:>11.4e} {Pe:>11.4e} "
              f"{float(np.max(np.abs(np.array(eta)))):>10.4e} {float(np.max(np.abs(ue))):>10.4e}")
    if not np.all(np.isfinite(np.array(eta))): print("NaN"); break
