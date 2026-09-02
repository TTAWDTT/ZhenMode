"""Test the CORRECT semi-implicit SW: eta uses NEW velocity (no lag).

The explicit FB lags: eta_new = eta - dt*H*div(u_OLD), u_new = u + dt*(-g*grad(eta_new)+F).
The semi-implicit removes the lag by making eta depend on the NEW velocity:
  eta_new = eta - dt*H*div(u_NEW)
  u_new   = u + dt*(-g*grad(eta_new) + F)
Eliminating u_new:
  (I - dt^2*g*H*L) eta_new = eta - dt*H*div(u) - dt^2*H*div(F)
then u_new = u + dt*(-g*grad(eta_new)+F).

The KEY difference from the earlier (wrong) test: there I only checked whether
eta balances F (it can't, div(F)~0). Here eta instead balances the VELOCITY-
driven divergence with no lag. The F enters eta via -dt^2*H*div(F) (~0, fine)
AND via u_new's dt*F which then drives the NEXT step's div(u). The question is
whether removing the eta-u lag kills the unbounded growth.

Isolated test (no Coriolis/diffusion, r_bot=0, cap off) with full F_rho.
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax
import jax.numpy as jnp
import numpy as np
import jax.scipy.sparse.linalg as jsp
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from jax_solver_global import (make_solver_global, JaxStateG, RHO_0, G_EARTH,
                               _compute_bt_rho_pgf, _divergence_conservative,
                               _gradient_conservative, _barotropic_velocity)
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
_, _, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

wm = np.array(grid.wet_mask) > 0.5
nx, ny, nz = grid.nx, grid.ny, grid.nz
H = 4000.0
dt = 60.0; dt_half = dt/2.0
g = float(G_EARTH)
dz = np.array(grid.dz); dz_norm = dz/dz.sum()

def energy(eta, u, v):
    e = np.array(eta)
    ua = 0.5*(np.array(u)[...,:-1]+np.array(u)[...,1:])
    va = 0.5*(np.array(v)[...,:-1]+np.array(v)[...,1:])
    ubt = np.sum(ua*dz_norm,-1); vbt = np.sum(va*dz_norm,-1)
    Ke = 0.5*H*np.sum((ubt**2+vbt**2)*wm); Pe = 0.5*g*H*np.sum(e**2*wm)
    return Ke+Pe, Ke, Pe

def lap(eta):
    gx, gy = _gradient_conservative(eta, params)
    return _divergence_conservative(gx, gy, params)

# Helmholtz operator
def helmholtz_op(eta):
    return params.wet_mask * (eta - (dt_half**2) * g * H * lap(eta))

@jax.jit
def solve_eta(rhs):
    eta_sol, info = jsp.cg(helmholtz_op, rhs, maxiter=200, tol=1e-10)
    return eta_sol

def semi_implicit_step(eta, u, v, Fx, Fy):
    """Semi-implicit SW: eta uses NEW velocity (no lag)."""
    ubt, vbt = _barotropic_velocity(u, v, params)
    ubt = ubt * params.wet_mask; vbt = vbt * params.wet_mask
    F_x = Fx; F_y = Fy  # no wind, no drag here
    # rhs = eta - dt_half*H*div(u) - dt_half^2*H*div(F)
    div_u = _divergence_conservative(ubt, vbt, params)
    div_F = _divergence_conservative(F_x*params.wet_mask, F_y*params.wet_mask, params)
    rhs = (eta - dt_half*H*div_u - (dt_half**2)*H*div_F) * params.wet_mask
    eta_new = solve_eta(rhs)
    eta_new = eta_new * params.wet_mask
    # u_new = u + dt*(-g*grad(eta_new) + F)
    gx, gy = _gradient_conservative(eta_new, params)
    ubt_new = ubt + dt_half*(-g*gx + F_x)
    vbt_new = vbt + dt_half*(-g*gy + F_y)
    ubt_new = ubt_new * params.wet_mask; vbt_new = vbt_new * params.wet_mask
    # project back to 3D
    delta_ubt = (ubt_new - ubt)[:, :, None]
    delta_vbt = (vbt_new - vbt)[:, :, None]
    u_new = u + delta_ubt; v_new = v + delta_vbt
    return eta_new, u_new, v_new

# F_rho
eta0 = jnp.zeros((nx, ny)); u0 = jnp.zeros((nx,ny,nz)); v0 = jnp.zeros((nx,ny,nz))
st = JaxStateG(u0, v0, jnp.array(T_init), jnp.array(S_init), eta0)
F_rho_x, F_rho_y = _compute_bt_rho_pgf(st, params)

print("=== Semi-implicit SW (eta uses NEW velocity), full F_rho, r_bot=0, no cap ===")
eta = jnp.zeros((nx, ny)); u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))
E0,_,_ = energy(eta, u, v)
for k in range(800):
    eta, u, v = semi_implicit_step(eta, u, v, F_rho_x, F_rho_y)
    if (k+1) % 100 == 0:
        E,Ke,Pe = energy(eta,u,v)
        print(f"  stp {k+1:>4}: E={E:.4e} Ke={Ke:.3e} Pe={Pe:.3e} "
              f"max|eta|={float(np.max(np.abs(np.array(eta)))):.3e} "
              f"max|u|={float(np.max(np.abs(np.array(u)))):.3e} "
              f"sum_eta={float(np.sum(np.array(eta)[wm])):.3e}")
    if not np.all(np.isfinite(np.array(eta))):
        print(f"  NaN at step {k+1}"); break

print("\n=== Reference: explicit FB (current), full F_rho ===")
from jax_solver_global import _free_surface_step_fd
eta = jnp.zeros((nx, ny)); u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))
for k in range(400):
    eta, u, v = _free_surface_step_fd(eta, u, v, params, F_rho_x, F_rho_y)
    if (k+1) % 100 == 0:
        E,Ke,Pe = energy(eta,u,v)
        print(f"  stp {k+1:>4}: E={E:.4e} Ke={Ke:.3e} Pe={Pe:.3e} "
              f"max|eta|={float(np.max(np.abs(np.array(eta)))):.3e} "
              f"max|u|={float(np.max(np.abs(np.array(u)))):.3e}")
    if not np.all(np.isfinite(np.array(eta))):
        print(f"  NaN at step {k+1}"); break
print("\nDONE.")
