"""Pinpoint the energy injection path of a ROTATIONAL (geostrophic) forcing.

(A) confirmed: F_rho (rotational, div~0) pumps energy unboundedly under FD FB,
while the divergent part is harmless. The semi-implicit Helmholtz (which only
couples div(F)) therefore CANNOT fix it. This script isolates the mechanism:

A steady rotational force F (div F = 0) on the SW system:
  d(eta)/dt = -H*div(u)
  du/dt     = -g*grad(eta) + F
If F is exactly geostrophic (F = -g*grad(eta_geo) for some steady eta_geo), then
the STEADY state is u_g = 0, eta = eta_geo (F balanced by eta). But under explicit
FB, eta lags: eta_new = eta - dt*H*div(u), u_new = u + dt*(-g*grad(eta_new)+F).
The question: does the explicit FB of a PURELY rotational F blow up, and does
making eta IMPLICIT in F (i.e. eta_new balances F within-step) stop it?

Test: F = pure rotational (take F_rho, project out divergent part exactly via a
Poisson solve so div(F)=0 to machine precision). Then compare:
  (1) explicit FB  (current)
  (2) semi-implicit where eta_new solves (I - dt^2 g H L) eta_new = eta - dt*H*div(u) + dt^2*H*F_grad
      i.e. the FULL F enters the eta equation as a pressure-equivalent source,
      not just div(F). This is the "treat F as -g*grad(eta_forced)" coupling.
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
                               _free_surface_step_fd, _compute_bt_rho_pgf,
                               _divergence_conservative, _gradient_conservative)
import jax.scipy.sparse.linalg as jsp
import jax
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

# --- F_rho from climatology ---
eta0 = jnp.zeros((nx, ny)); u0 = jnp.zeros((nx,ny,nz)); v0 = jnp.zeros((nx,ny,nz))
st = JaxStateG(u0, v0, jnp.array(T_init), jnp.array(S_init), eta0)
F_rho_x, F_rho_y = _compute_bt_rho_pgf(st, params)

# --- Project F_rho to PURELY rotational (div=0) via Helmholtz Poisson ---
# We want F_rot = F - grad(phi) where div(grad(phi)) = div(F). Then div(F_rot)=0.
def lap(eta):
    gx, gy = _gradient_conservative(eta, params)
    return _divergence_conservative(gx, gy, params)
divF = _divergence_conservative(F_rho_x*params.wet_mask, F_rho_y*params.wet_mask, params)
# Solve lap(phi) = divF
@jax.jit
def solve_phi(divF):
    op = lambda phi: params.wet_mask * lap(phi)
    phi, info = jsp.cg(op, divF, maxiter=300, tol=1e-12)
    return phi
phi = solve_phi(divF)
gx, gy = _gradient_conservative(phi, params)
F_rot_x = (F_rho_x - gx) * params.wet_mask
F_rot_y = (F_rho_y - gy) * params.wet_mask
# verify div(F_rot) ~ 0
div_rot = np.array(_divergence_conservative(F_rot_x, F_rot_y, params))
print(f"purely-rotational F_rho: rms|F_rot|={np.sqrt(np.mean(np.array(F_rot_x)**2+np.array(F_rot_y)**2)):.4e} "
      f"rms|div(F_rot)|={np.sqrt(np.mean(div_rot**2)):.4e}")

def run(Fx, Fy, label, n=400):
    eta = jnp.zeros((nx, ny)); u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))
    E0,_,_ = energy(eta,u,v)
    print(f"\n=== {label} ===")
    for k in range(n):
        eta, u, v = _free_surface_step_fd(eta, u, v, params, Fx, Fy)
        if (k+1) % 100 == 0:
            E,Ke,Pe = energy(eta,u,v)
            print(f"  stp {k+1:>4}: E={E:.4e} Ke={Ke:.3e} Pe={Pe:.3e} "
                  f"max|eta|={float(np.max(np.abs(np.array(eta)))):.3e} "
                  f"max|u|={float(np.max(np.abs(np.array(u)))):.3e}")
        if not np.all(np.isfinite(np.array(eta))):
            print(f"  NaN at step {k+1}"); break

# (1) pure rotational F_rho under explicit FB
run(F_rot_x, F_rot_y, "(1) Explicit FB, PURELY ROTATIONAL F_rho (div=0)")

# (2) full F_rho under explicit FB (for reference)
run(F_rho_x, F_rho_y, "(2) Explicit FB, FULL F_rho (reference)")
print("\nDONE.")
