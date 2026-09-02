"""Direct spectral-vs-FD comparison of the free-surface step under F_rho.

The hypothesis under test: the spectral solver's "particular solution" makes it
stable under F_rho while the FD explicit FB blows up. But the diagnostic
_diag_helmholtz_test showed div(F_rho) ~ 9e-6 (F_rho is ~rotational). If div(F)=0,
then the spectral eta_part = -(1-cos)/(g*k^2)*div(F) ~ 0 and the spectral ubt_part
reduces to dt_half*F (pure explicit). So the spectral and FD should be NEARLY
IDENTICAL for this F_rho. If they ARE identical, the root cause is NOT the
particular-solution gap and must lie elsewhere (the matrix-exponential exactness
of the FREE wave + F as pure acceleration, vs FD first-order FB).

This runs the SAME initial state, SAME F_rho, through:
  (A) spectral _free_surface_step (exact matrix exp)
  (B) FD _free_surface_step_fd (first-order FB)
both with r_bot=0, no cap, no sponge, no wind, isolated. Compare energy growth.
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
                               _free_surface_step_fd, _compute_bt_rho_pgf)
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
dz = np.array(grid.dz); dz_norm = dz / dz.sum()

def energy(eta, u, v):
    e = np.array(eta)
    ua = 0.5*(np.array(u)[..., :-1]+np.array(u)[..., 1:])
    va = 0.5*(np.array(v)[..., :-1]+np.array(v)[..., 1:])
    ubt = np.sum(ua*dz_norm, -1); vbt = np.sum(va*dz_norm, -1)
    Ke = 0.5*H*np.sum((ubt**2+vbt**2)*wm)
    Pe = 0.5*G_EARTH*H*np.sum(e**2*wm)
    return Ke+Pe, Ke, Pe

# F_rho from climatology (steady)
eta0 = jnp.zeros((nx, ny))
u0 = jnp.zeros((nx, ny, nz)); v0 = jnp.zeros((nx, ny, nz))
st = JaxStateG(u0, v0, jnp.array(T_init), jnp.array(S_init), eta0)
F_rho_x, F_rho_y = _compute_bt_rho_pgf(st, params)

# --- Build a SPECTRAL free-surface step on the SAME global grid ---
# The spectral solver's _free_surface_step needs spectral params (kx, ky, omega).
# Instead of instantiating the whole spectral solver, replicate the exact matrix
# exp on the global grid via FFT (lon periodic; lat we use a DCT-like sine/cos
# would be needed for closed walls). For a clean comparison we restrict to the
# ZONALLY-UNIFORM component: take the zonal mean of F_rho and eta, run a 1D
# meridional spectral solve vs the FD 1D meridional solve. This isolates the
# time-stepping difference from the 2D mask complications.
# Simpler and more direct: run FD step isolated WITH F_rho (we already know it
# blows up ~step 1200). The question is whether a SPECTRAL-equivalent (exact
# matrix exp) of the SAME forced system also blows up. If the spectral is stable,
# the gap is real; if it also blows up, F_rho itself is the problem (not the
# time-stepping).
#
# Replicate the spectral particular solution on the FD grid is hard (no FFT in
# lat). Instead: test the FREE WAVE (F=0) — both should be neutral/stable. Then
# test WITH a DIVERGENT forcing (not F_rho, which is rotational) — a synthetic
# div-free vs curl-free forcing. This separates "rotational F pumps via free-wave
# resonance" from "divergent F pumps via eta mismatch".

print("=== (A) FD free-surface step, F_rho (rotational), r_bot=0, no cap ===")
eta = jnp.zeros((nx, ny)); u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))
E0,_,_ = energy(eta, u, v)
for k in range(400):
    eta, u, v = _free_surface_step_fd(eta, u, v, params, F_rho_x, F_rho_y)
    if (k+1) % 100 == 0:
        E,Ke,Pe = energy(eta,u,v)
        print(f"  stp {k+1:>4}: E/E0={E/E0:>10.4f} Ke={Ke:.3e} Pe={Pe:.3e} "
              f"max|eta|={float(np.max(np.abs(np.array(eta)))):.3e} "
              f"max|u|={float(np.max(np.abs(np.array(u)))):.3e}")
    if not np.all(np.isfinite(np.array(eta))):
        print(f"  NaN at step {k+1}"); break

print("\n=== (B) FD free-surface step, NO F_rho (free wave), r_bot=0, no cap ===")
eta = jnp.zeros((nx, ny)); u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))
# tiny seiche bump to seed the free mode
yy = jnp.arange(ny); eta = eta.at[:,:].set(0.001*jnp.sin(np.pi*yy/ny))
E0,_,_ = energy(eta, u, v)
for k in range(400):
    eta, u, v = _free_surface_step_fd(eta, u, v, params, None, None)
    if (k+1) % 100 == 0:
        E,Ke,Pe = energy(eta,u,v)
        print(f"  stp {k+1:>4}: E/E0={E/E0:>10.4f} Ke={Ke:.3e} Pe={Pe:.3e} "
              f"max|eta|={float(np.max(np.abs(np.array(eta)))):.3e} "
              f"max|u|={float(np.max(np.abs(np.array(u)))):.3e}")
    if not np.all(np.isfinite(np.array(eta))):
        print(f"  NaN at step {k+1}"); break

# (C) Rotational forcing synthesized: take F_rho, remove its tiny divergent part,
# so it's PURELY rotational. If pure-rotational still pumps, the mechanism is
# free-wave resonance / geostrophic adjustment, NOT eta-balance gap.
print("\n=== (C) FD step, DIVERGENT forcing only (synthetic, same |div| as F_rho) ===")
# synthetic divergent forcing with the same divergence magnitude as F_rho
from jax_solver_global import _divergence_conservative, _gradient_conservative
divF = np.array(_divergence_conservative(F_rho_x*params.wet_mask, F_rho_y*params.wet_mask, params))
# construct a forcing whose divergence = divF: F = grad(phi), phi s.t. lap(phi)=divF
# use a few Jacobi iterations on (I - c*L) to get a phi
phi = jnp.zeros((nx,ny))
def lap(eta):
    gx,gy = _gradient_conservative(eta, params)
    return _divergence_conservative(gx, gy, params)
# phi such that div(grad phi) = divF => grad phi is the divergent part of F
# iterate: phi <- phi + alpha*(divF - lap(phi))
alpha = 0.5
for _ in range(2000):
    phi = phi + alpha * (jnp.array(divF) - lap(phi)) * params.wet_mask
gx, gy = _gradient_conservative(phi, params)
Fdiv_x = np.array(gx); Fdiv_y = np.array(gy)
print(f"  reconstructed divergent F: rms|F_div|={np.sqrt(np.mean(Fdiv_x**2+Fdiv_y**2)):.4e} "
      f"(vs |F_rho|={np.sqrt(np.mean(np.array(F_rho_x)**2+np.array(F_rho_y)**2)):.4e})")
eta = jnp.zeros((nx, ny)); u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))
E0,_,_ = energy(eta, u, v)
for k in range(400):
    eta, u, v = _free_surface_step_fd(eta, u, v, params, jnp.array(Fdiv_x), jnp.array(Fdiv_y))
    if (k+1) % 100 == 0:
        E,Ke,Pe = energy(eta,u,v)
        print(f"  stp {k+1:>4}: E/E0={E/E0:>10.4f} Ke={Ke:.3e} Pe={Pe:.3e} "
              f"max|eta|={float(np.max(np.abs(np.array(eta)))):.3e} "
              f"max|u|={float(np.max(np.abs(np.array(u)))):.3e}")
    if not np.all(np.isfinite(np.array(eta))):
        print(f"  NaN at step {k+1}"); break
print("\nDONE.")
