"""Early-time evolution of isolated FS step + constant F_rho. The eta reaches
6.7m quickly while u saturates at 0.078. Where does the 6.7m eta come from if
u is tiny? Check: is it the geostrophic setup (eta balancing F_rho via PGF)?
F_rho ~ -g*grad(eta_geo), so the steady state is eta = eta_geo with u~0. The
6.7m may BE the physical geostrophic setup, and the slow drift after is the
residual imbalance. Compare WITH vs WITHOUT F_rho over 4000 steps, early+late.
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
from jax_solver_global import (make_solver_global, JaxStateG, RHO_0, G_EARTH,
                               _free_surface_step_fd, _compute_bt_rho_pgf,
                               _gradient_conservative)
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
g = float(G_EARTH)

_, _, _, params = make_solver_global(
    grid, replace(PhysicsConfig(), nu_h=0.0, nu_bi=0, kappa_bi=0, kappa_h=0.0, r_bot=1e-3),
    60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)
nx,ny,nz = grid.nx, grid.ny, grid.nz

wm_j = jnp.array(grid.wet_mask); dz_norm_j = jnp.array(dz_norm)
@jax.jit
def energy_j(eta,u,v):
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1); vbt=jnp.sum(va*dz_norm_j,-1)
    return (0.5*H*jnp.sum((ubt**2+vbt**2)*wm_j)+0.5*g*H*jnp.sum(eta**2*wm_j),
            0.5*g*H*jnp.sum(eta**2*wm_j), 0.5*H*jnp.sum((ubt**2+vbt**2)*wm_j))

eta0 = jnp.zeros((nx,ny)); u0=jnp.zeros((nx,ny,nz)); v0=jnp.zeros((nx,ny,nz))
st = JaxStateG(u0,v0,jnp.array(T_init),jnp.array(S_init),eta0)
Fx, Fy = _compute_bt_rho_pgf(st, params)

# The geostrophic eta that balances F_rho: g*grad(eta_geo) = F_rho (so PGF cancels F).
# Solve lap(eta_geo) = div(F)/g, but div(F)~0. Instead the rotational F drives
# geostrophic VELOCITY (u_g = -F_y/f), not eta. So eta does NOT balance F directly.
# Report the F_rho magnitude and what eta it implies.
print(f"max|F_rho|={float(jnp.max(jnp.abs(Fx))):.4e}")
print(f"(F_rho is rotational: drives geostrophic flow u_g=-F_y/f, not eta setup)")

step_fs = jax.jit(lambda eta,u,v: _free_surface_step_fd(eta,u,v,params,Fx,Fy))
step_nof = jax.jit(lambda eta,u,v: _free_surface_step_fd(eta,u,v,params,None,None))

def run(label, step_fn, n=4000):
    eta = jnp.zeros((nx,ny)); u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))
    print(f"\n=== {label} ===")
    print(f"{'stp':>5} {'E':>12} {'PE':>12} {'KE':>12} {'max|eta|':>10} {'max|u|':>10}")
    for k in range(n):
        eta, u, v = step_fn(eta, u, v)
        if k in (0,1,2,4,9,19) or (k+1)%500==0:
            E,Pe,Ke=energy_j(eta,u,v)
            print(f"{k+1:>5} {float(E):>12.4e} {float(Pe):>12.4e} {float(Ke):>12.4e} "
                  f"{float(jnp.max(jnp.abs(eta))):>10.4e} {float(jnp.max(jnp.abs(u))):>10.4e}")
            if not bool(jnp.isfinite(eta).all()): print(f"  NaN step {k+1}"); break

run("Isolated FS, CONSTANT F_rho, r_bot=1e-3", step_fs)
run("Isolated FS, NO F_rho (free wave from rest), r_bot=1e-3", step_nof)
print("\nDONE.")
