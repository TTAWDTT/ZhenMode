"""G3 instability diagnostic: isolate which residual term drives the growth.

Setup: no-wind, uniform T, a small barotropic eta bump, r_bot=0.
With r_bot=0 and uniform T the residual *should* be ~0 (advection negligible,
PGF cancels bt_pgf, Coriolis cancels, no wind, no bot friction). If growth
persists, the residual is NOT ~0 and we find which term survives.
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
import sys; sys.path.insert(0, 'src')

from config import PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
import jax_solver_global as G
from jax_solver_global import (make_solver_global, JaxStateG,
                               _compute_momentum_residual, _compute_momentum_tendency,
                               _laplacian_h, _compute_pressure_gradient,
                               _compute_bt_rho_pgf, _compute_vertical_velocity,
                               _advection_flux_form, _d_dx, _d_dy, G_EARTH)

BATHY = r"C:\Users\zhen.luo\Desktop\ETOPO_2022_v1_r3600x1800_surface.nc"
gc = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gc, BATHY, smooth_passes=30, min_depth=100.0)

phys = replace(PhysicsConfig(),
               nu_h=5e6, nu_bi=0.0, kappa_bi=0.0,
               r_bot=0.0, bottom_friction='none')
step, init_state, diag, params = make_solver_global(
    grid, phys, dt=60.0, forcing=None, polar_cap_rows=0, return_params=True)

nx, ny, nz = params.nx, params.ny, params.nz
wm = params.wet_mask
wmz = params.wet_mask_z

T0 = 15.0
T = jnp.ones((nx, ny, nz)) * T0 * wmz + (1 - wmz) * T0
S = jnp.ones((nx, ny, nz)) * 35.0 * wmz + (1 - wmz) * 35.0
u = jnp.zeros((nx, ny, nz))
v = jnp.zeros((nx, ny, nz))

lon_c, lat_c = nx // 2, ny // 2
ii = np.arange(nx)[:, None]
jj = np.arange(ny)[None, :]
bump = np.exp(-((ii - lon_c)**2 + (jj - lat_c)**2) / (2 * (nx/8)**2))
eta = jnp.array(bump * np.array(wm) * 1.0)

state = JaxStateG(u, v, T, S, eta)

def ke(s):
    return float(jnp.sum(s.u**2 + s.v**2))

def max_eta(s):
    return float(jnp.max(jnp.abs(s.eta)))

print(f"init: KE={ke(state):.6e} max|eta|={max_eta(state):.4f}")

# Residual + breakdown on initial state
du, dv = _compute_momentum_residual(state, params)
print(f"residual: max|du|={float(jnp.max(jnp.abs(du))):.6e} "
      f"max|dv|={float(jnp.max(jnp.abs(dv))):.6e}")

w = _compute_vertical_velocity(state, params)
adv_u, adv_v = _advection_flux_form(state.u, state.v, w, params)
f3 = params.f[:, :, None]
pgf_x, pgf_y = _compute_pressure_gradient(state, params)
bt_pgf_x = -G_EARTH * _d_dx(state.eta[:, :, None], params)[:, :, 0]
print(f"  adv_u  max={float(jnp.max(jnp.abs(adv_u))):.6e}")
print(f"  cor_u  max={float(jnp.max(jnp.abs(f3*state.v))):.6e}")
print(f"  pgf_x  max={float(jnp.max(jnp.abs(pgf_x))):.6e}")
print(f"  bt_pgf max={float(jnp.max(jnp.abs(bt_pgf_x))):.6e}")
print(f"  pgf-bt_pgf (should ~0 barotropic uniform T) = "
      f"{float(jnp.max(jnp.abs(pgf_x[:,:,0] - bt_pgf_x))):.6e}")

# Step forward
print("\nstepping (dt=60, r_bot=0, no wind)...")
for i in range(120):
    state = step(state)
    if (i+1) % 20 == 0:
        print(f"  step {i+1:3d}: KE={ke(state):.6e} max|eta|={max_eta(state):.4f}")
