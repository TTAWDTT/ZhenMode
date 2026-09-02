"""G3: Isolate L/2 (linear) vs N (nonlinear) step contribution to growth.

Bump test, r_bot=0, uniform T. Run three variants for 120 steps:
  (a) full step (L/2 -> N -> L/2)
  (b) L/2 only (no N-step)  -> does the linear half-step alone grow?
  (c) N only (no L/2)
Also test pure free-surface step in isolation (no diffusion, no Coriolis).
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
                               _linear_half_step, _explicit_full_step,
                               _free_surface_step_fd, _barotropic_velocity,
                               _d_dx, _d_dy, G_EARTH, _step_impl)

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

def make_bump_state():
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
    return JaxStateG(u, v, T, S, eta)

def ke(s):
    return float(jnp.sum(s.u**2 + s.v**2))

def max_eta(s):
    return float(jnp.max(jnp.abs(s.eta)))

# (a) full step
sa = make_bump_state()
# (b) L/2 only (double it to match one full step's worth of linear time)
@jax.jit
def lin_only(s):
    s = _linear_half_step(s, params, params.dt/2.0)
    s = _linear_half_step(s, params, params.dt/2.0)
    return s
# (c) N only
@jax.jit
def n_only(s):
    return _explicit_full_step(s, params, params.dt)
# (d) pure free surface: just the FB step, no diffusion, no Coriolis, repeated
#     (mimics one full step = 2 half free-surface steps)
@jax.jit
def fs_only(s):
    eta, u, v = _free_surface_step_fd(s.eta, s.u, s.v, params, None, None, params.dt/2.0)
    eta, u, v = _free_surface_step_fd(eta, u, v, params, None, None, params.dt/2.0)
    return JaxStateG(u, v, s.T, s.S, eta)

print(f"{'step':>4} | {'(a) full KE':>13} {'eta':>7} | {'(b) L/2 KE':>13} {'eta':>7} | "
      f"{'(c) N-only KE':>14} {'eta':>7} | {'(d) FS-only KE':>15} {'eta':>7}")
sb = make_bump_state(); sc = make_bump_state(); sd = make_bump_state()
print(f"{'init':>4} | {ke(sa):13.6e} {max_eta(sa):7.4f} | "
      f"{ke(sb):13.6e} {max_eta(sb):7.4f} | "
      f"{ke(sc):14.6e} {max_eta(sc):7.4f} | "
      f"{ke(sd):15.6e} {max_eta(sd):7.4f}")

for i in range(120):
    sa = step(sa)
    sb = lin_only(sb)
    sc = n_only(sc)
    sd = fs_only(sd)
    if (i+1) % 30 == 0:
        print(f"{i+1:>4} | {ke(sa):13.6e} {max_eta(sa):7.4f} | "
              f"{ke(sb):13.6e} {max_eta(sb):7.4f} | "
              f"{ke(sc):14.6e} {max_eta(sc):7.4f} | "
              f"{ke(sd):15.6e} {max_eta(sd):7.4f}")
