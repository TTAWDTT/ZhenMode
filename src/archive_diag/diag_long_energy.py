"""G3: Run the bump test LONG (2000 steps = ~33h) to see if energy grows
without bound (instability) or oscillates/saturates (physical adjustment).

Also run WITH wind off and uniform T but NO bump (rest state) — should stay
at zero. And run the realistic config (wind + bulk flux) briefly.
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
import sys; sys.path.insert(0, 'src')

from config import PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from jax_solver_global import (make_solver_global, JaxStateG,
                               _barotropic_velocity, G_EARTH, RHO_0)

BATHY = r"C:\Users\zhen.luo\Desktop\ETOPO_2022_v1_r3600x1800_surface.nc"
gc = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gc, BATHY, smooth_passes=30, min_depth=100.0)

phys = replace(PhysicsConfig(),
               nu_h=5e6, nu_bi=0.0, kappa_bi=0.0,
               r_bot=0.0, bottom_friction='none')
step, init_state, diag, params = make_solver_global(
    grid, phys, dt=60.0, forcing=None, polar_cap_rows=0, return_params=True)

nx, ny, nz = params.nx, params.ny, params.nz
wm = np.array(params.wet_mask)
wmz = params.wet_mask_z
H_sw = float(params.H_sw)
dA = np.array(params.dx_2d) * float(params.dy)

def ke(s):
    ubt, vbt = _barotropic_velocity(s.u, s.v, params)
    return float(0.5*RHO_0*H_sw*jnp.sum((ubt**2+vbt**2)*jnp.array(dA)))
def pe(s):
    return float(0.5*RHO_0*G_EARTH*jnp.sum(s.eta**2*jnp.array(dA)))
def maxu(s):
    return float(jnp.max(jnp.abs(s.u)))

T = jnp.ones((nx,ny,nz))*15.0*wmz + (1-wmz)*15.0
S = jnp.ones((nx,ny,nz))*35.0*wmz + (1-wmz)*35.0
u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))
lc, mc = nx//2, ny//2
ii = np.arange(nx)[:,None]; jj = np.arange(ny)[None,:]
bump = np.exp(-((ii-lc)**2+(jj-mc)**2)/(2*(nx/8)**2))
eta = jnp.array(bump*wm*1.0)
s = JaxStateG(u,v,T,S,eta)

print(f"init: KE={ke(s):.4e} PE={pe(s):.4e} E={ke(s)+pe(s):.4e} max|u|={maxu(s):.4e} max|eta|={float(jnp.max(jnp.abs(s.eta))):.4f}")
print(f"{'step':>5} {'KE':>12} {'PE':>12} {'E_tot':>12} {'max|u|':>10} {'max|eta|':>9}")
for i in range(2400):
    s = step(s)
    if i%200==0 or i==2399:
        print(f"{i+1:5d} {ke(s):12.4e} {pe(s):12.4e} {ke(s)+pe(s):12.4e} "
              f"{maxu(s):10.4e} {float(jnp.max(jnp.abs(s.eta))):9.4f}")
