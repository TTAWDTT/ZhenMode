"""G3: Why does the pure FB free-surface step grow energy?

Test variables:
  1. H_sw value and effective CFL at the bump latitude.
  2. Does growth depend on dt? (CFL vs splitting artifact)
  3. Does growth happen on a FLAT-bottom f-plane (no bathymetry variation,
     no Coriolis gradient) — isolates whether variable H / coast injects.
  4. Is energy injected at boundaries (no-flux wall) or interior?
Monitor total energy (KE + PE = 0.5*rho*int(u^2+v^2) + 0.5*rho*g*eta^2*H... )
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
                               _free_surface_step_fd, _barotropic_velocity,
                               _d_dx, _d_dy, G_EARTH, RHO_0)

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
print(f"H_sw = {H_sw:.1f} m,  nz={nz}")
print(f"dx_eq = {float(jnp.min(params.dx_2d)):.0f} m,  dx@60 = {float(jnp.min(params.dx_2d)/0.5):.0f} m")
c = np.sqrt(G_EARTH * H_sw)
print(f"c = sqrt(g*H_sw) = {c:.1f} m/s,  CFL_eq(dt=60) = {c*60/float(jnp.min(params.dx_2d)):.4f}")

def pe(s):
    # PE = 0.5 * rho * g * integral(eta^2) dA  (linear SW)
    dA = np.array(params.dx_2d) * float(params.dy)
    return float(0.5 * RHO_0 * G_EARTH * jnp.sum(s.eta**2 * jnp.array(dA)))

def ke(s):
    dA = np.array(params.dx_2d) * float(params.dy)
    # depth-integrated KE of barotropic mode
    ubt, vbt = _barotropic_velocity(s.u, s.v, params)
    return float(0.5 * RHO_0 * H_sw * jnp.sum((ubt**2 + vbt**2) * jnp.array(dA)))

def make_bump(scale_deg=8.0, amp=1.0):
    T = jnp.ones((nx,ny,nz))*15.0*wmz + (1-wmz)*15.0
    S = jnp.ones((nx,ny,nz))*35.0*wmz + (1-wmz)*35.0
    u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))
    lc, mc = nx//2, ny//2
    ii = np.arange(nx)[:,None]; jj = np.arange(ny)[None,:]
    bump = np.exp(-((ii-lc)**2+(jj-mc)**2)/(2*(nx/scale_deg)**2))
    eta = jnp.array(bump*wm*amp)
    return JaxStateG(u,v,T,S,eta)

# dt dependence: run FS-only at dt=60, 30, 15 — same physical time
print("\n=== dt dependence (FS-only, 120 steps at dt=60 => 2h physical) ===")
for dt in [60, 30, 15]:
    nsteps = int(120*60/dt)
    s = make_bump()
    fs = jax.jit(lambda s: _free_surface_step_fd(s.eta, s.u, s.v, params, None, None, dt/2.0))
    for _ in range(nsteps):
        eta,u,v = fs(s)
        s = JaxStateG(u,v,s.T,s.S,eta)
        for _ in range(1):  # one half-step per call; need 2 per full dt
            pass
    # redo cleanly: 2 half-steps per full step
    s = make_bump()
    @jax.jit
    def full_fs(s):
        eta,u,v = _free_surface_step_fd(s.eta, s.u, s.v, params, None, None, dt/2.0)
        eta,u,v = _free_surface_step_fd(eta, u, v, params, None, None, dt/2.0)
        return JaxStateG(u,v,s.T,s.S,eta)
    for _ in range(nsteps):
        s = full_fs(s)
    print(f"  dt={dt:3d} nsteps={nsteps:4d}: KE={ke(s):.6e} PE={pe(s):.6e} "
          f"E_total={ke(s)+pe(s):.6e} max|eta|={float(jnp.max(jnp.abs(s.eta))):.4f}")

print("\n=== energy evolution FS-only dt=60 (200 steps) ===")
s = make_bump()
print(f"  {'step':>4} {'KE':>12} {'PE':>12} {'E_tot':>12} {'max|eta|':>9}")
for i in range(200):
    eta,u,v = _free_surface_step_fd(s.eta, s.u, s.v, params, None, None, 30.0)
    eta,u,v = _free_surface_step_fd(eta, u, v, params, None, None, 30.0)
    s = JaxStateG(u,v,s.T,s.S,eta)
    if i%20==0 or i==199:
        print(f"  {i:4d} {ke(s):12.4e} {pe(s):12.4e} {ke(s)+pe(s):12.4e} "
              f"{float(jnp.max(jnp.abs(s.eta))):9.4f}")
