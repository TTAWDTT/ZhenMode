"""DECISIVE: regional SPECTRAL solver, real WOA stratification, dt=60, 1400 steps.
The stability_scan (e8ae3cc) only ran 2h (24 steps @ dt=300, or 120 @ dt=60) and
declared dt=300 STABLE. The FD global solver blows up ~step 1000-1400. Does the
SPECTRAL solver ALSO blow up at 1400 steps under real stratification, or does its
exact matrix-exp linear step keep it stable where the FD explicit step fails?
  spectral stable @1400 => FD discretization (explicit linear step) is the cause
  spectral blows @1400  => shared physics; neither structure is enough
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax
import jax.numpy as jnp
import numpy as np
from config import DEFAULT_CONFIG
from grid import make_grid
from jax_solver import make_solver, RHO_0, G_EARTH

grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
print(f"regional spectral: {grid.nx}x{grid.ny}x{grid.nz} dx={grid.dx:.0f} f0={grid.f0:.4e}")
d = np.load('src/_cache_regional_init.npz')
T_init = d['T']; S_init = d['S']
print(f"T range [{T_init.min():.2f},{T_init.max():.2f}] S [{S_init.min():.2f},{S_init.max():.2f}]")

physics = DEFAULT_CONFIG.physics
dt = 60.0
step, init_state_fn, diag = make_solver(grid, physics, dt, forcing=None,
                                        eos_type='linear', T_init=T_init, S_init=S_init)

wm = np.array(grid.ocean_mask, dtype=np.float64)
dz = np.array(grid.dz); dz_norm = dz/dz.sum(); H=float(dz.sum())
wm_j = jnp.array(wm); dz_norm_j = jnp.array(dz_norm)
@jax.jit
def energy_j(state):
    e=state.eta; u=state.u; v=state.v
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1); vbt=jnp.sum(va*dz_norm_j,-1)
    return (0.5*H*jnp.sum((ubt**2+vbt**2)*wm_j)+0.5*G_EARTH*H*jnp.sum(e**2*wm_j))
@jax.jit
def diag_j(state):
    ua=0.5*(state.u[...,:-1]+state.u[...,1:]); va=0.5*(state.v[...,:-1]+state.v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1,keepdims=True); vbt=jnp.sum(va*dz_norm_j,-1,keepdims=True)
    wm3=wm_j[:,:,None]
    kebc=0.5*H*jnp.sum(((ua-ubt)**2+(va-vbt)**2)*wm3)
    return (jnp.max(jnp.abs(state.eta)), jnp.max(jnp.abs(state.u)), kebc,
            jnp.isfinite(state.eta).all())

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
print("=== REGIONAL SPECTRAL, real stratification, dt=60, advection ON ===")
print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10} {'KE_bc':>12}")
for k in range(1400):
    state = step(state)
    if (k+1) in (100,200,400,800,1000,1200,1400):
        E=float(energy_j(state)); me, mu, kebc, fin = diag_j(state)
        print(f"{k+1:>5} {E:>12.4e} {float(me):>10.4e} {float(mu):>10.4e} {float(kebc):>12.4e}")
        if not bool(fin): print(f"  NaN step {k+1}"); break
print("\nDONE.")
