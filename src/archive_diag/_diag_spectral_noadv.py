"""DECISIVE-2: regional SPECTRAL solver, no-advection (residual PGF only).
Does the spectral baroclinic KE stay FLAT with advection OFF, or grow like FD?
  flat   => spectral linear step prevents the baroclinic PGF injection entirely
  growth => spectral stability comes from advection side, not the linear step
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax
import jax.numpy as jnp
import numpy as np
from config import DEFAULT_CONFIG
from grid import make_grid
import jax_solver as S
from jax_solver import make_solver, RHO_0, G_EARTH

grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
d = np.load('src/_cache_regional_init.npz')
T_init = d['T']; S_init = d['S']
physics = DEFAULT_CONFIG.physics
dt = 60.0

# Monkeypatch advection to zero BEFORE make_solver (so JIT trace picks up zeros)
def _zero_adv_u(u, v, w, p):
    return jnp.zeros_like(v), jnp.zeros_like(v)
def _zero_adv_s(T, u, v, w, p):
    return jnp.zeros_like(T)
S._advection_flux_form = _zero_adv_u
S._advection_scalar = _zero_adv_s

step, init_state_fn, diag = make_solver(grid, physics, dt, forcing=None,
                                        eos_type='linear', T_init=T_init, S_init=S_init)

wm = np.array(grid.ocean_mask, dtype=np.float64)
dz = np.array(grid.dz); dz_norm = dz/dz.sum(); H=float(dz.sum())
wm_j = jnp.array(wm); dz_norm_j = jnp.array(dz_norm); wm3 = wm_j[:,:,None]
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
    kebc=0.5*H*jnp.sum(((ua-ubt)**2+(va-vbt)**2)*wm3)
    return (jnp.max(jnp.abs(state.eta)), jnp.max(jnp.abs(state.u)), kebc,
            jnp.isfinite(state.eta).all())

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
print("=== REGIONAL SPECTRAL, no advection (residual PGF + linear step) ===")
print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10} {'KE_bc':>12}")
for k in range(1400):
    state = step(state)
    if (k+1) in (100,400,800,1200,1400):
        E=float(energy_j(state)); me, mu, kebc, fin = diag_j(state)
        print(f"{k+1:>5} {E:>12.4e} {float(me):>10.4e} {float(mu):>10.4e} {float(kebc):>12.4e}")
        if not bool(fin): print(f"  NaN step {k+1}"); break
print("\nDONE.")
