"""TEST: geostrophic initialization. Instead of starting from u=v=0 (massively
unbalanced w.r.t. the climatological density PGF), start from the geostrophic
velocity that balances the baroclinic PGF: f*k x u_geo = -grad(p_bc)/rho0.
  u_geo = -(1/(rho0*f)) * dp/dy
  v_geo =  (1/(rho0*f)) * dp/dx
If this kills the linear injection => the imbalance of u=0 vs real density
was the root cause, and the fix is geostrophic init (cheap, standard).
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
import jax_solver_global as G
from jax_solver_global import (make_solver_global, JaxStateG, RHO_0, G_EARTH,
                               _step_impl, _compute_hydrostatic_pressure,
                               _d_dx, _d_dy)
from forcing import air_temp_profile, heat_flux_meridional

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
d = np.load('src/_cache_init.npz')
T_init = d['T']; S_init = d['S']
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])
dz = np.array(grid.dz); dz_norm = dz/dz.sum(); H=4000.0
wm = np.array(grid.wet_mask) > 0.5
f2d = np.array(grid.f)  # (nx, ny)

physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

# Zero advection for clean injection test
def _zero_adv_u(u, v, w, p):
    z = jnp.zeros_like(v); return z, z
def _zero_adv_s(T, u, v, w, p):
    return jnp.zeros_like(T)
G._advection_flux_form = _zero_adv_u
G._advection_scalar = _zero_adv_s
step_no_adv = jax.jit(lambda s: _step_impl(s, params))

# Build geostrophic initial velocity from the density PGF.
state_rest = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
pressure = _compute_hydrostatic_pressure(state_rest, params)  # (nx,ny,nz)
# PGF per level: -grad(p)/rho0
pgf_x = -_d_dx(pressure, params) / RHO_0   # du/dt forcing
pgf_y = -_d_dy(pressure, params) / RHO_0
# Geostrophic balance: du/dt=0 => f*v = pgf_x? Let's be careful.
# Momentum: du/dt - f*v = pgf_x ; dv/dt + f*u = pgf_y
# Steady geostrophic: -f*v = pgf_x => v = -pgf_x/f ;  f*u = pgf_y => u = pgf_y/f
f3d = params.f[:, :, None]
f_safe = jnp.where(jnp.abs(f3d) > 1e-8, f3d, 1e-8)  # avoid div-by-0 at equator
u_geo = pgf_y / f_safe
v_geo = -pgf_x / f_safe
# At equator (|f|<1e-8), zero the geostrophic velocity (no balance possible)
mask_eq = (jnp.abs(f3d) > 1e-8).astype(jnp.float64)
u_geo = u_geo * mask_eq * params.wet_mask_z
v_geo = v_geo * mask_eq * params.wet_mask_z
# zero normal v at boundary
v_geo = v_geo * params.interior_mask_z

state_geo = JaxStateG(u_geo, v_geo, state_rest.T, state_rest.S, state_rest.eta)

wm_j = jnp.array(grid.wet_mask); dz_norm_j = jnp.array(dz_norm)
wm_3d = wm_j[:, :, None]
@jax.jit
def ke_bc_j(state):
    u=state.u; v=state.v
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1,keepdims=True); vbt=jnp.sum(va*dz_norm_j,-1,keepdims=True)
    u_bc=ua-ubt; v_bc=va-vbt
    return 0.5*H*jnp.sum((u_bc**2+v_bc**2)*wm_3d)
@jax.jit
def diag_j(state):
    return (jnp.max(jnp.abs(state.u)), ke_bc_j(state), jnp.isfinite(state.eta).all())

def run(label, state0):
    state = state0
    print(f"\n=== {label} (no adv) ===")
    mu0, kebc0, _ = diag_j(state)
    print(f"{'stp':>5} {'max|u|':>10} {'KE_bc':>12}   (init max|u|={float(mu0):.3e})")
    for k in range(800):
        state = step_no_adv(state)
        if (k+1) in (100,400,800):
            mu, kebc, fin = diag_j(state)
            print(f"{k+1:>5} {float(mu):>10.4e} {float(kebc):>12.4e}")
            if not bool(fin): print(f"  NaN step {k+1}"); break

run("REST init (u=v=0)", state_rest)
run("GEOSTROPHIC init (u_geo balances density PGF)", state_geo)
print("\nDONE.")
