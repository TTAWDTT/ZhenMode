"""DECISIVE FULL-STEP test: equator-masked geostrophic init + advection ON.
Init u_geo from the density PGF where |f|>f_min (mid/high lats); rest (u=0)
in the equatorial band where geostrophy is singular. Run the FULL step
(advection ON) and see if the explosion is averted.
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

physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

# Build equator-masked geostrophic init
state_rest = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
pressure = _compute_hydrostatic_pressure(state_rest, params)
pgf_x = -_d_dx(pressure, params) / RHO_0
pgf_y = -_d_dy(pressure, params) / RHO_0
f3d = params.f[:, :, None]
# Taper f to a floor to avoid singularity: use max(|f|, f_floor) with smooth taper
f_floor = 2 * 7.2921e-5 * np.sin(np.radians(3.0))  # f at 3 deg
f_mag = jnp.abs(f3d)
f_eff = jnp.where(f_mag > f_floor, f3d, f_floor * jnp.sign(f3d + 1e-30))
u_geo = pgf_y / f_eff
v_geo = -pgf_x / f_eff
# Only apply where |lat|>3 deg (equator mask); taper 2-3 deg
lat = np.array(grid.lat)
taper = np.clip((np.abs(lat) - 2.0) / 1.0, 0.0, 1.0)  # 0 for |lat|<2, 1 for |lat|>3
taper3d = jnp.array(taper)[None, :, None]
u_geo = u_geo * taper3d * params.wet_mask_z
v_geo = v_geo * taper3d * params.wet_mask_z
v_geo = v_geo * params.interior_mask_z
del pressure, pgf_x, pgf_y
state_geo = JaxStateG(u_geo, v_geo, state_rest.T, state_rest.S, state_rest.eta)

wm_j = jnp.array(grid.wet_mask); dz_norm_j = jnp.array(dz_norm); wm_3d = wm_j[:,:,None]
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
    kebc=0.5*H*jnp.sum(((ua-ubt)**2+(va-vbt)**2)*wm_3d)
    return (jnp.max(jnp.abs(state.eta)), jnp.max(jnp.abs(state.u)), kebc,
            jnp.isfinite(state.eta).all())

step_full = jax.jit(lambda s: _step_impl(s, params))
mu0 = float(jnp.max(jnp.abs(state_geo.u)))
print(f"=== GEO init (eq-masked), FULL step (advection ON) ===  init max|u|={mu0:.3f}")
print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10} {'KE_bc':>12}")
state = state_geo
for k in range(1400):
    state = step_full(state)
    if (k+1) in (200,400,800,1000,1200,1400):
        E=float(energy_j(state)); me, mu, kebc, fin = diag_j(state)
        print(f"{k+1:>5} {E:>12.4e} {float(me):>10.4e} {float(mu):>10.4e} {float(kebc):>12.4e}")
        if not bool(fin): print(f"  NaN step {k+1}"); break
print("\nDONE.")
