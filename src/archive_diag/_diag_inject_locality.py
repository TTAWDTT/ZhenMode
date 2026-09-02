"""Where does the FD no-adv baroclinic energy growth localize? If it's at
coastlines/closed walls (mask discontinuities), the mask discretization is the
injector. If it's interior (broadband), it's a propagation/balance issue."""
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
from jax_solver_global import (make_solver_global, RHO_0, G_EARTH, _step_impl)
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

def _zero_adv_u(u, v, w, p):
    return jnp.zeros_like(v), jnp.zeros_like(v)
def _zero_adv_s(T, u, v, w, p):
    return jnp.zeros_like(T)
G._advection_flux_form = _zero_adv_u
G._advection_scalar = _zero_adv_s
step_no_adv = jax.jit(lambda s: _step_impl(s, params))

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
wm = np.array(grid.wet_mask) > 0.5
wm_pad = np.pad(wm, 1, mode='constant', constant_values=False)
coast = np.zeros_like(wm)
for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
    coast |= wm & ~wm_pad[1+di:1+di+wm.shape[0], 1+dj:1+dj+wm.shape[1]]
print(f"wet cells: {wm.sum()}, coastline cells: {coast.sum()}")
from scipy.ndimage import binary_dilation
coast_buf = binary_dilation(coast, iterations=5) & wm
interior = wm & ~coast_buf
print(f"coast-buffer cells: {coast_buf.sum()}, deep-interior cells: {interior.sum()}")

for k in range(800):
    state = step_no_adv(state)
u = np.array(state.u); v = np.array(state.v)
ua = 0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
ke_2d = (0.5*H*(ua**2+va**2)).sum(-1)
ke_coast = ke_2d[coast].sum(); ke_buf = ke_2d[coast_buf].sum()
ke_int = ke_2d[interior].sum(); ke_total = ke_2d[wm].sum()
print(f"\nAfter 800 no-adv steps (max|u|={np.max(np.abs(u)):.4e}):")
print(f"  KE total (wet):     {ke_total:.4e}")
print(f"  KE coast (1 cell):  {ke_coast:.4e}  ({100*ke_coast/max(ke_total,1e-30):.1f}%)")
print(f"  KE coast-buffer(5): {ke_buf:.4e}  ({100*ke_buf/max(ke_total,1e-30):.1f}%)")
print(f"  KE deep-interior:   {ke_int:.4e}  ({100*ke_int/max(ke_total,1e-30):.1f}%)")
print("DONE.")
