"""Where does the injection live by latitude band? Run no-adv from REST, bin
KE_bc by |lat| band to see if the equator (f->0, no geostrophic balance) dominates."""
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
lat = np.array(grid.lat)  # (ny,)
# latitude band masks: |lat|<2, 2-10, 10-30, >30
bands = [('|lat|<2', np.abs(lat)<2), ('2-10', (np.abs(lat)>=2)&(np.abs(lat)<10)),
         ('10-30', (np.abs(lat)>=10)&(np.abs(lat)<30)), ('>30', np.abs(lat)>=30)]

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

# per-band KE_bc: weight by lat-band column mask
band_masks = []
for name, m in bands:
    col = np.broadcast_to(m[None,:,None], (grid.nx, grid.ny, 13)).astype(np.float64)
    band_masks.append((name, jnp.array(col)))
dz_norm_j = jnp.array(dz_norm); wm_3d = jnp.array(grid.wet_mask)[:,:,None]
band_stack = jnp.stack([m for _,m in band_masks], axis=-1)  # (nx,ny,13,nb)
@jax.jit
def kebc_bands(state):
    u=state.u; v=state.v
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1,keepdims=True); vbt=jnp.sum(va*dz_norm_j,-1,keepdims=True)
    u_bc=ua-ubt; v_bc=va-vbt
    e = 0.5*H*(u_bc**2+v_bc**2)*wm_3d  # (nx,ny,13)
    return jnp.sum(e[...,None]*band_stack, axis=(0,1,2))  # (nb,)

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
hdr = "stp  " + "  ".join(f"{n:>10}" for n,_ in bands)
print("=== REST init, no adv, KE_bc by lat band ===")
print(hdr)
for k in range(800):
    state = step_no_adv(state)
    if (k+1) in (100,400,800):
        vals = np.array(kebc_bands(state))
        print(f"{k+1:>4}  " + "  ".join(f"{v:>10.3e}" for v in vals))
print("DONE.")
