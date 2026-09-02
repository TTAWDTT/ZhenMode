"""Key test: does masking div_h per-layer by wet_mask_z BEFORE cumsum fix the spurious w?
Also a clean synthetic sign test using the REAL params structure."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from dataclasses import replace
from config import DEFAULT_CONFIG, GlobalGridConfig, PhysicsConfig
from grid import make_global_grid
from jax_solver_global import make_solver_global, _divergence_h, _compute_vertical_velocity
from forcing import heat_flux_meridional, air_temp_profile
from wind_reanalysis import real_wind_forcing
from woa_data import get_initial_fields

gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
lat = np.asarray(grid.lat); lon = np.asarray(grid.lon)
physics = replace(PhysicsConfig(), nu_h=5e6, nu_bi=0.0, kappa_bi=0.0)
T_init, S_init = get_initial_fields(grid); T_init=np.array(T_init); S_init=np.array(S_init)
tau_x, tau_y = real_wind_forcing(month_idx=(2023-1948)*12+0, grid=grid)
step, init_state, _, p = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x,tau_y,heat_flux_meridional(grid,50.0)),
    eos_type='linear', T_atm=air_temp_profile(grid,T_init[:,:,0]), lambda_bulk=40.0,
    sponge_days=3.0, sponge_cells=16, T_init=T_init, S_init=S_init,
    polar_cap_rows=0, return_params=True)

state = init_state(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
for _ in range(5760):
    state = step(state)

div_h = np.asarray(_divergence_h(state.u, state.v, p))  # (nx,ny,nz=14)
w_code = np.asarray(_compute_vertical_velocity(state, p))  # (nx,ny,14)
wet_z = np.asarray(p.wet_mask_z)  # (nx,ny,14)
dz1d = np.asarray(p.dz_3d).ravel()  # (13,) layer thicknesses
nz = div_h.shape[2]
print(f"div_h shape {div_h.shape}, w_code shape {w_code.shape}, dz1d len {len(dz1d)}, wet_z shape {wet_z.shape}")

i138 = int(np.argmin(np.abs(lon-138.5)))
i180 = int(np.argmin(np.abs(lon-180.0)))

# Manual w from div_h (matching the code's algorithm):
# code: div_avg = 0.5*(div_h[:-1]+div_h[1:])  -> (nz-1)=13 interfaces
#       integrand = div_avg * dz_3d  (dz_3d is (1,1,13))
#       w[interfaces] = -cumsum(integrand reversed) reversed
# layer-center w[k] for k in 0..nz-1: w_code has nz entries.
def manual_w(div_col, dz_col):
    div_avg = 0.5*(div_col[:-1]+div_col[1:])  # (nz-1,)
    integ = div_avg * dz_col  # (nz-1,)
    cs = np.cumsum(integ[::-1])[::-1]  # interface, bottom->top
    return -cs  # (nz-1,)

print("\n=== hot lon138 j=1 ===")
print("div_h     :", div_h[i138,1])
print("w_code    :", w_code[i138,1])
wman = manual_w(div_h[i138,1], dz1d)
print("w_manual(iface):", wman, "len", len(wman))
print("w_code[0]=", w_code[i138,1,0], "  w_manual[0]=", wman[0])

print("\n=== ctrl lon180 j=1 ===")
print("w_code[0]=", w_code[i180,1,0])
wman_c = manual_w(div_h[i180,1], dz1d)
print("w_manual[0]=", wman_c[0])

# --- the FIX test: mask div_h by wet_z per-layer before cumsum ---
print("\n=== FIX: mask div_h by wet_mask_z per-layer, then recompute w ===")
div_h_masked = div_h * wet_z
for i,j,lbl in [(i138,1,'hot lon138'),(i180,1,'ctrl lon180')]:
    wman_m = manual_w(div_h_masked[i,j], dz1d)
    print(f"  {lbl}: w_code[0]={w_code[i,j,0]:.3e}  w_masked[0]={wman_m[0]:.3e}  (wet_z col: {wet_z[i,j].astype(int).tolist()})")

# --- synthetic sign test: u=0.001*lon => du/dx>0 (divergence) => expect w>0 (up) ---
print("\n=== SYNTHETIC SIGN TEST (real params) ===")
u_syn = jnp.array(0.001*np.asarray(grid.lon)[None,:,None].astype(float))  # (1,ny,1) -> broadcast
u_syn = jnp.broadcast_to(u_syn, state.u.shape)
v_syn = jnp.zeros_like(state.u)
class S: pass
ss = S(); ss.u=u_syn; ss.v=v_syn
div_syn = np.asarray(_divergence_h(u_syn, v_syn, p))
w_syn = np.asarray(_compute_vertical_velocity(ss, p))
print(f"u=0.001*lon (du/dx>0, divergence). Sample at (180,60): div_h={div_syn[180,60,0]:.3e} w={w_syn[180,60,0]:.3e}")
print("EXPECT: div>0 => w>0 (upwelling). VERDICT:", "CORRECT" if w_syn[180,60,0]>0 else "SIGN BUG" if w_syn[180,60,0]<0 else "zero")
