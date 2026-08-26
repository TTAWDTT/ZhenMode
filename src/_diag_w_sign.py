"""Verify _compute_vertical_velocity: does div_h*dz cumsum (bottom->top) match w?
And: what does w look like if we mask div_h per-layer by wet_mask_z BEFORE cumsum?"""
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
Q_heat = heat_flux_meridional(grid, Q0=50.0)
T_init, S_init = get_initial_fields(grid); T_init=np.array(T_init); S_init=np.array(S_init)
T_atm = air_temp_profile(grid, T_init[:,:,0])
tau_x, tau_y = real_wind_forcing(month_idx=(2023-1948)*12+0, grid=grid)
step, init_state, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x,tau_y,Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=40.0, sponge_days=3.0, sponge_cells=16,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, return_params=True)
p = params
state = init_state(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
for _ in range(5760):
    state = step(state)

div_h = np.asarray(_divergence_h(state.u, state.v, p))  # (nx,ny,nz)
w_code = np.asarray(_compute_vertical_velocity(state, p))  # (nx,ny,nz)
dz1d = np.asarray(p.dz_3d).ravel()  # (nz,)
wet_z = np.asarray(p.wet_mask_z)  # (nx,ny,nz)

i138 = int(np.argmin(np.abs(lon-138.5)))
i180 = int(np.argmin(np.abs(lon-180.0)))
print("="*78)
print("CUMSUM VERIFICATION: does -cumsum(div_avg*dz, bottom->top) == w_code?")
print("div_avg[k] = 0.5*(div_h[k]+div_h[k+1]) on interfaces; w at layer centers.")
print("="*78)
for i,j,lbl in [(i138,1,'hot lon138'),(i180,1,'ctrl lon180')]:
    # interface divergence (nz-1)
    div_avg = 0.5*(div_h[i,j,:-1]+div_h[i,j,1:])  # (nz-1,)
    dz_int = dz1d[:-1]  # (nz-1,)
    integ = div_avg*dz_int  # (nz-1,)
    # cumsum bottom->top: reverse, cumsum, reverse -> value at each interface from bottom
    cs = np.cumsum(integ[::-1])[::-1]  # (nz-1,) interface w (negative = upwelling convention?)
    w_manual_iface = -cs  # interface
    print(f"\n{lbl} (depth={np.asarray(grid.depth)[i,j]:.0f}m):")
    print(f"  div_h[layer]   = {div_h[i,j]}")
    print(f"  w_code[layer]  = {w_code[i,j]}")
    print(f"  (interface w_manual = -cumsum(div_avg*dz) bottom->top = {w_manual_iface})")
    print(f"  sum of integ (should ~ -w_surface for mass balance): {integ.sum():.3e} vs w_code[0]={w_code[i,j,0]:.3e}")

print("\n"+"="*78)
print("PER-LAYER-MASK TEST: mask div_h by wet_mask_z BEFORE cumsum — does w clean up?")
print("="*78)
div_h_masked = div_h * wet_z  # zero div_h in ghost layers before cumsum
for i,j,lbl in [(i138,1,'hot lon138'),(i180,1,'ctrl lon180')]:
    div_avg = 0.5*(div_h_masked[i,j,:-1]+div_h_masked[i,j,1:])
    integ = div_avg*dz_int
    cs = np.cumsum(integ[::-1])[::-1]
    w_masked_iface = -cs
    print(f"  {lbl}: w_code[0]={w_code[i,j,0]:.3e}  w_masked_iface[0]={w_masked_iface[0]:.3e}  (diff={w_code[i,j,0]-w_masked_iface[0]:.3e})")
