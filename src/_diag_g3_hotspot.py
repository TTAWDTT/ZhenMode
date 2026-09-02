"""
Tendency decomposition at the j=1 south-wall hotspot (day 4 state).

Reproduces the day-4 state (5760 steps, sponge16/3d, bulk40), then at the 3
hotspot cells (lon 138.5/232.5/285.5, j=1) and a control cell, decomposes the
TRACER tendency dT/dt into its components to find which term injects the heat.

Imports the SAME FD operators the solver uses, so the decomposition is exact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from dataclasses import replace

from config import DEFAULT_CONFIG, GlobalGridConfig, PhysicsConfig, RHO_0, C_P
from grid import make_global_grid
from jax_solver_global import (make_solver_global, _compute_tracer_tendency,
                               _compute_vertical_velocity, _advection_scalar,
                               _laplacian_h, _d2_dz2, _density_anomaly)
from forcing import heat_flux_meridional, air_temp_profile
from wind_reanalysis import real_wind_forcing
from woa_data import get_initial_fields

gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file,
                        smooth_passes=30, min_depth=100.0)
lat = np.asarray(grid.lat); lon = np.asarray(grid.lon)
physics = replace(PhysicsConfig(), nu_h=5e6, nu_bi=0.0, kappa_bi=0.0)
Q_heat = heat_flux_meridional(grid, Q0=50.0)
T_init, S_init = get_initial_fields(grid); T_init = np.array(T_init); S_init = np.array(S_init)
T_atm = air_temp_profile(grid, T_init[:, :, 0])
tau_x, tau_y = real_wind_forcing(month_idx=(2023 - 1948) * 12 + 0, grid=grid)

step, init_state, _diag, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=40.0, sponge_days=3.0, sponge_cells=16,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, return_params=True)
p = params

state = init_state(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
for _ in range(5760):
    state = step(state)

# --- decompose the TRACER tendency at the hotspot cells ---
w = _compute_vertical_velocity(state, p)
adv_T = _advection_scalar(state.T, state.u, state.v, w, p)
diff_h_T = p.kappa_h * _laplacian_h(state.T, p)
diff_v_T = p.kappa_v * _d2_dz2(state.T, p)
rho_prime = _density_anomaly(state.T, state.S, p)
unstable = rho_prime[..., :-1] > rho_prime[..., 1:]
conv_mask_3d = jnp.any(unstable, axis=-1, keepdims=True)
conv_T = p.kappa_conv * conv_mask_3d * _d2_dz2(state.T, p)
heat_factor = 1.0 / (RHO_0 * C_P * p.dz_surface)
heat_T = p.Q_heat_2d[:, :, None] * heat_factor * p.surface_mask
bulk_T = (p.lambda_bulk * (p.T_atm_3d - state.T[:, :, 0:1]) * heat_factor * p.surface_mask)

# block until computed
adv_T = np.asarray(adv_T); diff_h_T = np.asarray(diff_h_T)
diff_v_T = np.asarray(diff_v_T); conv_T = np.asarray(conv_T)
heat_T = np.asarray(heat_T); bulk_T = np.asarray(bulk_T)
T_arr = np.asarray(state.T)

cells = []
for target_lon in [138.5, 232.5, 285.5]:
    i = int(np.argmin(np.abs(lon - target_lon))); cells.append((i, 1, f"lon{target_lon}"))
# control: a normal cell at j=1, lon 180 (mid-ocean)
cells.append((int(np.argmin(np.abs(lon - 180.0))), 1, "lon180(control)"))
# control at j=10 (interior, away from wall)
cells.append((int(np.argmin(np.abs(lon - 138.5))), 10, "lon138 j10(interior)"))

print("=" * 80)
print("TRACER TENDENCY DECOMPOSITION at day-4 state (surface layer k=0)")
print("hotspot cells: j=1 (lat -58.5), the 3 south-wall blow-up lons")
print("=" * 80)
print(f"{'cell':<18}{'T':>7}{'adv':>9}{'diff_h':>9}{'diff_v':>9}"
      f"{'heat':>9}{'bulk':>9}{'conv':>9}{'sum':>9}")
print("-" * 80)
for (i, j, lbl) in cells:
    k = 0
    s = (adv_T[i,j,k] + diff_h_T[i,j,k] + diff_v_T[i,j,k]
         + heat_T[i,j,k] + bulk_T[i,j,k] + conv_T[i,j,k])
    print(f"{lbl:<18}{T_arr[i,j,k]:7.1f}"
          f"{adv_T[i,j,k]:9.2e}{diff_h_T[i,j,k]:9.2e}{diff_v_T[i,j,k]:9.2e}"
          f"{heat_T[i,j,k]:9.2e}{bulk_T[i,j,k]:9.2e}{conv_T[i,j,k]:9.2e}{s:9.2e}")

# also: full-tendency call (sanity, should match sum)
dTdt, _ = _compute_tracer_tendency(state, p)
dTdt = np.asarray(dTdt)
print("-" * 80)
print("sanity: _compute_tracer_tendency (full call) surface layer:")
for (i, j, lbl) in cells:
    print(f"  {lbl:<18} dT/dt = {dTdt[i,j,0]:9.2e}")

# --- PART 2: velocity field at the hotspot (what drives the advective heat source?) ---
print("\n" + "=" * 80)
print("VELOCITY & VERTICAL ADVECTION at hotspot (day-4 state)")
print("=" * 80)
u_arr = np.asarray(state.u); v_arr = np.asarray(state.v)
w_arr = np.asarray(w)
# vertical advection term: -w * dT/dz (rough). Sign of w tells upwelling/downwelling.
# dT/dz at surface: T[0]-T[1]
dTdz0 = (T_arr[:,:,0] - T_arr[:,:,1])
print(f"{'cell':<18}{'u_sfc':>9}{'v_sfc':>9}{'w_sfc':>11}{'dT/dz(0)':>10}{'-w*dTdz':>11}")
for (i, j, lbl) in cells:
    print(f"{lbl:<18}{u_arr[i,j,0]:9.3e}{v_arr[i,j,0]:9.3e}{w_arr[i,j,0]:11.3e}"
          f"{dTdz0[i,j]:10.2f}{-w_arr[i,j,0]*dTdz0[i,j]:11.3e}")
print("\n(positive -w*dTdz = warming; w<0 = downwelling of warm surface water INTO layer, "
      "w>0 = upwelling of cold water -> cooling)")
print("NOTE: solver w sign convention — check against adv_T sign above.")

# --- PART 3: horizontal divergence decomposition at hotspot ---
from jax_solver_global import _divergence_h, _d_dx, _d_dy
div_h = _divergence_h(state.u, state.v, p)
dudx = _d_dx(state.u[:,:,0:1], p)[:,:,0]
dvdy = _d_dy(state.v[:,:,0:1], p)[:,:,0]
div_h = np.asarray(div_h[:,:,0]); dudx = np.asarray(dudx); dvdy = np.asarray(dvdy)
print("\n" + "=" * 80)
print("HORIZONTAL DIVERGENCE at hotspot (surface, day-4)")
print("div_h = du/dx + dv/dy ; w is the vertical integral of -div_h")
print("=" * 80)
print(f"{'cell':<18}{'du/dx':>11}{'dv/dy':>11}{'div_h':>11}{'w_sfc':>11}")
for (i, j, lbl) in cells:
    print(f"{lbl:<18}{dudx[i,j]:11.3e}{dvdy[i,j]:11.3e}{div_h[i,j]:11.3e}{w_arr[i,j,0]:11.3e}")
print("\n(j=1 is the first interior row; j=0 is the wall row with v=0. The dv/dy")
print(" stencil at j=1 uses a mirror ghost cell across the wall — check if it's spuriously large.)")

# Is the dv/dy at j=1 driven by v at j=0 (which is masked to 0) vs j=2?
print("\n--- v field around the hotspot (lon138.5) at j=0,1,2 ---")
i138 = int(np.argmin(np.abs(lon - 138.5)))
print(f"  v[i,j=0] (wall) = {v_arr[i138,0,0]:.3e}  (should be 0, masked)")
print(f"  v[i,j=1]        = {v_arr[i138,1,0]:.3e}")
print(f"  v[i,j=2]        = {v_arr[i138,2,0]:.3e}")
print(f"  v[i,j=3]        = {v_arr[i138,3,0]:.3e}")

# --- PART 4: vertical profile of w and div_h at the hotspot vs control ---
print("\n" + "=" * 80)
print("VERTICAL PROFILE of w and div_h at lon138.5 (hot) vs lon180 (control), j=1")
print("=" * 80)
div_h_3d = np.asarray(_divergence_h(state.u, state.v, p))  # (nx,ny,nz)
w_3d = np.asarray(w)  # (nx,ny,nz)
i138 = int(np.argmin(np.abs(lon - 138.5)))
i180 = int(np.argmin(np.abs(lon - 180.0)))
z = np.asarray(grid.z)
print(f"{'k':>3}{'z(m)':>8}{'div_h[h]':>11}{'w[h]':>11}  | {'div_h[c]':>11}{'w[c]':>11}")
for k in range(grid.nz):
    print(f"{k:3d}{z[k]:8.0f}{div_h_3d[i138,1,k]:11.3e}{w_3d[i138,1,k]:11.3e}  | "
          f"{div_h_3d[i180,1,k]:11.3e}{w_3d[i180,1,k]:11.3e}")

# --- PART 5: is the spurious w from ghost-water (below-seafloor) layers? ---
print("\n" + "=" * 80)
print("GHOST-WATER CHECK: depth vs nz, and u/v in below-seafloor layers")
print("=" * 80)
wet_mask_z = np.asarray(p.wet_mask_z)  # (nx,ny,nz) 1 above seafloor, 0 below
i138 = int(np.argmin(np.abs(lon - 138.5)))
i180 = int(np.argmin(np.abs(lon - 180.0)))
print(f"lon138.5 j=1: depth={np.asarray(grid.depth)[i138,1]:.0f}m")
print(f"  wet_mask_z profile (1=wet, 0=ghost/below seafloor): {wet_mask_z[i138,1].astype(int).tolist()}")
print(f"  u in ghost layers (where wet_mask_z=0): u={u_arr[i138,1][wet_mask_z[i138,1]==0].tolist()}")
print(f"  v in ghost layers: v={v_arr[i138,1][wet_mask_z[i138,1]==0].tolist()}")
print(f"  div_h in ghost layers: {div_h_3d[i138,1][wet_mask_z[i138,1]==0].tolist()}")
print()
print(f"lon180(control) j=1: depth={np.asarray(grid.depth)[i180,1]:.0f}m")
print(f"  wet_mask_z profile: {wet_mask_z[i180,1].astype(int).tolist()}")
# KEY: does _compute_vertical_velocity mask div_h per-layer before cumsum?
# It does: w = -cumsum(div_avg*dz) then * wet_mask_z at the END only.
# So div_h in ghost layers ENTERS the cumsum. If ghost-layer div_h is nonzero
# (from unmasked u/v there), it corrupts w in ALL layers above (cumsum).
import inspect
print()
print("=== _compute_vertical_velocity source (does it mask div_h before cumsum?) ===")
print(inspect.getsource(_compute_vertical_velocity))
