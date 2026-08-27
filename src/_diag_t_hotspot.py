"""
Tracer-tendency decomposition at the T-hotspot — polcap-OFF, no-sponge (the
production config that FAILs day-10). Reproduces the day-5 state (7200 steps,
the point where max|T| starts its exponential run: 34.4C -> 51.3C by day 6),
then decomposes dT/dt at the hottest surface cells into its components to
find which term injects the runaway heat.

Polcap OFF + sponge OFF matches logs/g10d_nopolcap.log exactly. Imports the
SAME FD operators the solver uses, so the decomposition is exact.
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
                               _laplacian_h, _d2_dz2, _density_anomaly,
                               _divergence_h, _d_dx, _d_dy)
from forcing import heat_flux_meridional, air_temp_profile
from wind_reanalysis import real_wind_forcing
from woa_data import get_initial_fields

# ── build the EXACT failing config (polcap OFF, no sponge) ──
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
    T_atm=T_atm, lambda_bulk=40.0,
    sponge_days=0.0, sponge_cells=0,               # NO sponge (production default)
    T_init=T_init, S_init=S_init,
    polar_cap_rows=0, polar_cap_taper=0,            # polcap OFF (the fix)
    return_params=True)
p = params

state = init_state(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
N = 7200   # day 5 (the turn: T=34.4, just before the 51.3 explosion)
print(f"stepping to day {N*60/86400:.0f} ({N} steps)...", flush=True)
for k in range(N):
    state = step(state)
    if (k + 1) % 1440 == 0:
        Tm = float(np.max(np.abs(np.asarray(state.T))))
        print(f"  day {(k+1)*60/86400:.0f}: max|T|={Tm:.2f}", flush=True)

# ── find the hottest surface cell ──
T_arr = np.asarray(state.T)
surf = T_arr[:, :, 0]
# mask land
ocean2d = np.asarray(grid.wet_mask, dtype=bool)
surf_masked = np.where(ocean2d, surf, -999)
flat_idx = np.argmax(surf_masked)
iy_hot, ix_hot = np.unravel_index(flat_idx, surf.shape)   # surf[iy,ix]? check axes
# axes: T is (nx, ny, nz) per jax_solver_global. surf = T[:,:,0] -> (nx,ny)
ix_hot, iy_hot = np.unravel_index(flat_idx, surf.shape)    # (nx,ny) -> ix,iy
print(f"\nHottest surface cell: lon={lon[ix_hot]:.1f}E lat={lat[iy_hot]:.1f}, T={surf[ix_hot,iy_hot]:.2f}C")

# ── decompose the TRACER tendency ──
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

adv_T = np.asarray(adv_T); diff_h_T = np.asarray(diff_h_T)
diff_v_T = np.asarray(diff_v_T); conv_T = np.asarray(conv_T)
heat_T = np.asarray(heat_T); bulk_T = np.asarray(bulk_T)
w_arr = np.asarray(w)

# split adv_T into horizontal + vertical parts
# vertical advection contribution: recompute -w*dT/dz
dT_dz = np.asarray(_d2_dz2(state.T, p))  # rough; use for sign diagnosis
# better: vertical advection = adv_T - (horizontal part). Just report total adv
# and the w magnitude/sign.

print("\n" + "=" * 90)
print("TRACER TENDENCY DECOMPOSITION at day-5 state (surface layer k=0)")
print("Units: C/s. Positive = warming.")
print("=" * 90)
print(f"{'cell':<22}{'T':>7}{'adv':>11}{'diff_h':>11}{'diff_v':>11}"
      f"{'heat':>11}{'bulk':>11}{'conv':>11}{'SUM':>11}")
print("-" * 90)
cells = [(ix_hot, iy_hot, "HOTSPOT")]
# a control: a quiescent equatorial cell
ctrl_iy = int(np.argmin(np.abs(lat - 0.0)))
ctrl_ix = int(np.argmin(np.abs(lon - 180.0)))
if ocean2d[ctrl_ix, ctrl_iy]:
    cells.append((ctrl_ix, ctrl_iy, "eq(180,0)"))
else:
    cells.append((ix_hot, ctrl_iy, "hotlon_eq"))
for (i, j, lbl) in cells:
    k = 0
    s = (adv_T[i,j,k] + diff_h_T[i,j,k] + diff_v_T[i,j,k]
         + heat_T[i,j,k] + bulk_T[i,j,k] + conv_T[i,j,k])
    print(f"{lbl:<22}{T_arr[i,j,k]:7.2f}"
          f"{adv_T[i,j,k]:11.2e}{diff_h_T[i,j,k]:11.2e}{diff_v_T[i,j,k]:11.2e}"
          f"{heat_T[i,j,k]:11.2e}{bulk_T[i,j,k]:11.2e}{conv_T[i,j,k]:11.2e}{s:11.2e}")

# sanity: full call
dTdt, _ = _compute_tracer_tendency(state, p)
dTdt = np.asarray(dTdt)
print("-" * 90)
print("sanity: _compute_tracer_tendency full-call surface dT/dt:")
for (i, j, lbl) in cells:
    print(f"  {lbl:<20} {dTdt[i,j,0]:11.2e} C/s   ({dTdt[i,j,0]*86400:.3f} C/day)")

# ── vertical structure of the hotspot ──
print("\n" + "=" * 90)
print("VERTICAL PROFILE at hotspot (T, dT/dt, w)")
print("=" * 90)
print(f"{'k':>3}{'z(m)':>8}{'T':>8}{'dT/dt':>12}{'adv':>12}{'diff_v':>12}{'w':>12}")
z_levels = np.array([0,-5,-15,-30,-50,-75,-100,-150,-200,-300,-500,-1000,-2000,-4000])
for k in range(p.nz):
    z = z_levels[k] if k < len(z_levels) else 0
    print(f"{k:3d}{z:8d}{T_arr[ix_hot,iy_hot,k]:8.2f}"
          f"{dTdt[ix_hot,iy_hot,k]:12.2e}{adv_T[ix_hot,iy_hot,k]:12.2e}"
          f"{diff_v_T[ix_hot,iy_hot,k]:12.2e}{w_arr[ix_hot,iy_hot,k]:12.2e}")

# ── velocity + vertical advection at hotspot ──
print("\n" + "=" * 90)
print("VELOCITY at hotspot (surface)")
print("=" * 90)
u_arr = np.asarray(state.u); v_arr = np.asarray(state.v)
print(f"u_sfc={u_arr[ix_hot,iy_hot,0]:.3e}  v_sfc={v_arr[ix_hot,iy_hot,0]:.3e}  "
      f"w_sfc(k=0)={w_arr[ix_hot,iy_hot,0]:.3e}  w(k=1)={w_arr[ix_hot,iy_hot,1]:.3e}")
dTdz01 = T_arr[ix_hot,iy_hot,0] - T_arr[ix_hot,iy_hot,1]
print(f"dT/dz(0->1)={dTdz01:.2f} C/10m   -w[1]*dTdz={-w_arr[ix_hot,iy_hot,1]*dTdz01:.3e} C/s")
print("(w>0 = upwelling; warm surface over cold deep => dT/dz>0; upwelling cools, downwelling warms)")

# ── horizontal divergence at hotspot ──
div_h = np.asarray(_divergence_h(state.u, state.v, p)[:,:,0])
print(f"\nhorizontal div_h(sfc) at hotspot = {div_h[ix_hot,iy_hot]:.3e} /s")
print("(convergent flow (div<0) drives downwelling (w<0 at surface) -> warm-water subduction)")

# ── where is the hotspot geographically? neighbors? ──
print("\n" + "=" * 90)
print("HOTSPOT GEOGRAPHIC CONTEXT")
print("=" * 90)
print(f"location: lon={lon[ix_hot]:.1f}E lat={lat[iy_hot]:.1f}")
# bathymetry depth at hotspot
depth_field = np.asarray(grid.depth)
print(f"bathymetry depth: {depth_field[ix_hot,iy_hot]:.0f} m")
# land neighbors
nb_land = 0
for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
    ii=(ix_hot+di)%grid.nx; jj=iy_hot+dj
    if 0<=jj<grid.ny and not ocean2d[ii,jj]:
        nb_land+=1
print(f"land neighbors (4-connected): {nb_land}")
print(f"surface mask: {np.asarray(p.surface_mask[ix_hot,iy_hot,0])}, "
      f"wet: {np.asarray(p.wet_mask_z[ix_hot,iy_hot,0])}")
