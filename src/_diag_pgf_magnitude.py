"""Investigate the PGF magnitude and u_geo=121 m/s. Is it physical (real steep
climatology fronts) or a discretization artifact (central-diff PGF amplification
at 1deg near coasts/fronts)? Check distribution of |PGF| and |u_geo|: where are
the extremes (coasts? equator? specific regions)?"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from jax_solver_global import (make_solver_global, RHO_0, G_EARTH,
                               _compute_hydrostatic_pressure, _d_dx, _d_dy)
from forcing import air_temp_profile, heat_flux_meridional

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
d = np.load('src/_cache_init.npz')
T_init = d['T']; S_init = d['S']
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
Q_heat = heat_flux_meridional(grid, Q0=0.0)
T_atm = air_temp_profile(grid, T_init[:, :, 0])

physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
pressure = np.array(_compute_hydrostatic_pressure(state, params))
pgf_x = np.array(-_d_dx(jnp.array(pressure), params) / RHO_0)
pgf_y = np.array(-_d_dy(jnp.array(pressure), params) / RHO_0)
pgf_mag = np.sqrt(pgf_x**2 + pgf_y**2)
f3d = np.array(params.f)[:, :, None]
wm = np.array(grid.wet_mask) > 0.5
wm3d = np.broadcast_to(wm[:, :, None], pgf_mag.shape)

# Where is PGF extreme?
pgf_wet = pgf_mag * wm3d
print(f"PGF acceleration [m/s^2] over wet points:")
print(f"  max:  {pgf_wet.max():.4e}")
print(f"  99.9pct: {np.percentile(pgf_wet[wm3d], 99.9):.4e}")
print(f"  99pct:    {np.percentile(pgf_wet[wm3d], 99):.4e}")
print(f"  median:   {np.median(pgf_wet[wm3d]):.4e}")
print(f"  mean:     {pgf_wet[wm3d].mean():.4e}")

# u_geo magnitude
f_safe = np.where(np.abs(f3d) > 1e-7, f3d, np.nan)
u_geo = pgf_y / f_safe
v_geo = -pgf_x / f_safe
u_geo_mag = np.sqrt(u_geo**2 + v_geo**2) * wm3d
print(f"\nu_geo [m/s] over wet (|f|>1e-7):")
print(f"  max:  {np.nanmax(u_geo_mag):.3f}")
print(f"  99.9pct: {np.nanpercentile(u_geo_mag[np.isfinite(u_geo_mag)], 99.9):.3f}")
print(f"  99pct:    {np.nanpercentile(u_geo_mag[np.isfinite(u_geo_mag)], 99):.3f}")
print(f"  median:   {np.nanmedian(u_geo_mag[np.isfinite(u_geo_mag)]):.3f}")

# Where are the extremes? Find the (lon,lat) of the max u_geo
imax = np.unravel_index(np.nanargmax(np.where(np.isfinite(u_geo_mag), u_geo_mag, -1)), u_geo_mag.shape)
lon = np.array(grid.lon); lat = np.array(grid.lat)
print(f"\nMax u_geo at ijk={imax}, lon={lon[imax[0]]:.1f}, lat={lat[imax[1]]:.1f}, k={imax[2]}")
print(f"  PGF there: {pgf_wet[imax]:.4e}, f={f3d[imax[0],imax[1],0]:.4e}")
# top 5 locations
flat_idx = np.argsort(np.where(np.isfinite(u_geo_mag), u_geo_mag, -1).ravel())[-10:][::-1]
print("Top-10 u_geo locations (lon, lat, k, u_geo):")
for fi in flat_idx:
    ii, jj, kk = np.unravel_index(fi, u_geo_mag.shape)
    print(f"  lon={lon[ii]:6.1f} lat={lat[jj]:6.1f} k={kk:2d} u_geo={u_geo_mag[ii,jj,kk]:.3f} PGF={pgf_wet[ii,jj,kk]:.3e} f={f3d[ii,jj,0]:.3e} depth_water={wm3d[ii,jj,kk]}")
print("DONE.")
