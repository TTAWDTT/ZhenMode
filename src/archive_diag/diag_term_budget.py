"""Term-by-term tendency budget at the polar-edge divergence point.

Run to ~step 150 (pre-divergence) under no-wind, then dump the per-term
momentum tendency at the (+79.5, +124.5) point to find WHICH operator term
is growing unboundedly.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from dataclasses import replace

from config import PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from woa_data import get_initial_fields
import jax_solver_global as G
from jax_solver_global import (make_solver_global, _laplacian_h, _d2_dz2,
    _compute_vertical_velocity, _advection_flux_form, _compute_pressure_gradient,
    _d_dx, _d_dy, make_fd_params, JaxStateG, FDPhysParams, RHO_0, G_EARTH)

BATHY = r"C:\Users\zhen.luo\Desktop\ETOPO_2022_v1_r3600x1800_surface.nc"

gcfg = replace(GlobalGridConfig(), lat_max=80.0, ny=160)
grid = make_global_grid(gcfg, BATHY, smooth_passes=20, min_depth=25.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
phys = replace(PhysicsConfig(), nu_h=5e5, nu_bi=0.0, kappa_bi=0.0)
step, init_state, _ = make_solver_global(grid, phys, 60.0, forcing=None,
                                         polar_cap_rows=0)

# Build a full FDPhysParams (the term functions need physics fields).
base = make_fd_params(grid)
nx, ny, nz = base.nx, base.ny, base.nz
H_sw = float(np.sum(grid.dz))
p = FDPhysParams(
    dx_2d=base.dx_2d, dy=base.dy, cos_lat=base.cos_lat,
    inv_dx=base.inv_dx, inv_dy=base.inv_dy,
    inv_dx2=base.inv_dx2, inv_dy2=base.inv_dy2,
    f=base.f, wet_mask=base.wet_mask, wet_mask_3d=base.wet_mask_3d,
    wet_mask_z=base.wet_mask_z,
    dz_denom_interior=base.dz_denom_interior,
    dz_bnd_top=base.dz_bnd_top, dz_bnd_bot=base.dz_bnd_bot,
    d2z_hm=base.d2z_hm, d2z_hp=base.d2z_hp, d2z_denom=base.d2z_denom,
    d2z_h0_top=base.d2z_h0_top, d2z_h0_bot=base.d2z_h0_bot,
    dz_3d=base.dz_3d, dz_surface=base.dz_surface,
    surface_mask=base.surface_mask, bottom_mask=base.bottom_mask,
    nx=nx, ny=ny, nz=nz,
    nu_h=phys.nu_h, nu_v=phys.nu_v, kappa_h=phys.kappa_h, kappa_v=phys.kappa_v,
    kappa_conv=phys.kappa_conv, nu_bi=0.0, kappa_bi=0.0,
    T_ref=phys.T_ref, S_ref=phys.S_ref, eos_type='linear',
    r_bot=phys.r_bot, cd=phys.cd, bottom_friction=phys.bottom_friction,
    tau_x_2d=jnp.zeros((nx,ny)), tau_y_2d=jnp.zeros((nx,ny)),
    Q_heat_2d=jnp.zeros((nx,ny)), H_sw=H_sw,
    dz_norm=jnp.array(grid.dz).reshape(1,1,-1)/H_sw, dt=60.0,
    T_atm_3d=jnp.zeros((nx,ny,1)), lambda_bulk=0.0,
    sponge_rate=jnp.zeros((nx,ny,1)), sponge_rate_2d=jnp.zeros((nx,ny)),
    T_clim_3d=jnp.zeros((nx,ny,nz)), S_clim_3d=jnp.zeros((nx,ny,nz)),
    polar_cap_rows=0)
# rebuild a FDPhysParams-like for term eval — but the term fns take FDPhysParams.
# Easier: just use the params the solver built by re-calling make_solver_global
# internals. Instead, evaluate operators with the FDParams `p` (make_fd_params).
state = init_state(T_init, S_init)

# step to ~100 (pre-divergence; max|u| ~1)
for k in range(1, 101):
    state = step(state)
print(f"step 100: max|u|={float(jnp.max(jnp.abs(state.u))):.3e} "
      f"max_T={float(jnp.max(jnp.abs(state.T))):.3f}")

lat = np.array(grid.lat); lon = np.array(grid.lon)
iy = int(np.argmin(np.abs(lat - 79.5)))
# find the largest-|u| column at this lat
u_row = np.abs(np.array(state.u[:, iy, :]))
ix, iz = np.unravel_index(np.argmax(u_row), u_row.shape)
print(f"max|u| point: lat={lat[iy]:.1f} lon={lon[ix]:.1f} iz={iz} "
      f"depth={grid.depth[ix,iy]:.0f} u={float(state.u[ix,iy,iz]):.3e}")

# Term budget for du/dt at this point
st = state
w = _compute_vertical_velocity(st, p)
adv_u, _ = _advection_flux_form(st.u, st.v, w, p)
pgf_x, _ = _compute_pressure_gradient(st, p)
diff_h = phys.nu_h * _laplacian_h(st.u, p)
diff_v = phys.nu_v * _d2_dz2(st.u, p)
cor = p.f[ix, iy] * st.v[ix, iy, iz]

print(f"\nMomentum tendency terms at (lat={lat[iy]:.1f}, lon={lon[ix]:.1f}, iz={iz}):")
print(f"  advection   = {float(adv_u[ix,iy,iz]):+.3e}")
print(f"  Coriolis    = {float(cor):+.3e}")
print(f"  PGF (x)     = {float(pgf_x[ix,iy,iz]):+.3e}")
print(f"  diff_h (nu_h*lap) = {float(diff_h[ix,iy,iz]):+.3e}")
print(f"  diff_v      = {float(diff_v[ix,iy,iz]):+.3e}")
print(f"  u itself    = {float(st.u[ix,iy,iz]):+.3e}")
print(f"  f           = {float(p.f[ix,iy]):+.3e}")

# Also dump the Laplacian sub-terms at this point
u3 = st.u
d2dx2 = (jnp.roll(u3, -1, axis=0) - 2.0*u3 + jnp.roll(u3, 1, axis=0)) * p.inv_dx2
print(f"\n  Laplacian sub-terms at point:")
print(f"    d2u/dx2   = {float(d2dx2[ix,iy,iz]):+.3e}   (inv_dx2={float(p.inv_dx2[ix,iy,0]):.3e})")
print(f"    nu_h*d2dx2= {float(phys.nu_h*d2dx2[ix,iy,iz]):+.3e}")
# the full laplacian value
lap = _laplacian_h(st.u, p)
print(f"    full lap  = {float(lap[ix,iy,iz]):+.3e}")

# Check the NEIGHBORS of this point (is there a huge gradient next door?)
print(f"\n  u in 3x3 (iz={iz}) around point (lon,lat):")
for dj in [-1,0,1]:
    row=[]
    for di in [-1,0,1]:
        j=iy+dj; i=(ix+di)%len(lon)
        row.append(f"{float(st.u[i,j,iz]):+.2e}")
    print(f"    {' '.join(row)}")

# PGF decomposition: barotropic (eta gradient) vs baroclinic (density integral)
from jax_solver_global import _compute_bt_rho_pgf, _compute_hydrostatic_pressure, _density_anomaly
pressure = _compute_hydrostatic_pressure(st, p)
# d(pressure)/dx at the point = PGF_x
dpgf_x = _d_dx(pressure, p)
bt_pgf_x = -G_EARTH * _d_dx(st.eta[:, :, None], p)[:, :, 0]
rho_pgf_x, _ = _compute_bt_rho_pgf(st, p)
print(f"\n  PGF decomposition at point:")
print(f"    total PGF_x       = {float(pgf_x[ix,iy,iz]):+.3e}")
print(f"    -g*d(eta)/dx (bt) = {float(bt_pgf_x[ix,iy]):+.3e}")
print(f"    rho PGF_x (bt)    = {float(rho_pgf_x[ix,iy]):+.3e}")
print(f"    eta at point      = {float(st.eta[ix,iy]):+.3e}")
print(f"    eta 3x3 around pt:")
for dj in [-1,0,1]:
    row=[]
    for di in [-1,0,1]:
        j=iy+dj; i=(ix+di)%len(lon)
        row.append(f"{float(st.eta[i,j]):+.2e}")
    print(f"      {' '.join(row)}")
print(f"    T at point (iz={iz}) = {float(st.T[ix,iy,iz]):.3f}")
print(f"    wet_mask_z col     = {np.array(p.wet_mask_z[ix,iy,:]).astype(int)}")
print(f"    depth = {float(grid.depth[ix,iy]):.0f}m")
