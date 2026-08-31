import numpy as np, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig
from grid import make_global_grid
import jax_solver_global as jsg

from config import GlobalGridConfig
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
import config as cfgmod
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file,
                        smooth_passes=30, min_depth=100.0)
base = jsg.make_fd_params(grid)
nx, ny, nz = grid.nx, grid.ny, grid.nz

# load d110 state
arr = np.load('../results/global_gm365d_3d/snap_00022.npy')
T, u, v, S = arr

physics = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
                  kappa_gm=1000.0, kappa_redi=0.0, gm_slope_max=0.01)

# Build FDPhysParams directly: base fields + physics fields
import collections
base_fields = base._asdict()
phys_fields = dict(
    nu_h=5e6, nu_v=physics.nu_v, kappa_h=physics.kappa_h,
    kappa_v=physics.kappa_v, kappa_conv=physics.kappa_conv,
    nu_bi=2e14, kappa_bi=2e14, T_ref=physics.T_ref, S_ref=physics.S_ref,
    eos_type='linear', r_bot=physics.r_bot, cd=physics.cd,
    bottom_friction=physics.bottom_friction,
    tau_x_2d=jnp.zeros((nx, ny)), tau_y_2d=jnp.zeros((nx, ny)),
    Q_heat_2d=jnp.zeros((nx, ny)),
    H_sw=float(jnp.sum(jnp.array(grid.dz))),
    dz_norm=(jnp.array(grid.dz).reshape(1, 1, -1) / float(jnp.sum(jnp.array(grid.dz)))),
    dt=120.0, T_atm_3d=jnp.zeros((nx, ny, 1)), lambda_bulk=0.0,
    sponge_rate=0.0, sponge_rate_2d=jnp.zeros((nx, ny)),
    T_clim_3d=jnp.array(T), S_clim_3d=jnp.array(S),
    polar_cap_rows=2, polar_cap_taper=3,
    kappa_gm=1000.0, gm_slope_max=0.01, kappa_redi=0.0)
# dealias_lon_mask may be needed by _dealias_h_fd
all_names = jsg.FDPhysParams._fields
missing = [n for n in all_names if n not in base_fields and n not in phys_fields]
print('missing fields:', missing)
phys_fields['dealias_lon_mask'] = jnp.ones((nx, 1, 1))
merged = dict(base_fields); merged.update(phys_fields)
merged = {k: merged[k] for k in all_names if k in merged}
p = jsg.FDPhysParams(**merged)

print('params built ok')

# ── Compute GM isopycnal slopes + skew tendency at d110 state ──
state = jsg.JaxStateG(u=jnp.array(u), v=jnp.array(v), T=jnp.array(T),
                      S=jnp.array(S), eta=jnp.zeros((nx, ny)))
Sx, Sy = jsg._isopycnal_slope(state, p)
Sx_n = np.asarray(Sx); Sy_n = np.asarray(Sy)
print('Slope S_x at (302,70,0:3):', Sx_n[302,70,:3])
print('Slope S_y at (302,70,0:3):', Sy_n[302,70,:3])
gm_S = _gm = jsg._redi_skew_flux_tendency(
    jsg.JaxStateG(u=jnp.array(u), v=jnp.array(v), T=jnp.array(T), S=jnp.array(S), eta=None).S,
    Sx, Sy, p, kappa=1000.0) if False else jsg._redi_skew_flux_tendency(jnp.array(S), Sx, Sy, p, kappa=1000.0)
gmS_n = np.asarray(gm_S)
print('gm_S tend at (302,70,0:3):', gmS_n[302,70,:3])
print('gm_S tend units: per second; x3600*24 = per day:', gmS_n[302,70,:3]*86400)
# Global max
ik = np.unravel_index(np.argmax(np.abs(gmS_n)), gmS_n.shape)
print(f'gm_S max|dS/dt|={gmS_n[ik]:.4g}/s at i={ik[0]} j={ik[1]} k={ik[2]} -> per day {gmS_n[ik]*86400:.2f} PSU/day')
# rho' profile at cell: is the column unstable?
from jax_solver_global import _density_anomaly
rho = np.asarray(_density_anomaly(jnp.array(T), jnp.array(S), p)) 
print('rho_prime (302,70,0:6):', rho[302,70,:6])
print('drho_dz sign: rho[k+1]-rho[k]:', rho[302,70,1:6]-rho[302,70,0:5])

