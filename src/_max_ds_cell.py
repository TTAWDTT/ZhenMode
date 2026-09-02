import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# (302,70) itself is NOT conv-flagged: total dS/dt = -1.78 PSU/day there.
# But the max |dS/dt| = 85.9 PSU/day at (167,67,0) — conv_S max = 84.5/day
# at (167,67,0). The (167,67) column: is it conv-flagged? rho[0]=-0.000923 >
# rho[1]=-0.001092 -> iface 0 unstable (REAL surface instability, both wet).
# This is the West Pacific WARM POOL: surface T=19.1C?? That's cold for the
# warm pool... T dropped from 29.6 to 19.1! The dipole T ran away DOWN.
# So: the surface T dipole cools the surface below the layer below
# (T0=19.1 < T1=18.96? no, 19.1 > 18.96 -> stable in T. But S0=34.87 >
# S1=34.61 -> surface saltier than below -> rho[0] > rho[1] -> UNSTABLE).
# Then conv triggers: kappa_conv * d2T/dz2 + d2S/dz2 over whole column.
# d2S at (167,67,0) = +0.49 -> conv_S=+6.4/day at surface (raises S0).
# But the measured max dS/dt = 85.9 at (167,67,0)?? My gated conv_S at
# (167,67) = +6.37/day. But the solver total at (167,67,0):
import numpy as np, sys
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from dataclasses import replace
from config import DEFAULT_CONFIG, GlobalGridConfig
from grid import make_global_grid
import jax_solver_global as jsg
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
base = jsg.make_fd_params(grid)
nx, ny, nz = grid.nx, grid.ny, grid.nz
arr = np.load('../results/global_gm365d_3d/snap_00022.npy')
T, u, v, S = arr
H_sw = float(jnp.sum(jnp.array(grid.dz)))
phys_fields = dict(
    nu_h=5e6, nu_v=1e-4, kappa_h=100.0, kappa_v=1e-5, kappa_conv=0.05,
    nu_bi=2e14, kappa_bi=2e14, T_ref=15.0, S_ref=35.0,
    eos_type='linear', r_bot=1e-4, cd=2.5e-3, bottom_friction='quadratic',
    tau_x_2d=jnp.zeros((nx, ny)), tau_y_2d=jnp.zeros((nx, ny)),
    Q_heat_2d=jnp.zeros((nx, ny)),
    H_sw=H_sw,
    dz_norm=(jnp.array(grid.dz).reshape(1, 1, -1) / H_sw),
    dt=120.0, T_atm_3d=jnp.zeros((nx, ny, 1)), lambda_bulk=0.0,
    sponge_rate=0.0, sponge_rate_2d=jnp.zeros((nx, ny)),
    T_clim_3d=jnp.array(T), S_clim_3d=jnp.array(S),
    polar_cap_rows=2, polar_cap_taper=3,
    kappa_gm=1000.0, gm_slope_max=0.01, kappa_redi=0.0)
phys_fields['dealias_lon_mask'] = jnp.ones((nx, 1, 1))
merged = dict(base._asdict()); merged.update(phys_fields)
merged = {k: merged[k] for k in jsg.FDPhysParams._fields if k in merged}
p = jsg.FDPhysParams(**merged)
state = jsg.JaxStateG(u=jnp.array(u), v=jnp.array(v), T=jnp.array(T),
                      S=jnp.array(S), eta=jnp.zeros((nx, ny)))
dT, dS = jsg._compute_tracer_tendency(state, p)
dS_n = np.asarray(dS); dT_n = np.asarray(dT)
print('dS/dt at (167,67,0):', dS_n[167,67,0]*86400, '/day; dT:', dT_n[167,67,0]*86400, 'K/day')
# terms at (167,67):
w = np.asarray(jsg._compute_vertical_velocity(state, p))
adv = np.asarray(jsg._advection_scalar(jnp.array(S), jnp.array(u), jnp.array(v), jnp.array(w), p))
Sx, Sy = jsg._isopycnal_slope(state, p)
gm = np.asarray(jsg._redi_skew_flux_tendency(jnp.array(S), Sx, Sy, p, kappa=1000.0))
wmz = np.asarray(p.wet_mask_z)
def rho_f(T, S): return (-2.0e-4*(T-15.0) + 7.6e-4*(S-35.0)) * wmz
r = rho_f(T, S)
gate = (wmz[:,:,:-1]>0.5) & (wmz[:,:,1:]>0.5)
unst = (r[:,:,:-1] > r[:,:,1:]) & gate
cm = np.any(unst, axis=-1, keepdims=True)
d2S = np.asarray(jsg._d2_dz2(jnp.array(S), p))
conv = 0.05*cm*d2S*wmz
print('adv:', adv[167,67,0]*86400, 'conv:', conv[167,67,0]*86400, 'gm:', gm[167,67,0]*86400)
print('column conv-flagged:', cm[167,67,0])
# where does the dipole actually GROW? compare d125 vs d110 S dipole cells
