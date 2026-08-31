import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
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
dS_n = np.asarray(dS)
print('conv-included dS/dt at (302,70,0):', dS_n[302,70,0]*86400, 'PSU/day')
print('is (302,70) flagged by gate? (my calc said False)')
# what other term gives -84/day there?
wmz = np.asarray(p.wet_mask_z)
def rho_f(T, S): return (-2.0e-4*(T-15.0) + 7.6e-4*(S-35.0)) * wmz
r = rho_f(T, S)
gate = (wmz[:,:,:-1]>0.5) & (wmz[:,:,1:]>0.5)
unst = (r[:,:,:-1] > r[:,:,1:]) & gate
print('unstable ifaces at (302,70):', np.where(unst[302,70])[0])
# if column unflagged, the -84/day comes from adv/diff/gm. adv was -1.55,
# diff_h 0, diff_v -0.22, gm -0.01. Sum=-1.8. Total measured? 
print('total dS/dt at cell *86400:', dS_n[302,70,0]*86400)
# recompute conv explicitly as solver does:
conv_mask = np.any(unst, axis=-1, keepdims=True)
d2S = np.asarray(jsg._d2_dz2(jnp.array(S), p))
conv_S = 0.05 * conv_mask * d2S * wmz
print('conv_S (gated) at cell:', conv_S[302,70,0]*86400)
