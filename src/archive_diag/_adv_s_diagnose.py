import numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
import jax_solver_global as jsg

gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, DEFAULT_CONFIG.bathymetry_file,
                        smooth_passes=30, min_depth=100.0)
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
dT_n = np.asarray(dT); dS_n = np.asarray(dS)
# where is max |dS/dt|?
ik = np.unravel_index(np.argmax(np.abs(dS_n)), dS_n.shape)
print(f'max|dS/dt|={dS_n[ik]:.3e}/s = {dS_n[ik]*86400:.1f} PSU/day at i={ik[0]} j={ik[1]} k={ik[2]}')
ikT = np.unravel_index(np.argmax(np.abs(dT_n)), dT_n.shape)
print(f'max|dT/dt|={dT_n[ikT]:.3e}/s = {dT_n[ikT]*86400:.2f} K/day at i={ikT[0]} j={ikT[1]} k={ikT[2]}')
# decompose at the runaway cell (302,70)
w = np.asarray(jsg._compute_vertical_velocity(state, p))
adv_S = np.asarray(jsg._advection_scalar(jnp.array(S), jnp.array(u), jnp.array(v), jnp.array(w), p))
diff_h = np.asarray(100.0 * jsg._laplacian_h(jnp.array(S), p))
diff_v = np.asarray(1e-5 * jsg._d2_dz2(jnp.array(S), p))
Sx, Sy = jsg._isopycnal_slope(state, p)
gm_S = np.asarray(jsg._redi_skew_flux_tendency(jnp.array(S), Sx, Sy, p, kappa=1000.0))
for name, val in [('adv_S', adv_S), ('diff_h', diff_h), ('diff_v', diff_v), ('gm_S', gm_S)]:
    v0 = val[302,70,0]*86400
    vmax = np.abs(val).max()*86400
    where = np.unravel_index(np.argmax(np.abs(val)), val.shape)
    print(f'{name:8s}: at cell={v0:+9.2f}/day  max|.|={vmax:9.2f}/day at (i={where[0]},j={where[1]},k={where[2]})')
