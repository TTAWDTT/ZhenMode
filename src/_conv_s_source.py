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

# decompose conv_S at the runaway cell + the max |dS| cell
dT, dS = jsg._compute_tracer_tendency(state, p)
rho_prime = np.asarray(jsg._density_anomaly(jnp.array(T), jnp.array(S), p)) * np.asarray(p.wet_mask_z)
wm = np.asarray(p.wet_mask_z)
wet_iface = (wm[..., :-1] > 0.5) & (wm[..., 1:] > 0.5)
unstable = (rho_prime[..., :-1] > rho_prime[..., 1:]) & wet_iface
conv_mask = np.any(unstable, axis=-1, keepdims=True)
conv_S = np.asarray(0.05 * conv_mask * jsg._d2_dz2(jnp.array(S), p) * np.asarray(p.wet_mask_z))
print('conv_S at (302,70,0):', conv_S[302,70,0]*86400, '/day')
icell = np.unravel_index(np.argmax(np.abs(conv_S)), conv_S.shape)
print(f'conv_S max: {conv_S[icell]*86400:.1f}/day at (i={icell[0]}, j={icell[1]}, k={icell[2]})')
# conv mask at that region
print('conv fraction in box i=160..172, j=64..74, k=0:', conv_mask[160:172, 64:74, 0].mean())
# d2S/dz2 at runaway cell: the S profile has kink at k=0..1
print('S(302,70,0:4):', S[302,70,:4])
# kappa_conv = 0.05 m^2/s; d2z scale at surface = 1/25 m^-2
# conv_S ~ 0.05 * (S2 - 2S1 + S0)/25^2... check the actual d2z_h0_top:
print('d2z_h0_top:', np.asarray(p.d2z_h0_top)[:3])
print('d2z scale = 1/d2z_h0_top^2 -> m^2:', 1.0/np.asarray(p.d2z_h0_top)**2)
# conv_S ~ 0.05 * (S[2]-2S[1]+S[0])/d2z_h0_top^2
d2S = S[302,70,2] - 2*S[302,70,1] + S[302,70,0]
print('d2S raw:', d2S, ' conv_S = 0.05*d2S/25^2*86400 =', 0.05*d2S/625*86400)
