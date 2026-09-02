"""Quantify the barotropic rho-PGF defect at the d75 state.

OLD (current code): F = -(1/rho0) * grad( p_bc_avg ),  p_bc_avg = (1/H_sw)*SUM_k 0.5*(p_k+p_k+1)*h_k
  - includes the ghost layer (p_bc constant below bottom -> weighted (H_sw-H)/H_sw)
  - includes the ghost-dp leak: rho_avg ghost = 0.5*rho_bottom (rho masked -> 0)
NEW (proposed):   F = (1/H_sw) * SUM_k pgf3d_k * h_k * wgate_k  (transport-weighted
  mean of the face-gated 3D baroclinic PGF — what the N-step momentum actually feels)
"""
import numpy as np, dataclasses
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from grid import make_global_grid
from config import GlobalGridConfig
import jax_solver_global as jsg

gcfg = dataclasses.replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, r'C:\Users\zhen.luo\Desktop\ETOPO_2022_v1_r3600x1800_surface.nc',
                        smooth_passes=30, min_depth=100.0)
p = jsg.make_fd_params(grid)
d = np.load('results/global_diag90_blob.npz', allow_pickle=True)
T_init = np.array(d['T_init']); S_init = np.array(d['S_init'])
wet3 = np.array(p.wet_mask_z)

sn = np.load('results/global_diag90_blob_3d/snap_00030.npy')
T3, u3, v3 = sn[0], sn[1], sn[2]
S3 = S_init * wet3 + (1 - wet3) * 35.0
Tj = jnp.array(T3); Sj = jnp.array(S3)

G = jsg.G_EARTH; RHO_0 = jsg.RHO_0
dz_norm = (np.array(grid.dz).reshape(1, 1, -1) / float(np.sum(np.array(grid.dz))))
rho_prime = RHO_0 * (-jsg.ALPHA_T * (Tj - 15.0) + jsg.BETA_S * (Sj - 35.0))
rho_prime = rho_prime * p.wet_mask_z
rho_avg = 0.5 * (rho_prime[..., :-1] + rho_prime[..., 1:])
dp = G * rho_avg * p.dz_3d
p_bc = jnp.zeros_like(Tj)
p_bc = p_bc.at[..., 1:].set(jnp.cumsum(dp, axis=-1))

# ---- OLD ----
p_bc_avg = jnp.sum(0.5 * (p_bc[..., :-1] + p_bc[..., 1:]) * jnp.array(dz_norm), axis=-1)
gx_old, gy_old = jsg._gradient_conservative(p_bc_avg, p)
F_old_x = -np.array(gx_old) / RHO_0
F_old_y = -np.array(gy_old) / RHO_0

# ---- NEW ----
# NEW = (1/H_sw) * depth-integral over the WET column of the 3D baroclinic
# PGF, i.e. layer-centered PGF values gated by wet interfaces. This is exactly
# the depth-average of what the 3D momentum equation feels — no ghost water,
# no ghost-dp leak, no below-bottom constant term.
gx3, gy3 = jsg._gradient_conservative_3d(p_bc, p)
h13 = jnp.array(grid.dz).reshape(1, 1, -1)                      # (1,1,13)
wet_iface = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
pgf3d_x = 0.5 * (gx3[..., :-1] + gx3[..., 1:])                  # layer-centered
pgf3d_y = 0.5 * (gy3[..., :-1] + gy3[..., 1:])
F_new_x = -np.array(jnp.sum(pgf3d_x * wet_iface * h13, axis=-1)) / (RHO_0 * float(np.sum(np.array(grid.dz))))
F_new_y = -np.array(jnp.sum(pgf3d_y * wet_iface * h13, axis=-1)) / (RHO_0 * float(np.sum(np.array(grid.dz))))

print('|F_old| global: mean=%.2e max=%.2e m/s2' % (
    np.sqrt(F_old_x**2+F_old_y**2).mean(), np.sqrt(F_old_x**2+F_old_y**2).max()))
print('|F_new| global: mean=%.2e max=%.2e m/s2' % (
    np.sqrt(F_new_x**2+F_new_y**2).mean(), np.sqrt(F_new_x**2+F_new_y**2).max()))
print('|F_new - F_old| global: mean=%.2e max=%.2e' % (
    np.sqrt((F_new_x-F_old_x)**2+(F_new_y-F_old_y)**2).mean(),
    np.sqrt((F_new_x-F_old_x)**2+(F_new_y-F_old_y)**2).max()))
blob = (311, 66)
print()
print('at blob (311,66): F_old=(%.2e,%.2e) F_new=(%.2e,%.2e)' % (
    F_old_x[blob], F_old_y[blob], F_new_x[blob], F_new_y[blob]))
print('F_old vs F_new along i=305..315, j=66:')
for i in range(305, 316):
    print('  i=%d: F_old=(%+.2e,%+.2e)  F_new=(%+.2e,%+.2e)' % (
        i, F_old_x[i,66], F_old_y[i,66], F_new_x[i,66], F_new_y[i,66]))
# Equilibrium eta implied by each: eta_eq = -p_bc_avg/(g*rho0) for OLD;
# for NEW the equilibrium is the transport-mean PGF balance — compute the
# implied eta_eq_new = (1/(g*H)) * int_z^0 ... just report p_bc_avg vs the
# local-depth average
h = np.abs(np.diff(np.array(grid.z)))
H_local = np.sum(h[None,None,:] * np.array(p.wet_mask_z[...,1:]), axis=2)
p_bc_avg_local = np.sum(np.array(0.5*(p_bc[...,:-1]+p_bc[...,1:])) * h[None,None,:] *
                        np.array(p.wet_mask_z[...,1:]), axis=2) / np.maximum(H_local, 1.0)
print()
print('implied equilibrium eta (m): eta_eq = -p_bc_avg/(g*rho0)')
g = 9.8; rho0 = 1027.0
print('i, H(m), eta_eq_OLD, eta_eq_LOCALDEPTH:')
for i in range(305, 316):
    print('  i=%d H=%5.0f  old=%+7.3f  localdepth=%+7.3f' % (
        i, H_local[i,66], -p_bc_avg[i,66]/(g*rho0), -p_bc_avg_local[i,66]/(g*rho0)))
np.savez('../results/_frho_old_new_d75.npz', F_old_x=F_old_x, F_old_y=F_old_y,
         F_new_x=F_new_x, F_new_y=F_new_y, p_bc_avg=np.array(p_bc_avg),
         p_bc_avg_local=p_bc_avg_local, H_local=H_local)
