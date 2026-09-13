"""Is the node-form _d2_dz2 conservative on the non-uniform vertical grid?

_conv_flux_tendency's docstring states the node-form stencil integrated with
dz_node weights does NOT telescope, which is why convection was rewritten in
interface-flux form. But diff_v (kappa_v*_d2_dz2) and the L half-step
(kappa_v*_d2_dz2 over dt/2, twice) still use the node form.

A conservative vertical operator must satisfy, for ANY tracer column C:
    sum_k d2C_dz2[k] * dz_node[k] == 0        (zero flux at top and bottom)
up to the boundary flux difference. Test that directly, with no solver.

Also test the same for _laplacian_h on the 3-D grid, and for the conv
interface-flux form as a positive control (docstring says it IS exact).
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
import jax_solver_global as JS

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
base = JS.make_fd_params(g)
DZN = np.asarray(base.dz_node).ravel()
print("dz_node =", np.array2string(DZN, precision=2))
print("sum(dz_node) = %.4f" % DZN.sum())

# the raw stencil coefficients, no solver needed
hm = np.asarray(base.d2z_hm).ravel()
hp = np.asarray(base.d2z_hp).ravel()
den = np.asarray(base.d2z_denom).ravel()
h0t = float(base.d2z_h0_top)
h0b = float(base.d2z_h0_bot)
nz = len(DZN)
print()
print("k   dz_node    h_m      h_p     denom      h0")
for k in range(nz):
    print("  %2d %8.2f  " % (k, DZN[k]), end="")
    if 1 <= k <= nz - 2:
        print("%8.2f %8.2f %10.1f" % (hm[k - 1], hp[k - 1], den[k - 1]))
    else:
        print("   --       --         --    %8.2f" % (h0t if k == 0 else h0b))

# Build the matrix form of _d2_dz2 for a single column and test conservation.
def d2_matrix():
    A = np.zeros((nz, nz))
    A[0, 0] = -2.0 / h0t ** 2
    A[0, 1] = 2.0 / h0t ** 2
    for k in range(1, nz - 1):
        A[k, k - 1] = hp[k - 1] / den[k - 1]
        A[k, k + 1] = hm[k - 1] / den[k - 1]
        A[k, k] = -(hm[k - 1] + hp[k - 1]) / den[k - 1]
    A[nz - 1, nz - 1] = -2.0 / h0b ** 2
    A[nz - 1, nz - 2] = 2.0 / h0b ** 2
    return A


A = d2_matrix()
w = DZN
# conservation defect functional: w^T A  (should be ~0 for a conservative op)
defl = w @ A
print()
print("=" * 94)
print("CONSERVATION: w^T A  (must be 0 for a flux-form operator)")
print("=" * 94)
print("  ||w^T A||_inf = %.6e" % np.abs(defl).max())
print("  per-column-node defect:")
for k in range(nz):
    print("    k=%2d  %+14.6e   (relative to 1/dz: %+10.4e)"
          % (k, defl[k], defl[k] * DZN[k]))

# Now the diagnostic that matters: apply to a REAL T column and integrate.
rng = np.random.default_rng(0)
print()
print("=" * 94)
print("APPLIED TEST: sum_k (A C)_k * dz_node_k for several analytic columns")
print("=" * 94)
cases = {
    "constant   C=1": np.ones(nz),
    "linear     C=z": -np.cumsum(np.concatenate([[0], (DZN[:-1] + DZN[1:]) / 2])) * 0 + np.arange(nz, dtype=float),
    "quadratic  C=z^2": None,
    "random     C=rand": rng.standard_normal(nz),
}
zz = np.concatenate([[0.0], np.cumsum((DZN[:-1] + DZN[1:]) / 2.0)])
cases["linear     C=z"] = zz.copy()
cases["quadratic  C=z^2"] = zz ** 2
for nm, C in cases.items():
    r = A @ C
    s = float((r * w).sum())
    print("  %-20s  sum(A C * dz) = %+14.6e    (C range %.1f..%.1f)"
          % (nm, s, C.min(), C.max()))

# interface-flux (conservative) positive control, mirroring _conv_flux_tendency
print()
print("=" * 94)
print("POSITIVE CONTROL: interface-flux form (what conv uses) on the same columns")
print("=" * 94)
zt = np.concatenate([[0.0], np.cumsum((DZN[:-1] + DZN[1:]) / 2.0)])  # node depths
zif = (zt[:-1] + zt[1:]) / 2.0                                        # interfaces
for nm, C in cases.items():
    dCdz = (C[1:] - C[:-1]) / (zt[1:] - zt[:-1])      # nz-1 interface gradients
    F = np.concatenate([[0.0], dCdz, [0.0]])          # zero flux at both ends
    tend = (F[:-1] - F[1:]) / DZN                     # per-node tendency
    print("  %-20s  sum(tend * dz) = %+14.6e" % (nm, float((tend * w).sum())))

# Does _d2_dz2 in the solver match this matrix?
print()
print("=" * 94)
print("SOLVER CHECK: does JS._d2_dz2 reproduce A?")
print("=" * 94)
C3 = jnp.asarray(cases["random     C=rand"]).reshape(1, 1, nz)
got = np.asarray(JS._d2_dz2(C3, base)).ravel()
want = A @ cases["random     C=rand"]
print("  max|JS._d2_dz2 - A C| = %.3e" % np.abs(got - want).max())

# And the ACTUAL global kappa_v contribution, on the real checkpoint, per level
print()
print("=" * 94)
print("REAL FIELD: kappa_v * _d2_dz2 on the 10-yr checkpoint, per level")
print("=" * 94)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
T = np.asarray(d["T"], np.float64)
wet3 = np.asarray(base.wet_mask_z, float)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
RHO_CP = 1025.0 * 3992.0 / 1e21
DT = 3600.0
SEC = 3.1536e7
lap = np.asarray(JS._d2_dz2(jnp.asarray(T), base), float)
percol = (lap * (wet3 * DZN[None, None, :])).sum(axis=-1)      # nonzero => not conservative
print("  per-column sum(d2Tdz2 * dz_node):")
print("    wet columns: min %+ .4e  max %+ .4e  mean %+ .4e"
      % (percol[wet3[:, :, 0] > 0].min(), percol[wet3[:, :, 0] > 0].max(),
         percol[wet3[:, :, 0] > 0].mean()))
print("    max |defect| over ALL columns = %.4e K" % np.abs(percol).max())
dh = np.array([float((lap[:, :, k] * vol[:, :, k]).sum()) * RHO_CP for k in range(nz)])
net = dh.sum() / (DT / SEC)
print("  volume-integrated kappa_v*d2Tdz2 (kappa_v=1e-5):")
for k in range(nz):
    print("    k=%2d  dH %+12.4e ZJ   ->  kappa_v*that = %+10.3f ZJ/yr"
          % (k, dh[k], dh[k] * 1e-5 / (DT / SEC)))
print("    NET              %+10.3f ZJ/yr  (a conservative op must give 0)"
      % (net * 1e-5))
