"""Is the deep-warming signal real, or an artifact of my masked-out AREA?

_vert_test.py / _kappa_eff.py computed per-level column integrals with
    vol = AREA[:, :, None] * dz_node          <-- NO wet_mask
so dry/ghost cells (T = 15.0 sentinel) were included at every level where
the level is above the seafloor.  Recompute the production one-step
per-level tendency with the CORRECT weight vol = wet3*AREA*dz_node, and also
with the bare AREA weight, side by side.
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from dataclasses import replace
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT
import jax_solver_global as JS

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)
step, _, _, p, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol_true = wet3 * AREA[:, :, None] * DZN[None, None, :]   # correct
vol_bare = AREA[:, :, None] * DZN[None, None, :]          # what I used before
print("vol_true (1e18) %.6f   vol_bare (1e18) %.6f   ratio %.4f"
      % (vol_true.sum() / 1e18, vol_bare.sum() / 1e18, vol_bare.sum() / vol_true.sum()))

d = np.load("results/ckpt_tenyr_ms_gm.npz")
T0 = np.asarray(d["T"], np.float64)
st = JS.JaxStateG(u=jnp.asarray(d["u"]), v=jnp.asarray(d["v"]),
                  T=jnp.asarray(T0), S=jnp.asarray(d["S"]),
                  eta=jnp.asarray(d["eta"]))
s1 = step(st); jax.block_until_ready(s1)
dT = np.asarray(s1.T) - T0
ZJ = 1025.0 * 3992.0 / 1e21 * 8760.0

def cs(f, vol):
    f = np.asarray(f)
    return np.array([(f[:, :, k] * vol[:, :, k]).sum() * ZJ for k in range(14)])

a = cs(dT, vol_true); b = cs(dT, vol_bare)
print("\nk                        " + " ".join("%8d" % k for k in range(14)))
print("WET-MASKED (correct)     " + " ".join("%+8.1f" % x for x in a) + " | %+8.2f" % a.sum())
print("BARE AREA (my old bug)   " + " ".join("%+8.1f" % x for x in b) + " | %+8.2f" % b.sum())
print("difference               " + " ".join("%+8.1f" % x for x in (b - a)) + " | %+8.2f" % (b - a).sum())

print("\nghost-cell T values by level (should be all 15.0):")
for k in range(14):
    m = wet3[:, :, k] < 0.5
    if m.sum():
        vals = T0[:, :, k][m]
        print("  k=%2d  n_dry=%6d  min %.3f max %.3f  frac_exactly_15 %.4f"
              % (k, m.sum(), vals.min(), vals.max(), np.mean(vals == 15.0)))

print("\nraw (unweighted) dT per level -- the ghost contamination directly:")
print("  k   raw dT mean     wet-only dT mean   ghost dT mean")
for k in range(14):
    m = wet3[:, :, k] > 0.5
    print("  %2d  %+.6f       %+.6f          %s" % (
        k, dT[:, :, k].mean(), dT[:, :, k][m].mean(),
        ("%+.6f" % dT[:, :, k][~m].mean()) if (~m).sum() else "n/a"))
