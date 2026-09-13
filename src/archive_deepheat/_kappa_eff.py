"""How much diapycnal diffusivity is the GM/Redi skew flux injecting?

The skew flux vertical component is
    Fz_i = -k*( S_x*dC_dx + S_y*dC_dy + S2_eff*dC_dz_iface )
whose last term acts exactly like a vertical diffusivity k*S2_eff.  Measure
that against kappa_v (=1e-5), and against the standard isopycnal bound.

Also report the slope distribution and how much of the ocean sits at the DM95
taper cap, plus the per-level OHC response to kappa_gm/redi and gm_slope_max.
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
d = np.load("results/ckpt_tenyr_ms_gm.npz")
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)

def mk(phys):
    return JS.make_solver_global(
        g, phys, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
        mode_split=True, dt_bt=300.0, return_params=True)

_, _, _, p, _ = mk(replace(PhysicsConfig(), **BASE))
wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = 1025.0 * 3992.0 / 1e21 * 8760.0
T0 = np.asarray(d["T"], np.float64); S0 = np.asarray(d["S"], np.float64)
st = JS.JaxStateG(u=jnp.asarray(d["u"], np.float64), v=jnp.asarray(d["v"], np.float64),
                  T=jnp.asarray(T0), S=jnp.asarray(S0),
                  eta=jnp.asarray(d["eta"], np.float64))
def cs(f):
    f = np.asarray(f)
    return np.array([(f[:, :, k] * vol[:, :, k]).sum() * ZJ for k in range(14)])

print("=" * 100)
print("A. EFFECTIVE DIAPYCNAL DIFFUSIVITY FROM THE SKEW FLUX  (kappa_v = 1e-5)")
print("=" * 100)
Sx, Sy = JS._isopycnal_slope(st, p)
Sx = np.asarray(Sx); Sy = np.asarray(Sy)
Smag = np.sqrt(Sx ** 2 + Sy ** 2)
zmid = 0.5 * (DZN[:-1] + DZN[1:])
ZTOP = np.array([0, 5, 15, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000], float)
print("  k  z_top   |S|_mean  |S|_p95  frac@cap  k*S^2 x2 (m2/s)   vs kappa_v")
for k in range(13):
    m = wet3[:, :, k] > 0.5
    s = Smag[:, :, k][m]
    keff = 2.0 * 1000.0 * (s ** 2).mean()
    print("  %2d  %-6.0f  %.2e  %.2e   %6.1f%%     %.3e          %8.0fx"
          % (k, ZTOP[k], s.mean(), np.percentile(s, 95),
             100.0 * np.mean(s > 0.99 * 0.005), keff, keff / 1e-5))

print("\n" + "=" * 100)
print("B. MEASURED K_eff = -Fz_redi / dT_dz at each interface")
print("=" * 100)
T = st.T
Ttr = JS._fill_ghost_bottom(T, p)
dCdz = np.asarray((Ttr[..., 1:] - Ttr[..., :-1]) / p.dz_iface)
# full skew flux vertical component from the production operator
k_use = 1000.0
dCdx, dCdy = JS._gradient_conservative_3d(T * p.wet_mask_z, p)
Sxi = 0.5 * (Sx[..., :-1] + Sx[..., 1:]); Syi = 0.5 * (Sy[..., :-1] + Sy[..., 1:])
dCdx_i = 0.5 * (np.asarray(dCdx)[..., :-1] + np.asarray(dCdx)[..., 1:])
dCdy_i = 0.5 * (np.asarray(dCdy)[..., :-1] + np.asarray(dCdy)[..., 1:])
S2i = Sxi ** 2 + Syi ** 2
Fz = -k_use * (Sxi * dCdx_i + Syi * dCdy_i + S2i * dCdz)
wet_if = (wet3[..., :-1] > 0.5) & (wet3[..., 1:] > 0.5)
print("  k  z_top   mean K_eff(S2 term only)   mean K_eff(full Fz)   |dT/dz| mean")
for k in range(13):
    m = wet_if[:, :, k]
    k1 = (2.0 * k_use * S2i[:, :, k])[m].mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        k2 = np.where(np.abs(dCdz[:, :, k]) > 1e-9, -Fz[:, :, k] / dCdz[:, :, k], np.nan)
    print("  %2d  %-6.0f  %12.3e            %12.3e         %8.2e"
          % (k, ZTOP[k], k1, np.nanmean(k2[m]), np.abs(dCdz[:, :, k][m]).mean()))

print("\n" + "=" * 100)
print("C. per-level OHC response: kappa_gm/redi and gm_slope_max")
print("=" * 100)
print("k                        " + " ".join("%7d" % k for k in range(14)))
base_rows = None
cases = [("gm=redi=1000 (prod)", dict(kappa_gm=1000.0, kappa_redi=1000.0)),
         ("gm=redi=500", dict(kappa_gm=500.0, kappa_redi=500.0)),
         ("gm=redi=250", dict(kappa_gm=250.0, kappa_redi=250.0)),
         ("redi=0 (gm only)", dict(kappa_gm=1000.0, kappa_redi=0.0)),
         ("gm=redi=0", dict(kappa_gm=0.0, kappa_redi=0.0)),
         ("slope_max=0.002", dict(gm_slope_max=0.002)),
         ("slope_max=0.010", dict(gm_slope_max=0.010)),
         ("gm=redi=1000,slope.002", dict(gm_slope_max=0.002)),
         ("kappa_v=1e-4", dict(kappa_v=1e-4)),
         ("conv=0", dict(kappa_conv=0.0))]
out = {}
for tag, kw in cases:
    kw2 = dict(BASE); kw2.update(kw)
    step, _, _, _, _ = mk(replace(PhysicsConfig(), **kw2))
    s1 = step(st); jax.block_until_ready(s1)
    r = cs(np.asarray(s1.T) - T0)
    out[tag] = r
    if base_rows is None:
        base_rows = r
    print("%-24s" % tag + " ".join("%+7.1f" % x for x in r) + " | %+8.2f" % r.sum())
print()
for tag in list(out)[1:]:
    print("delta %-18s" % tag + " ".join("%+7.1f" % x for x in (out[tag] - base_rows))
          + " | %+8.2f" % (out[tag].sum() - base_rows.sum()))
