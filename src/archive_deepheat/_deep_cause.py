"""Why does the deep keep warming? Test the convective instability source.

1. Density structure: decompose rho' into its T-part and S-part and find which
   interfaces are unstable, by depth.
2. Zero-velocity test: does the deep warming survive when advection is switched
   off (u=v=0 held through the whole step)?
3. Zero-conv test: does it survive when kappa_conv -> 0 ?
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
T0 = np.asarray(d["T"], np.float64); S0 = np.asarray(d["S"], np.float64)
u0 = np.asarray(d["u"], np.float64); v0 = np.asarray(d["v"], np.float64)
e0 = np.asarray(d["eta"], np.float64)
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])

phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)

step, _, _, p, terms_fn = JS.make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = 1025.0 * 3992.0 / 1e21 * 8760.0
st = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                  S=jnp.asarray(S0), eta=jnp.asarray(e0))

print("=" * 90)
print("1. DENSITY STRUCTURE / INSTABILITY SOURCE")
print("=" * 90)
rho = np.asarray(JS._density_anomaly(jnp.asarray(T0), jnp.asarray(S0), p))
rho = rho * wet3
wet_if = (wet3[:, :, :-1] > 0.5) & (wet3[:, :, 1:] > 0.5)
drho = rho[:, :, 1:] - rho[:, :, :-1]          # <0 => unstable (dense over light)
unst = (drho < 0) & wet_if
# T part: drho_T = -ALPHA*(T[k+1]-T[k]) ; S part: +BETA*(S[k+1]-S[k])
A_T, B_S = 2.0e-4, 7.6e-4
dT_ = T0[:, :, 1:] - T0[:, :, :-1]
dS_ = S0[:, :, 1:] - S0[:, :, :-1]
dTp = -A_T * dT_
dSp = +B_S * dS_
print("\niface  z_top      n_unst   mean drho     T-part     S-part   driver")
Ztop = np.array([0, 5, 15, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000], float)
for k in range(13):
    m = unst[:, :, k]
    n = int(m.sum())
    if n == 0:
        print("  %2d   %-7.0f  %6d      --" % (k, Ztop[k], n))
        continue
    dm = drho[:, :, k][m].mean()
    tp = dTp[:, :, k][m].mean()
    sp = dSp[:, :, k][m].mean()
    print("  %2d   %-7.0f  %6d   %+9.2e  %+9.2e %+9.2e   %s"
          % (k, Ztop[k], n, dm, tp, sp, "S" if abs(sp) > abs(tp) else "T"))

print("\nfraction of unstable interfaces where |S-part| > |T-part|: %.1f%%"
      % (100.0 * np.mean(np.abs(dSp[unst]) > np.abs(dTp[unst]))))
print("total wet ifaces %d, unstable %d (%.2f%%)"
      % (wet_if.sum(), unst.sum(), 100.0 * unst.sum() / wet_if.sum()))

# instability by column depth
col_unst = unst.sum(axis=2)
surf = wet3[:, :, 0] > 0.5
print("\nunst per-column count (wet surface cols): mean %.2f median %.0f max %d"
      % (col_unst[surf].mean(), np.median(col_unst[surf]), col_unst[surf].max()))
deep_unst = unst[:, :, 8:].sum(axis=2)   # below 200 m
print("columns with >=1 unstable iface below 200 m: %d of %d (%.1f%%)"
      % (int((deep_unst[surf] > 0).sum()), int(surf.sum()),
         100.0 * (deep_unst[surf] > 0).sum() / surf.sum()))

print("\n" + "=" * 90)
print("2/3. ABLATION: deep warming with advection off / convection off")
print("=" * 90)

def prof(tag, s):
    s1 = step(s)
    jax.block_until_ready(s1)
    dT = np.asarray(s1.T) - T0
    rows = np.array([(dT[:, :, k] * vol[:, :, k]).sum() * ZJ for k in range(14)])
    print("%-22s" % tag + " ".join("%+7.1f" % r for r in rows) + "  | col %+8.2f" % rows.sum())
    return rows

print("k                        " + " ".join("%7d" % k for k in range(14)))
base = prof("baseline", st)

zero_uv = JS.JaxStateG(u=jnp.zeros_like(st.u), v=jnp.zeros_like(st.v),
                       T=st.T, S=st.S, eta=st.eta)
no_adv = prof("u=v=0 (no advection)", zero_uv)

zero_eta = JS.JaxStateG(u=st.u, v=st.v, T=st.T, S=st.S, eta=jnp.zeros_like(st.eta))
no_eta = prof("eta=0 (no free-surf)", zero_eta)

print("\ndelta no-advection : " + " ".join("%+7.1f" % x for x in (no_adv - base)))
print("delta no-eta       : " + " ".join("%+7.1f" % x for x in (no_eta - base)))

# conv off requires rebuilding the solver
step2, _, _, _, _ = JS.make_solver_global(
    g, replace(phys, kappa_conv=0.0), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)
s1 = step2(st); jax.block_until_ready(s1)
dT = np.asarray(s1.T) - T0
rows = np.array([(dT[:, :, k] * vol[:, :, k]).sum() * ZJ for k in range(14)])
print("%-22s" % "kappa_conv=0" + " ".join("%+7.1f" % r for r in rows) + "  | col %+8.2f" % rows.sum())
print("delta no-conv      : " + " ".join("%+7.1f" % x for x in (rows - base)))
