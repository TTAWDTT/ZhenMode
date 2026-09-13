"""Per-level TRUE one-step dT/dstep vs terms_fn at the checkpoint.

Isolates which physical process warms the deep: the real step() tendency is
ground truth; terms_fn is the named decomposition. Any level where they
disagree localizes physics missing from terms_fn.
"""
import sys
from dataclasses import replace
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from jax_solver_global import make_solver_global, JaxStateG
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT

RHO_0, C_P, SPY, DT = 1025.0, 3992.0, 8760.0, 3600.0
Z = np.array([0, 5, 15, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 4000], float)
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
u0 = jnp.asarray(np.asarray(d["u"], np.float64)); v0 = jnp.asarray(np.asarray(d["v"], np.float64))
T0 = jnp.asarray(np.asarray(d["T"], np.float64)); S0 = jnp.asarray(np.asarray(d["S"], np.float64))
e0 = jnp.asarray(np.asarray(d["eta"], np.float64))
init = np.load("init_fields_g360x120.npz")
T_atm = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)
step, init_fn, diag, p, terms_fn = make_solver_global(
    g, phys, DT, T_atm=jnp.asarray(T_atm), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)
print("adv_nsub =", int(p.adv_nsub), " conv_nsub =", int(p.conv_nsub),
      " n_subcyc =", int(p.n_subcyc), " use_scan =", bool(p.use_scan))

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = RHO_0 * C_P / 1e21

st = JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)
s1 = step(st)

# true per-level tendency
dT_true = np.asarray(s1.T) - np.asarray(T0)
tn = np.asarray(terms_fn(st))
# terms_fn is a rate (per second); integrate over dt
dT_terms = tn * DT

names = ["adv_T", "diff_h_T", "diff_v_T", "conv_T", "gm_T", "redi_T"]
# surface term as the solver writes it
heat_factor = 1.0 / (RHO_0 * C_P * float(p.dz_surface))
bulk_T = (float(p.lambda_bulk) * (T_atm[:, :, None] - np.asarray(T0)[:, :, 0:1])
          * heat_factor * np.asarray(p.surface_mask)) * DT
qh_T = (np.asarray(p.Q_heat_2d)[:, :, None] * heat_factor
        * np.asarray(p.surface_mask)) * DT

print("\nper-level OHC tendency, ZJ/yr  (positive = warming)")
hdr = " k  zc(m)     TRUE   " + "".join("%10s" % n[:9] for n in names) + "%10s" % "bulk" + "%10s" % "SUM"
print(hdr)
print(" %2s %6s  %+10s  " % ("k", "zc", "TRUE") + "".join("%10s" % ("-"*8) for _ in names) + "%10s" % ("-"*8) + "%10s" % ("-"*8))
tot_true = 0.0
for k in range(14):
    v = vol[:, :, k]
    true_k = (dT_true[:, :, k] * v).sum() * ZJ * SPY
    row = [(dT_terms[t][:, :, k] * v).sum() * ZJ * SPY for t in range(6)]
    bk = (bulk_T[:, :, k] * v).sum() * ZJ * SPY
    tot_true += true_k
    print(" %2d %6.0f  %+10.2f  " % (k, -Z[k], true_k) + "".join("%+10.2f" % r for r in row)
          + "%+10.2f" % bk + "%+10.2f" % (sum(row) + bk))
print(" column TRUE = %+.2f ZJ/yr ; terms SUM = %+.2f ZJ/yr"
      % (tot_true, sum((dT_terms[t] * vol).sum() * ZJ * SPY for t in range(6))
         + (bulk_T * vol).sum() * ZJ * SPY))

print("\ncolumn totals (ZJ/yr):")
for t, n in enumerate(names):
    print("   %-10s %+12.3f" % (n, (dT_terms[t] * vol).sum() * ZJ * SPY))
print("   %-10s %+12.3f" % ("bulk", (bulk_T * vol).sum() * ZJ * SPY))
print("   %-10s %+12.3f" % ("Q_heat", (qh_T * vol).sum() * ZJ * SPY))
print("   %-10s %+12.3f" % ("TRUE", tot_true))
