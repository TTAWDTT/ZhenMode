"""Settle the rate question empirically.

Step the ten-year checkpoint 200 times and record, every 50 steps, the
volume-mean T per level and total OHC.  A one-step dT of -0.0012 K at the
surface layer implies -0.24 K over 200 steps; if instead the surface barely
moves, the one-step dT I have been quoting is not what it appears.

Also directly measure the surface heat-flux terms (bulk + penetrating SW)
so the -65.9 ZJ/yr surface tendency can be attributed.
"""
import sys, time
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
DZN = np.array([5, 7.5, 12.5, 17.5, 22.5, 25, 37.5, 50, 75, 150, 350, 750, 1500, 2000], float)
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)
step, _, _, p, terms = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
CP = 1025.0 * 3992.0
ZJYR = CP / 1e21 * 3.1536e7

d = np.load("results/ckpt_tenyr_ms_gm.npz")
st = JS.JaxStateG(u=jnp.asarray(d["u"]), v=jnp.asarray(d["v"]),
                  T=jnp.asarray(d["T"]), S=jnp.asarray(d["S"]),
                  eta=jnp.asarray(d["eta"]))

def prof(st_):
    T = np.asarray(st_.T)
    return np.array([(T[:, :, k] * vol[:, :, k]).sum() / vol[:, :, k].sum()
                     for k in range(14)]), float((T * vol).sum() * CP / 1e21)

p0, o0 = prof(st)
print("step    OHC/ZJ      dOHC(ZJ/yr)   T0       T1       T5       T10      T12      T13")
print("   0  %+.2f  %10s   %.4f  %.4f  %.4f  %.4f  %.4f  %.4f"
      % (o0, "-", p0[0], p0[1], p0[5], p0[10], p0[12], p0[13]))
t0 = time.time()
s = st
for n in range(1, 201):
    s = step(s)
    if n % 50 == 0:
        jax.block_until_ready(s)
        pn, on = prof(s)
        yrs = n * 3600.0 / 3.1536e7
        print("%4d  %+.2f  %+10.2f   %.4f  %.4f  %.4f  %.4f  %.4f  %.4f"
              % (n, on, (on - o0) / yrs, pn[0], pn[1], pn[5], pn[10], pn[12], pn[13]))
print("wall %.1f s for 200 steps (%.3f s/step)" % (time.time() - t0, (time.time() - t0) / 200))

t = np.asarray(terms(s)); jax.block_until_ready(t)
NAMES = ["adv_T", "diff_h_T", "diff_v_T", "conv_T", "gm_T", "redi_T"]
print("\nterms_fn at the 200-step state (ZJ/yr per level, x dt):")
print("k                        " + " ".join("%8d" % k for k in range(14)))
tot = 0
for i, nm in enumerate(NAMES):
    row = np.array([(t[i][:, :, k] * vol[:, :, k]).sum() for k in range(14)]) * ZJYR * 3600.0
    tot = tot + row
    print("%-22s " % nm + " ".join("%+8.1f" % x for x in row) + " | %+8.2f" % row.sum())
print("SUM of 6 terms         " + " ".join("%+8.1f" % x for x in tot) + " | %+8.2f" % tot.sum())
s2 = step(s)
dsurf = np.array([((np.asarray(s2.T)[:, :, k] - np.asarray(s.T)[:, :, k]) * vol[:, :, k]).sum()
                  for k in range(14)]) * CP / 1e21 * 8760.0
print("REAL step() diff       " + " ".join("%+8.1f" % x for x in dsurf) + " | %+8.2f" % dsurf.sum())
print("(terms exclude bulk_T and heat_T; the difference is the surface flux)")
