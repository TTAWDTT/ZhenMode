"""DECISIVE ARM TEST: 200 steps each, per-level T drift.

The tenyr checkpoint warms the deep at +155 ZJ/yr (k>=12) and cools the upper
ocean.  Which mechanism SUSTAINS it (not just shifts the first step)?

Arms (all from ckpt_tenyr_ms_gm, 200 steps = 200 h):
  baseline            production config
  gm=redi=0           kill the isopycnal skew flux
  kappa_v=0           kill the explicit vertical diffusion
  topzero             Fz_top = 0  (exact advective conservation)
  monosplit           mode_split=False
  redi=0              only the doubled GM half
  slope002            tighter DM95 taper

Report per-level volume-mean dT over 200 steps, plus deep/upper aggregates.
"""
import sys, re, types, time
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

# ---- build a top-face-zeroed copy of the module -------------------------
SRC = open("src/jax_solver_global.py", encoding="utf-8").read()
OLD = "Fz_top = Fz_in[:, :, :1] * T[..., :1]"
assert SRC.count(OLD) == 1, SRC.count(OLD)
TOPOZERO_SRC = SRC.replace(
    OLD, "Fz_top = Fz_in[:, :, :1] * T[..., :1] * 0.0  # ARM: topzero")
_ns = {"__name__": "js_topzero_arm", "__file__": "src/jax_solver_global.py"}
exec(compile(TOPOZERO_SRC, "js_topzero_arm.py", "exec"), _ns)
JS_TOPZERO = types.SimpleNamespace(
    **{k: v for k, v in _ns.items() if not k.startswith("__")})
print("topzero module built OK")

d = np.load("results/ckpt_tenyr_ms_gm.npz")
ST0 = JS.JaxStateG(u=jnp.asarray(d["u"]), v=jnp.asarray(d["v"]),
                   T=jnp.asarray(d["T"]), S=jnp.asarray(d["S"]),
                   eta=jnp.asarray(d["eta"]))
T0 = np.asarray(d["T"], float)

# rebuild the true wet mask from a params object
_, _, _, p_ref, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)
wet3 = np.asarray(p_ref.wet_mask_z, float)
DZN = np.asarray(p_ref.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
CP = 1025.0 * 3992.0

def levels(st_):
    T = np.asarray(st_.T, float)
    return np.array([(T[:, :, k] * vol[:, :, k]).sum() / vol[:, :, k].sum()
                     for k in range(14)])

def run(tag, phys_kw, mod=JS, nstep=200, **mk_kw):
    cfg = dict(BASE); cfg.update(phys_kw)
    kw = dict(mode_split=True, dt_bt=300.0)
    kw.update(mk_kw)
    step, _, _, _, _ = mod.make_solver_global(
        g, replace(PhysicsConfig(), **cfg), 3600.0, T_atm=jnp.asarray(T_atm_np),
        lambda_bulk=BULK_LAMBDA_DEFAULT, return_params=True, **kw)
    t0 = time.time()
    s = ST0
    for _ in range(nstep):
        s = step(s)
    jax.block_until_ready(s)
    L = levels(s) - levels(ST0)
    wall = time.time() - t0
    o = ((np.asarray(s.T) - T0) * vol).sum() * CP / 1e21
    print("%-12s " % tag + " ".join("%+.4f" % x for x in L) + " | dOHC %+8.2f ZJ" % o)
    return L, o, wall

print("\nper-level dT over 200 steps (K):")
print("%-12s " % "arm" + " ".join("%+7d" % k for k in range(14)) + " | aggregate")
Lb, ob, wb = run("baseline", {}, nstep=200)
print("(%.0f s)" % wb)
arms = [
    ("gm=redi=0", dict(kappa_gm=0.0, kappa_redi=0.0), {}, JS),
    ("kappa_v=0", dict(kappa_v=0.0), {}, JS),
    ("redi=0", dict(kappa_redi=0.0), {}, JS),
    ("slope002", dict(gm_slope_max=0.002), {}, JS),
    ("topzero", {}, {}, JS_TOPZERO),
    ("monosplit", {}, dict(mode_split=False), JS),
]
res = {}
for tag, pk, mkw, mod in arms:
    try:
        res[tag] = run(tag, pk, mod=mod, **mkw)
    except Exception as e:
        print("%-12s FAILED: %s: %s" % (tag, type(e).__name__, str(e)[:120]))

print("\ndelta vs baseline (K over 200 steps):")
for tag, (L, o, w) in res.items():
    print("%-12s " % tag + " ".join("%+.4f" % x for x in (L - Lb)) + " | dOHC %+8.2f" % (o - ob))

print("\nupper (k0-7) / deep (k11-13) aggregates:")
def agg(L):
    return L[:8].mean(), L[11:].mean()
print("%-12s  upper dT   deep dT" % "arm")
ub, db = agg(Lb)
print("%-12s  %+.4f   %+.4f" % ("baseline", ub, db))
for tag, (L, o, w) in res.items():
    u, dd = agg(L)
    print("%-12s  %+.4f   %+.4f" % (tag, u, dd))
