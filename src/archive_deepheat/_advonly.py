"""Does the advective defect show up in the TIME-STEPPED model?

_advection_scalar(T,...) integrates to +66.58 ZJ/yr at the checkpoint, while
adv(1)=0 exactly. Scale check: 66.58 ZJ/yr over 10 yr = 666 ZJ; the observed
10-yr dOHC is +531 ZJ. If the defect is real it is THE dominant term.

Run the full solver with every closure OFF and the bulk flux OFF, so the ONLY
process is advection. Then dH must be ~0. Anything else is the defect.
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from dataclasses import replace
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from forcing import air_temp_profile
import jax_solver_global as JS

RHO_0, C_P = 1025.0, 3992.0
DT = 3600.0
ZJ = RHO_0 * C_P / 1e21
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
ST0 = JS.JaxStateG(**{k: jnp.asarray(np.asarray(d[k], np.float64))
                      for k in ("u", "v", "T", "S", "eta")})
_, _, _, p, _ = JS.make_solver_global(g, PhysicsConfig(), DT, return_params=True)
wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
H = lambda st: float((np.asarray(st.T, float) * vol).sum() * ZJ)

NOMIX = dict(kappa_conv=0.0, kappa_h=0.0, kappa_v=0.0, kappa_gm=0.0,
             kappa_redi=0.0, kappa_bi=0.0)

print("=== advection-ONLY time integration (all closures + forcing off) ===")
print("    any dH != 0 is a pure advection conservation defect")
print("%8s %14s %14s %14s" % ("nstep", "days", "dH (ZJ)", "rate (ZJ/yr)"))
for N in (100, 200, 400, 800):
    phys = replace(PhysicsConfig(), **NOMIX)
    step, _, _, pp, _ = JS.make_solver_global(
        g, phys, DT, lambda_bulk=0.0, mode_split=True, dt_bt=300.0, return_params=True)
    s = ST0
    H0 = H(s)
    for _ in range(N):
        s = step(s)
    jax.block_until_ready(s)
    dh = H(s) - H0
    days = N * DT / 86400.0
    print("%8d %14.2f %+14.4f %+14.2f" % (N, days, dh, dh / (days / 365.25)), flush=True)

print("\n=== same, but WITHOUT the rigid-lid top-face closure (Fz_top := 0) ===")
SRC = open("src/jax_solver_global.py", encoding="utf-8").read()
import types
old = "Fz_top = Fz_in[:, :, :1] * T[..., :1]"
assert SRC.count(old) == 1
ns = {"__name__": "js_notop", "__file__": "src/jax_solver_global.py"}
exec(compile(SRC.replace(old, old + " * 0.0"), "js_notop.py", "exec"), ns)
JS2 = types.SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})
for N in (100, 200, 400):
    phys = replace(PhysicsConfig(), **NOMIX)
    step, _, _, pp, _ = JS2.make_solver_global(
        g, phys, DT, lambda_bulk=0.0, mode_split=True, dt_bt=300.0, return_params=True)
    s = ST0
    H0 = H(s)
    for _ in range(N):
        s = step(s)
    jax.block_until_ready(s)
    dh = H(s) - H0
    days = N * DT / 86400.0
    print("%8d %14.2f %+14.4f %+14.2f" % (N, days, dh, dh / (days / 365.25)), flush=True)
