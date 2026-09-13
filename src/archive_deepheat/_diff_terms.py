"""Measure each contribution by DIRECT DIFFERENCING of step().

dOHC(baseline) - dOHC(term disabled) = that term's true contribution.
No algebraic reconstruction, no unit guessing.
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

RHO_0, C_P, SPY = 1025.0, 3992.0, 8760.0
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
u0 = jnp.asarray(np.asarray(d["u"], np.float64)); v0 = jnp.asarray(np.asarray(d["v"], np.float64))
T0 = jnp.asarray(np.asarray(d["T"], np.float64)); S0 = jnp.asarray(np.asarray(d["S"], np.float64))
e0 = jnp.asarray(np.asarray(d["eta"], np.float64))
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])

BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)

def build(**over):
    phys = replace(PhysicsConfig(), **{**BASE, **over})
    return make_solver_global(g, phys, 3600.0, T_atm=jnp.asarray(T_atm_np),
                              lambda_bulk=BULK_LAMBDA_DEFAULT,
                              mode_split=True, dt_bt=300.0, return_params=True)

st = JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)

def measure(tag, **over):
    step, _, _, p, _ = build(**over)
    wet3 = np.asarray(p.wet_mask_z, float)
    DZN = np.asarray(p.dz_node).ravel()
    vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
    o0 = float((jnp.where(wet3 > 0.5, st.T, 0.0) * vol).sum())
    s1 = step(st)
    o1 = float((jnp.where(wet3 > 0.5, s1.T, 0.0) * vol).sum())
    r = (o1 - o0) * RHO_0 * C_P / 1e21 * SPY
    print("  %-24s dOHC = %+10.3f ZJ/yr" % (tag, r))
    return r

print("direct term differencing at ckpt_tenyr_ms_gm (real u,v):")
base = measure("BASELINE (all on)")

# disable one term at a time
name_map = [
    ("no bulk flux",        dict()),
    ("no gm+redi",          dict(kappa_gm=0.0, kappa_redi=0.0)),
    ("no kappa_v",          dict(kappa_v=0.0)),
    ("no adv",              None),
    ("no conv",             dict(kappa_conv=0.0)),
    ("no kappa_h",          dict(kappa_h=0.0)),
    ("no nu_h",             dict(nu_h=0.0)),
]
for tag, over in name_map:
    if over is None:
        print("  %-24s (skipped: adv not switchable)" % tag); continue
    if tag == "no bulk flux":
        # rebuild with T_atm = current SST -> zero bulk forcing
        phys = replace(PhysicsConfig(), **BASE)
        step, _, _, p, _ = make_solver_global(g, phys, 3600.0,
            T_atm=jnp.asarray(np.asarray(T0)[:, :, 0]), lambda_bulk=BULK_LAMBDA_DEFAULT,
            mode_split=True, dt_bt=300.0, return_params=True)
        wet3 = np.asarray(p.wet_mask_z, float); DZN = np.asarray(p.dz_node).ravel()
        vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
        o0 = float((jnp.where(wet3 > 0.5, st.T, 0.0) * vol).sum())
        s1 = step(st); o1 = float((jnp.where(wet3 > 0.5, s1.T, 0.0) * vol).sum())
        r = (o1 - o0) * RHO_0 * C_P / 1e21 * SPY
        print("  %-24s dOHC = %+10.3f ZJ/yr   => bulk contributes %+10.3f"
              % (tag, r, base - r))
        continue
    r = measure(tag, **over)
    print("       => %-16s contributes %+10.3f ZJ/yr" % (tag[3:], base - r))
