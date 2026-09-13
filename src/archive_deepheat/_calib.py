"""Calibrate terms_fn against an actual step() increment.

If sum(terms)*dt does not reproduce the real per-step dOHC, my interpretation
of the units/state is wrong. Also check for pathological cells.
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

RHO_0, C_P = 1025.0, 3992.0
SEC_YR = 3.15576e7
SPY = 8760.0

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)

d = np.load("results/ckpt_tenyr_ms_gm.npz")
u0 = jnp.asarray(np.asarray(d["u"], np.float64))
v0 = jnp.asarray(np.asarray(d["v"], np.float64))
T0 = jnp.asarray(np.asarray(d["T"], np.float64))
S0 = jnp.asarray(np.asarray(d["S"], np.float64))
e0 = jnp.asarray(np.asarray(d["eta"], np.float64))
print("T0  min %.4f max %.4f mean %.4f  nan %d" % (float(T0.min()), float(T0.max()), float(T0.mean()), int(jnp.isnan(T0).sum())))
print("S0  min %.4f max %.4f" % (float(S0.min()), float(S0.max())))
print("eta min %.4f max %.4f" % (float(e0.min()), float(e0.max())))
print("u   min %.4f max %.4f" % (float(u0.min()), float(u0.max())))

init = np.load("init_fields_g360x120.npz")
T_atm = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])

phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)

step, init_fn, diag, p, terms_fn = make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)

DZN = np.asarray(p.dz_node).ravel()
wet3 = np.asarray(p.wet_mask_z, float)
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]

def ohc(T):
    return float((jnp.where(wet3 > 0.5, T, 0.0) * vol).sum()) * RHO_0 * C_P / 1e21

# build state via the solver's own init to guarantee format
st_fresh = init_fn(jnp.asarray(T0), jnp.asarray(S0))
print("\ninit_fn output type:", type(st_fresh))
try:
    print("  fields:", st_fresh._fields)
except Exception:
    pass

st = JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)
o0 = ohc(st.T)
st1 = step(st)
o1 = ohc(st1.T)
print("\nREAL one-step dOHC = %+.6f ZJ/step = %+.4f ZJ/yr" % (o1 - o0, (o1 - o0) * SPY))

tn = np.asarray(terms_fn(st))
print("\nterm field stats (raw):")
for i, n in enumerate(["adv_T", "diff_h_T", "diff_v_T", "conv_T", "gm_T", "redi_T"]):
    a = tn[i]
    print("  %-9s min %+.4e max %+.4e mean %+.4e  |sum(dt=3600)| %.4f ZJ/yr"
          % (n, a.min(), a.max(), a.mean(),
             float((a * vol).sum()) * RHO_0 * C_P / 1e21 * 3600.0 * SPY))

# where is the mass concentrated?
i = 0
a = np.abs(tn[0])
tot = (a * vol).sum()
ordv = np.argsort(a.ravel())[::-1][:8]
print("\ntop-8 |adv_T| cells (flat idx, val, vol, share of total):")
vflat = vol.ravel()
for q in ordv:
    print("   %8d  %.4e  vol %.4e  share %.4f" % (q, a.ravel()[q], vflat[q], a.ravel()[q]*vflat[q]/tot))
