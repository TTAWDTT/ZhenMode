"""Locate the instability trigger and test an interface-local conv gate.

Hypothesis: the WHOLE-COLUMN gate (any(unstable_iface)) applies
kappa_conv=0.05 -- 5000x kappa_v -- to all 14 levels of a flagged column, so
a single near-surface instability mixes surface heat to 4000 m. Test by
patching the gate to interface-local and re-measuring the deep tendency.
"""
import sys, types
from dataclasses import replace
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
T0np = np.asarray(d["T"], np.float64); S0np = np.asarray(d["S"], np.float64)
u0np = np.asarray(d["u"], np.float64); v0np = np.asarray(d["v"], np.float64)
e0np = np.asarray(d["eta"], np.float64)
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
lat = init["lat"]
phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)

import jax_solver_global as JS

def build(mod, **over):
    ph = replace(PhysicsConfig(), **{**dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
        kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05,
        gm_slope_max=0.005), **over})
    step, init_fn, diag, p, terms_fn = mod.make_solver_global(
        g, ph, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
        mode_split=True, dt_bt=300.0, return_params=True)
    return step, p, terms_fn

# ── baseline ──
step_b, p_b, terms_b = build(JS)
DZN = np.asarray(p_b.dz_node).ravel()
wet3 = np.asarray(p_b.wet_mask_z, float)
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = 1025.0*3992.0/1e21*3600.0*8760.0
st = JS.JaxStateG(u=jnp.asarray(u0np), v=jnp.asarray(v0np),
                  T=jnp.asarray(T0np), S=jnp.asarray(S0np), eta=jnp.asarray(e0np))

# ── conv mask stats (fix lat indexing: lat is axis 1) ──
rho = np.asarray(JS._density_anomaly(st.T, st.S, p_b)) * wet3
wet_iface = (wet3[:, :, :-1] > 0.5) & (wet3[:, :, 1:] > 0.5)
unst = (rho[:, :, :-1] > rho[:, :, 1:]) & wet_iface
col_mask = unst.any(axis=-1)
surf = wet3[:, :, 0] > 0.5
print("flagged columns: %d of %d wet surface (%.1f%%)"
      % (int(col_mask.sum()), int(surf.sum()), 100.0*col_mask.sum()/surf.sum()))
jj, ii = np.where(col_mask)          # ii = lat index (axis 1)
print("lat range of flagged columns: %.1f .. %.1f N" % (lat[ii].min(), lat[ii].max()))
h, edges = np.histogram(lat[ii], bins=10, range=(-60, 60))
print("lat histogram (60S..60N):", h)

print("\nunstable-iface depth distribution (all columns):")
Z = np.array([0,5,15,30,50,75,100,150,200,300,500,1000,2000,4000], float)
for k in range(13):
    n = int(unst[:, :, k].sum())
    if n:
        print("   iface %2d (zc %6.0f m): %6d" % (k, -0.5*(Z[k]+Z[k+1]), n))

print("\nrho' rho-profile (ocean mean over wet cells):")
for k in range(14):
    m = wet3[:, :, k] > 0.5
    print("   k=%2d  rho' %+.6f  T %+.3f  S %+.3f" % (k, rho[:, :, k][m].mean(),
          T0np[:, :, k][m].mean(), S0np[:, :, k][m].mean()))

# where is the shallowest instability in flagged columns?
first = np.argmax(unst, axis=-1)
print("\nshallowest unstable iface in flagged columns:")
for k in range(13):
    n = int((first[col_mask] == k).sum())
    if n:
        print("   first-unstable iface %2d: %5d columns" % (k, n))

# ── patched module: interface-local conv gate ──
src = open("src/jax_solver_global.py", encoding="utf-8").read()
old = "    conv_mask_3d = jnp.any(unstable_iface, axis=-1, keepdims=True)\n    conv_T = _conv_flux_tendency(state.T, conv_mask_3d, p.kappa_conv, p)"
new = "    conv_mask_3d = jnp.any(unstable_iface, axis=-1, keepdims=True)\n    conv_T = _conv_flux_tendency(state.T, unstable_iface.astype(state.T.dtype), p.kappa_conv, p)"
assert src.count(old) == 1, "tracer_terms site count %d" % src.count(old)
old2 = """    conv_mask_3d = jnp.any(unstable_iface, axis=-1, keepdims=True)
    # kappa_conv CFL"""
new2 = """    conv_mask_3d = unstable_iface.astype(state.T.dtype)
    # kappa_conv CFL"""
assert src.count(old2) == 1, "tendency site count %d" % src.count(old2)
patched = src.replace(old, new).replace(old2, new2)
open("_js_patched.py", "w", encoding="utf-8").write(patched)
sys.path.insert(0, ".")
import _js_patched as JSP

def run(mod, tag):
    step, p, terms = build(mod)
    o0 = float((jnp.where(wet3 > 0.5, st.T, 0.0) * vol).sum())
    s1 = step(st)
    o1 = float((jnp.where(wet3 > 0.5, s1.T, 0.0) * vol).sum())
    dT = np.asarray(s1.T) - T0np
    print("\n%s: TRUE dOHC = %+.3f ZJ/yr" % (tag, (o1-o0)*ZJ/3600.0/8760.0*3600.0))
    print("   per-level dT (degC per step) and ZJ/yr:")
    tot = 0.0
    for k in range(14):
        v = (dT[:, :, k] * vol[:, :, k]).sum() * 1025.0*3992.0/1e21 * 8760.0
        tot += v
        print("     k=%2d zc %6.0f  dT %+.3e   %+9.2f ZJ/yr" % (k, -Z[k], dT[:, :, k][wet3[:, :, k] > 0.5].mean(), v))
    print("   column sum %+.2f ZJ/yr" % tot)
    return dT

dT_b = run(JS, "BASELINE (whole-column gate)")
dT_p = run(JSP, "PATCHED (interface-local gate)")
print("\nCHANGE in per-level ZJ/yr (patched - baseline):")
for k in range(14):
    a = (dT_b[:, :, k] * vol[:, :, k]).sum() * 1025.0*3992.0/1e21 * 8760.0
    b = (dT_p[:, :, k] * vol[:, :, k]).sum() * 1025.0*3992.0/1e21 * 8760.0
    print("   k=%2d zc %6.0f  %+9.2f" % (k, -Z[k], b - a))
