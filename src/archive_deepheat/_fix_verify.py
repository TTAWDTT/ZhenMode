"""Verify vertical-diffusion fixes vs baseline, with clean isolation.

Variants (all applied by string-patching a copy of the module):
  V0  baseline, unmodified
  V1  DEFECT B only : swap node-form kappa_v*_d2_dz2 -> conservative
                      interface-flux operator in all three places, leaving the
                      (wrong) double application in place
  V2  V1 + DEFECT A : also subtract the kappa_v term in
                      _compute_tracer_residual, mirroring how kappa_h is
                      subtracted, so kappa_v is applied once per step
  V3  DEFECT A only : add the node-form kappa_v subtraction to the residual,
                      keep node-form everywhere
"""
import sys, time
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
T0 = np.asarray(d["T"], np.float64); S0 = np.asarray(d["S"], np.float64)
u0 = np.asarray(d["u"], np.float64); v0 = np.asarray(d["v"], np.float64)
e0 = np.asarray(d["eta"], np.float64)
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])

src = open("src/jax_solver_global.py", encoding="utf-8").read()

HELPER = '''

def _diff_v_flux_tendency(tracer, kappa, p):
    """Conservative interface-flux vertical diffusion (zero flux top/bottom)."""
    if kappa <= 0.0:
        return jnp.zeros_like(tracer)
    tr = _fill_ghost_bottom(tracer, p)
    Fz_i = -kappa * (tr[..., 1:] - tr[..., :-1]) / p.dz_iface
    wet_if = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    Fz_i = jnp.where(wet_if, Fz_i, 0.0)
    up = jnp.concatenate([jnp.zeros_like(Fz_i[..., :1]), Fz_i], axis=-1)
    dn = jnp.concatenate([Fz_i, jnp.zeros_like(Fz_i[..., :1])], axis=-1)
    return (up - dn) / p.dz_node * p.wet_mask_z

'''
ANCHOR = "\ndef _conv_flux_tendency(tracer, conv_mask_3d, kappa, p):"
assert src.count(ANCHOR) == 1
src = src.replace(ANCHOR, HELPER + ANCHOR)

# --- the swap that defines defect B being fixed ---
NODE_USES = [
    ("diff_v_T = p.kappa_v * _d2_dz2(state.T, p)",
     "diff_v_T = _diff_v_flux_tendency(state.T, p.kappa_v, p)"),
    ("diff_v_S = p.kappa_v * _d2_dz2(state.S, p)",
     "diff_v_S = _diff_v_flux_tendency(state.S, p.kappa_v, p)"),
    ("    T = T + p.kappa_v * _d2_dz2(state.T, p) * dt_half\n"
     "    S = S + p.kappa_v * _d2_dz2(state.S, p) * dt_half",
     "    T = T + _diff_v_flux_tendency(state.T, p.kappa_v, p) * dt_half\n"
     "    S = S + _diff_v_flux_tendency(state.S, p.kappa_v, p) * dt_half"),
]

RESID_ANCHOR = """    dTdt = dTdt - p.kappa_h * _laplacian_h(state.T, p)
    dSdt = dSdt - p.kappa_h * _laplacian_h(state.S, p)
    return dTdt, dSdt"""
assert src.count(RESID_ANCHOR) == 1

def sub_kv(op):
    return (RESID_ANCHOR,
            RESID_ANCHOR.replace("    return dTdt, dSdt",
                                 "    dTdt = dTdt - " + op + "(state.T, p.kappa_v, p)\n"
                                 "    dSdt = dSdt - " + op + "(state.S, p.kappa_v, p)\n"
                                 "    return dTdt, dSdt"))

def apply(s, pairs):
    for old, new in pairs:
        n = s.count(old)
        assert n >= 1, "count %d for %r" % (n, old[:60])
        print("   swap x%d  %s" % (n, old.strip().splitlines()[0][:70]))
        s = s.replace(old, new)
    return s

variants = {}
variants["V1_B_only"] = apply(src, NODE_USES)
variants["V2_B_and_A"] = apply(apply(src, NODE_USES), [sub_kv("_diff_v_flux_tendency")])
variants["V3_A_only"] = apply(src, [sub_kv("_d2_dz2")])

import jax_solver_global as JS
for name, text in variants.items():
    open("_js_%s.py" % name, "w", encoding="utf-8").write(text)

phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)

def build(mod):
    step, _, _, p, _ = mod.make_solver_global(
        g, phys, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
        mode_split=True, dt_bt=300.0, return_params=True)
    return step, p

_, P = build(JS)
wet3 = np.asarray(P.wet_mask_z, float)
DZN = np.asarray(P.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = 1025.0 * 3992.0 / 1e21 * 8760.0
st = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                  S=jnp.asarray(S0), eta=jnp.asarray(e0))

def profile(mod, tag):
    step, _ = build(mod)
    t0 = time.time()
    s1 = step(st)
    jax.block_until_ready(s1)
    dT = np.asarray(s1.T) - T0
    print("\n%-14s (%2.0f s)" % (tag, time.time() - t0))
    rows = np.array([(dT[:, :, k] * vol[:, :, k]).sum() * ZJ for k in range(14)])
    print("   " + " ".join("%+7.1f" % r for r in rows))
    print("   column %+8.2f ZJ/yr" % rows.sum())
    return rows

print("per-level OHC tendency (ZJ/yr), k=0..13, positive = warming")
r0 = profile(JS, "V0 baseline")
res = {}
for name in ["V1_B_only", "V2_B_and_A", "V3_A_only"]:
    mod = __import__("_js_" + name)
    res[name] = profile(mod, name)
    print("   delta vs V0 : " + " ".join("%+7.1f" % x for x in (res[name] - r0)))
