"""Operator-level A/B of candidate fixes for the advection non-conservation.

The defect is exactly:
    integral adv(T) dV = sum_cells area * Fz[0] * T[0]
(the horizontal branches telescope to zero; the vertical branch telescopes
down to the single top face). So every candidate fix is a change to Fz[0].

NOTE ON UNITS: the operator is a TENDENCY (K/s) with the layer thickness
already divided out, so the volume weight here is area*dz per layer -- the
ZJ/yr numbers below are the rate at which the operator injects heat.

Variants:
  base   Fz[0] as produced by _vertical_transport_iface (the column-integrated
         _divergence_h)
  zero   Fz[0] = 0
  bt     Fz[0] = H_sw * div_bt  (the divergence the free surface actually
         absorbs, from _divergence_conservative(ubt, vbt))
  btcol  Fz[0] = H_sw * div_bt but with ubt re-derived as the node-centred
         depth average sum_k u_k*dz_node_k/H_sw (the exact depth-average
         whose 2D divergence equals the column sum of _divergence_h)
  demean Fz[0] -= area-weighted mean leak (removes the net, keeps the
         T-correlated part) -- expected to FAIL, controls for the hypothesis
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

RHO_0, C_P = 1025.0, 3992.0
ZJYR = RHO_0 * C_P / 1e21 * 3.1536e7
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
u0 = jnp.asarray(np.asarray(d["u"], np.float64))
v0 = jnp.asarray(np.asarray(d["v"], np.float64))
T0 = jnp.asarray(np.asarray(d["T"], np.float64))
S0 = jnp.asarray(np.asarray(d["S"], np.float64))
e0 = jnp.asarray(np.asarray(d["eta"], np.float64))
init = np.load("init_fields_g360x120.npz")
T_atm_np = np.asarray(air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0]), float)
BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)
_, _, _, p, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
AREA3 = AREA[:, :, None]
vol = wet3 * AREA3 * DZN[None, None, :]
H_sw = float(p.H_sw)
st = JS.JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)

print("H_sw = %.4f   sum(dz_node) = %.4f   nz = %d" % (H_sw, DZN.sum(), DZN.size))
print("p.dz_node shape %s ; p.wet_mask shape %s" %
      (np.asarray(p.dz_node).shape, np.asarray(p.wet_mask).shape))

def integ_rate(adv):
    """ZJ/yr the operator injects (adv is K/s per cell)."""
    return float((np.asarray(adv, float) * vol).sum()) * RHO_0 * C_P * 3.1536e7 / 1e21

Fz_base = JS._vertical_transport_iface(st.u, st.v, p)

# ── the barotropic divergence the free surface actually absorbs ──
ubt2, vbt2 = JS._barotropic_velocity(st.u, st.v, p)
div_bt = JS._divergence_conservative(ubt2 * p.wet_mask, vbt2 * p.wet_mask, p)
f0_bt = H_sw * np.asarray(div_bt, float)

# ── node-centred depth average, the exact depth-mean of _divergence_h ──
u_node = jnp.sum(st.u * p.dz_node, axis=-1) / H_sw
v_node = jnp.sum(st.v * p.dz_node, axis=-1) / H_sw
div_node = JS._divergence_conservative(u_node * p.wet_mask, v_node * p.wet_mask, p)
f0_node = H_sw * np.asarray(div_node, float)

f0_base = np.asarray(Fz_base[:, :, 0], float)
surfm = wet3[:, :, 0] > 0.5
print("\n--- top-face leak Fz[0] (m^2/s) ---")
for nm, f in [("base _divergence_h", f0_base), ("bt  _barotropic_velocity", f0_bt),
              ("nod node-centred depthavg", f0_node)]:
    print("  %-26s rms %.3e  max %.3e  sum*area %+.4e  sum|.|*area %.4e"
          % (nm, np.sqrt((f[surfm] ** 2).mean()), np.abs(f[surfm]).max(),
             (f * AREA * surfm).sum(), (np.abs(f) * AREA * surfm).sum()))

print("\n--- pairwise agreement (rms of difference over surface cells) ---")
print("  base vs bt    %.4e" % np.sqrt(((f0_base - f0_bt)[surfm] ** 2).mean()))
print("  base vs nod   %.4e" % np.sqrt(((f0_base - f0_node)[surfm] ** 2).mean()))
print("  bt   vs nod   %.4e" % np.sqrt(((f0_bt - f0_node)[surfm] ** 2).mean()))
print("  (bt should equal nod EXACTLY if _barotropic_velocity were the")
print("   node-centred depth average -- a nonzero value is the operator split)")

# ── evaluate the operator under each variant ──
def with_f0(f0):
    return Fz_base.at[:, :, 0].set(jnp.asarray(f0, jnp.float64))

demean = f0_base - (f0_base * AREA * surfm).sum() / (AREA * surfm).sum()

print("\n" + "=" * 78)
print("integral adv(T) dV for each variant   [ZJ/yr, exact conservation = 0]")
print("=" * 78)
print("  %-26s %+12.4f" % ("base (current code)", integ_rate(JS._advection_scalar(st.T, st.u, st.v, Fz_base, p))))
for nm, f0 in [("zero  Fz[0]=0", np.zeros_like(f0_base)),
               ("bt    H_sw*div_bt", f0_bt),
               ("nod   H_sw*div_node", f0_node),
               ("demean  (control)", demean)]:
    print("  %-26s %+12.4f" % (nm, integ_rate(JS._advection_scalar(st.T, st.u, st.v, with_f0(f0), p))))

print("\n  and adv(1) must stay ~0 (mass consistency at the top node):")
for nm, f0 in [("base", f0_base), ("zero", np.zeros_like(f0_base)),
               ("bt", f0_bt), ("nod", f0_node)]:
    a1 = np.asarray(JS._advection_scalar(jnp.ones_like(st.T), st.u, st.v, with_f0(f0), p), float)
    print("    %-6s max|adv(1)| in wet = %.4e" % (nm, np.abs(a1[wet3 > 0.5]).max()))
