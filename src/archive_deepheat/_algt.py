"""Why isn't the H_col shear projection EXACTLY zero?

Algebra: Fz[0] = sum_k div_h(u'_k)*dz_k. div_h is linear in (u,v) and dz has
no (i,j) dependence, so
    Fz[0] = div_h( sum_k u'_k*dz_k , sum_k v'_k*dz_k ).
With u'_k = (u_k - c)*m_k and c = sum_k u_k*m_k*dz_k / sum_k m_k*dz_k,
    sum_k u'_k*dz_k = sum_k u_k m_k dz_k - c*sum_k m_k dz_k = 0
IDENTICALLY (no assumption on m being 0/1). So Fz[0] must be exactly 0.
Measured: rms 6.75e-6. Resolve it.

Test each link of the chain numerically.
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

DT = 3600.0
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
u0 = np.asarray(d["u"], np.float64); v0 = np.asarray(d["v"], np.float64)
T0 = np.asarray(d["T"], np.float64); S0 = np.asarray(d["S"], np.float64)
e0 = np.asarray(d["eta"], np.float64)
init = np.load("init_fields_g360x120.npz")
T_atm_np = np.asarray(air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0]), float)

NOTRACER = dict(kappa_conv=0.0, kappa_h=0.0, kappa_v=0.0, kappa_gm=0.0,
                kappa_redi=0.0, kappa_bi=0.0, nu_h=5e6, nu_bi=2e14)
_, _, _, p, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **NOTRACER), DT, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=0.0, mode_split=True, dt_bt=300.0, polar_cap_rows=2,
    polar_cap_taper=3, return_params=True)

M = jnp.asarray(np.asarray(p.wet_mask_z, float))
DZN = np.asarray(p.dz_node).ravel()
DZ = jnp.asarray(DZN).reshape(1, 1, -1)
print("  dz_node       :", np.round(DZN, 3))
print("  p.dz_node shape: %s   sum %.4f" % (np.asarray(p.dz_node).shape, DZN.sum()))
print("  wet_mask_z unique:", np.unique(np.asarray(p.wet_mask_z, float))[:6],
      " n_unique", len(np.unique(np.asarray(p.wet_mask_z, float))))
wmv = np.asarray(p.wet_mask_z, float)
print("  wet_mask_z fractional entries: %d of %d (%.4f%%)"
      % (int(((wmv > 1e-9) & (wmv < 1 - 1e-9)).sum()), wmv.size,
         100.0 * ((wmv > 1e-9) & (wmv < 1 - 1e-9)).sum() / wmv.size))

U = jnp.asarray(u0); V = jnp.asarray(v0)


def shear(a):
    Hcol = jnp.sum(M * DZ, axis=-1, keepdims=True)
    c = jnp.sum(a * M * DZ, axis=-1, keepdims=True) / jnp.maximum(Hcol, 1e-9)
    return (a - c) * M


Us, Vs = shear(U), shear(V)

print()
print("=" * 88)
print("Link 1: is sum_k u'_k*dz_k == 0 pointwise?")
print("=" * 88)
sU = np.asarray(jnp.sum(Us * DZ, axis=-1), float)
sV = np.asarray(jnp.sum(Vs * DZ, axis=-1), float)
print("  sum_k u'_k dz_k : max|.| %.4e   rms %.4e" % (np.abs(sU).max(), np.sqrt((sU ** 2).mean())))
print("  sum_k v'_k dz_k : max|.| %.4e   rms %.4e" % (np.abs(sV).max(), np.sqrt((sV ** 2).mean())))

print()
print("=" * 88)
print("Link 2: is div_h(Us,Vs) == sum_k div_h(u'_k,v'_k)*dz_k  (linearity)?")
print("=" * 88)
Fz_proj = JS._vertical_transport_iface(Us, Vs, p)
f0 = np.asarray(Fz_proj[:, :, 0], float)
print("  Fz[0] direct from _vertical_transport_iface: rms %.4e  max %.4e"
      % (np.sqrt((f0 ** 2).mean()), np.abs(f0).max()))
# per-layer sum, i.e. recompute the identity by hand
divh = JS._divergence_h(Us, Vs, p)
hand = np.asarray(jnp.sum(divh * DZ, axis=-1), float)
print("  sum_k div_h(u'_k,v'_k)*dz_k (by hand):       rms %.4e  max %.4e"
      % (np.sqrt((hand ** 2).mean()), np.abs(hand).max()))
print("  match Fz[0]? max|diff| %.4e" % np.abs(hand - f0).max())
# div_h of the depth-summed velocity
dh_sum = np.asarray(JS._divergence_h(jnp.sum(Us * DZ, axis=-1, keepdims=True),
                                     jnp.sum(Vs * DZ, axis=-1, keepdims=True), p),
                    float)
print("  div_h(sum_k u'_k dz_k):                      rms %.4e  max %.4e"
      % (np.sqrt((dh_sum ** 2).mean()), np.abs(dh_sum).max()))

print()
print("=" * 88)
print("Link 3: same test but with a plain (unmasked) shear, to isolate the mask")
print("=" * 88)
for label, mm in (("masked m=wet_mask_z", M), ("all-ones m", jnp.ones_like(M))):
    Hc = jnp.sum(mm * DZ, axis=-1, keepdims=True)
    cc = jnp.sum(U * mm * DZ, axis=-1, keepdims=True) / jnp.maximum(Hc, 1e-9)
    Ux = (U - cc) * mm
    Vx = (V - cc * 0.0) * mm
    sx = np.asarray(jnp.sum(Ux * DZ, axis=-1), float)
    fz = np.asarray(JS._vertical_transport_iface(Ux, Vx, p)[:, :, 0], float)
    print("  %-22s sum_k u' dz: max %.3e | Fz[0] rms %.3e max %.3e"
          % (label, np.abs(sx).max(), np.sqrt((fz ** 2).mean()), np.abs(fz).max()))

print()
print("=" * 88)
print("Link 4: is _divergence_h actually linear?  div_h(a*u) vs a*div_h(u)")
print("=" * 88)
rng = np.random.default_rng(0)
A = jnp.asarray(rng.standard_normal(u0.shape) * wmv)
lhs = np.asarray(JS._divergence_h(2.5 * A, V, p), float)
rhs = 2.5 * np.asarray(JS._divergence_h(A, V, p), float)
print("  div_h(2.5*u) - 2.5*div_h(u): max %.4e  rms %.4e"
      % (np.abs(lhs - rhs).max(), np.sqrt(((lhs - rhs) ** 2).mean())))
lhs2 = np.asarray(JS._divergence_h(A + Us, V, p), float)
rhs2 = np.asarray(JS._divergence_h(A, V, p), float) + np.asarray(JS._divergence_h(Us, V, p), float)
print("  div_h(A+u) - div_h(A) - div_h(u): max %.4e" % np.abs(lhs2 - rhs2).max())
