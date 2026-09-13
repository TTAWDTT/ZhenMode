"""The shear projection that ACTUALLY zeroes Fz[0] exactly.

zf0 (Fz[0]:=0) is rejected: it conserves globally but reintroduces the
pointwise T-proportional pump (_zf0check.out: max|S| 38.9 -> 148 in 60 steps,
pointwise adv(1) 6.7e-5/s vs base 4e-21).  The codebase already documented
that exact failure mode (line 533: "resid * T is a T-PROPORTIONAL source --
a 4%/step exponential pump").

The two requirements are:
  (R1) adv(1) == 0 POINTWISE   -> no pump: needs Fz_top = Fz[0]*1, i.e. the
       face-flux divergence must close for a uniform field.
  (R2) I(adv(T)) == 0          -> heat conservation: the identity
       I(adv(T)) = -sum AREA*Fz[0]*T[0] forces Fz[0] == 0.

Both hold simultaneously iff Fz[0] == 0 EXACTLY, with the top face still
carried by Fz_top = Fz[0]*T[0] = 0 (consistent for BOTH uniform and varying
T -- no pump, no leak).

Why the earlier shearL/shearH attempts missed this:
  Fz[0] = sum_k div_h(u'_k)*dz_k = div_h( sum_k u'_k*dz_k )   (dz_k has no
  i,j dependence, div_h is linear, so it factors).
  With u'_k = (u_k - c)*m_k:
     sum_k u'_k dz_k = c*LSPAN - c*H_col   where H_col = sum_k m_k dz_k
  which is NOT zero because LSPAN = sum(dz_node) = 5002.5 but H_col is the
  WET column depth (127.5..5002.5).  The re-masking by m_k after subtracting
  a full-span mean breaks the identity.
  Correct normalization: c = sum_k u_k*m_k*dz_k / H_col, so
     sum_k (u_k - c) m_k dz_k = 0 EXACTLY -> Fz[0] = div_h(0) = 0 exactly.

Verify: Fz[0] rms, pointwise adv(1), I(adv(T)), I(adv(1)), and a 200-step
H budget + tracer boundedness, against base.
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

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
DZ = jnp.asarray(DZN).reshape(1, 1, -1)
M = jnp.asarray(wet3)

ORIG_VTI = JS._vertical_transport_iface
ORIG_ADV = JS._advection_scalar


def I(f):
    return float((np.asarray(f, float) * vol).sum() * ZJ * DT)


def H(Tf):
    return float((np.asarray(Tf, float) * vol).sum() * ZJ)


def _shear_correct(a):
    """Remove the WET-COLUMN depth mean (H_col-normalized), then re-mask."""
    Hcol = jnp.sum(M * DZ, axis=-1, keepdims=True)          # (nx,ny,1)
    c = jnp.sum(a * M * DZ, axis=-1, keepdims=True) / jnp.maximum(Hcol, 1e-9)
    return (a - c) * M


def install(variant):
    if variant == "base":
        JS._vertical_transport_iface = ORIG_VTI
        JS._advection_scalar = ORIG_ADV
        return

    def vti(u, v, pp):
        return ORIG_VTI(_shear_correct(u), _shear_correct(v), pp)

    def adv(T, u, v, Fz, pp):
        return ORIG_ADV(T, _shear_correct(u), _shear_correct(v), Fz, pp)

    JS._vertical_transport_iface = vti
    JS._advection_scalar = adv


print("=" * 92)
print("Does H_col-normalized shear zero Fz[0] EXACTLY?")
print("=" * 92)
s0 = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                  S=jnp.asarray(S0), eta=jnp.asarray(e0))
A2 = AREA * wet3[:, :, 0]
for vname in ("base", "shear"):
    install(vname)
    Fz = JS._vertical_transport_iface(s0.u, s0.v, p)
    f0 = np.asarray(Fz[:, :, 0], float)
    advT = JS._advection_scalar(s0.T, s0.u, s0.v, Fz, p)
    adv1 = np.asarray(JS._advection_scalar(jnp.ones_like(s0.T), s0.u, s0.v, Fz, p), float)
    a1w = np.abs(adv1 * wet3)
    print("  %-5s Fz[0] rms %.4e  max %.4e   sum(area*Fz0) %+.3e"
          % (vname, np.sqrt((f0 ** 2).mean()), np.abs(f0).max(), float((A2 * f0).sum())))
    print("        pointwise max|adv(1)|  all %.3e | k=0 %.3e | k>=1 %.3e"
          % (a1w.max(), np.abs(adv1[:, :, 0] * wet3[:, :, 0]).max(),
             np.abs(adv1[:, :, 1:] * wet3[:, :, 1:]).max()))
    print("        I(adv(T)) %+13.8f ZJ/step   I(adv(1))*1e9 %+.4e"
          % (I(advT), I(adv1) * 1e9))
    print("        removed-velocity rms: u %.4e  v %.4e m/s"
          % (float(jnp.sqrt(jnp.mean((_shear_correct(s0.u) - s0.u) ** 2))),
             float(jnp.sqrt(jnp.mean((_shear_correct(s0.v) - s0.v) ** 2)))))

install("base")

print()
print("=" * 92)
print("200-step H budget + boundedness, advection-only (all closures off, no forcing)")
print("=" * 92)
for vname in ("base", "shear"):
    install(vname)
    s = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                     S=jnp.asarray(S0), eta=jnp.asarray(e0))
    h0 = H(s.T)
    hs = [h0]
    rows = []
    n = 0
    for k in range(200):
        s = JS._step_impl(s, p)
        Tn = np.asarray(s.T, float)
        hs.append(H(Tn))
        n = k + 1
        if not np.isfinite(hs[-1]):
            print("  %s DIVERGED at step %d" % (vname, n))
            break
        if n % 40 == 0:
            rows.append((n, hs[-1] - h0, np.abs(Tn * wet3).max(),
                         np.abs(np.asarray(s.S, float) * wet3).max(),
                         np.abs(np.asarray(s.u, float) * wet3).max(),
                         np.abs(np.asarray(s.eta, float)).max()))
    else:
        pass
    dH = hs[-1] - hs[0]
    print("  --- %s ---   dH %+13.6f ZJ  rate %+11.3f ZJ/yr  |dH|/step max %.3e"
          % (vname, dH, dH / (n * DT / 3.1536e7) if n else 0,
             max(abs(hs[i + 1] - hs[i]) for i in range(len(hs) - 1)) if n else 0))
    print("     step       dH(ZJ)     max|T|    max|S|    max|u|   max|eta|")
    for r in rows:
        print("     %4d  %+12.6f  %8.3f  %8.3f  %8.4f  %8.4f" % r)

install("base")
