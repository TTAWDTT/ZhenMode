"""TIME-INTEGRATION gate for candidate advection fixes.

Operator-level facts (see _fixtest.out / _projtest.out):
  * integral adv(T) dV == -sum_cells area * Fz[0] * T[0]   (exact identity)
  * integral adv(1) dV == -sum_cells area * Fz[0]          (~0 in base)
  * therefore BOTH properties hold for all T IFF Fz[0] == 0.
  * no local choice of the top-face value can do this: "bt" (the divergence
    the free surface actually absorbs) still leaks +38.2 ZJ/yr, and the
    demean control leaks the full +66.6 -- the leak is carried by the
    CORRELATION of Fz[0] with SST, not by its mean.

So the fix has to make the tracer's column-integrated horizontal divergence
vanish, not re-value the top face.

This script runs the advection-only closed-domain test (every tracer closure
off, no surface heat flux) under each candidate and measures dH. Ground
truth: dH must be 0.
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
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
u0 = np.asarray(d["u"], np.float64); v0 = np.asarray(d["v"], np.float64)
T0 = np.asarray(d["T"], np.float64); S0 = np.asarray(d["S"], np.float64)
e0 = np.asarray(d["eta"], np.float64)
init = np.load("init_fields_g360x120.npz")
T_atm_np = np.asarray(air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0]), float)

NOTRACER = dict(kappa_conv=0.0, kappa_h=0.0, kappa_v=0.0, kappa_gm=0.0,
                kappa_redi=0.0, kappa_bi=0.0, nu_h=5e6, nu_bi=2e14)
_, _, _, p, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **NOTRACER), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=0.0, mode_split=True, dt_bt=300.0, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]

def heat(T):
    return float((T * vol).sum()) * RHO_0 * C_P / 1e21

# ── candidate Fz builders (monkeypatch _vertical_transport_iface) ──
_orig_vti = JS._vertical_transport_iface
DZNj = p.dz_node
colsum = jnp.sum(DZNj)

def vti_base(u, v, pp):
    return _orig_vti(u, v, pp)

def vti_zero(u, v, pp):
    return _orig_vti(u, v, pp).at[:, :, 0].set(0.0)

def vti_proj(u, v, pp):
    div_h = JS._divergence_h(u, v, pp)
    colint = jnp.sum(div_h * DZNj, axis=-1, keepdims=True)
    div_p = div_h - colint / colsum
    integ = div_p * DZNj
    Fz_int = jnp.cumsum(integ[..., ::-1], axis=-1)[..., ::-1]
    return jnp.concatenate([Fz_int, jnp.zeros_like(Fz_int[..., :1])], axis=-1)

VARIANTS = [("base ", vti_base), ("zero ", vti_zero), ("proj ", vti_proj)]
NSTEP = 80

print("advection-only closed domain, %d steps (all tracer closures off,"
      " no surface heat flux)" % NSTEP)
print("dH must be EXACTLY 0.\n")
print("%-8s %12s %12s %14s" % ("variant", "H0 (ZJ)", "H1 (ZJ)", "dH (ZJ)"))
for nm, fn in VARIANTS:
    JS._vertical_transport_iface = fn
    st = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                      S=jnp.asarray(S0), eta=jnp.asarray(e0))
    h0 = heat(T0)
    try:
        for k in range(NSTEP):
            st = JS._step_impl(st, p)
            if not np.isfinite(np.asarray(st.T)).all():
                print("%-8s NaN at step %d" % (nm, k + 1))
                break
        else:
            h1 = heat(np.asarray(st.T))
            print("%-8s %12.5f %12.5f %+14.6f" % (nm, h0, h1, h1 - h0))
    except Exception as ex:
        print("%-8s ERROR %s" % (nm, ex))
JS._vertical_transport_iface = _orig_vti
