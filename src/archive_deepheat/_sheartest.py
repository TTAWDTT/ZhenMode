"""THE FIX: advect tracers with the BAROCLINIC (shear) velocity only.

Algebra. Let u_k be the layer velocity, dz_node the layer thicknesses, and
div_h the horizontal divergence operator used throughout. Then

    C(i,j) := sum_k div_h(u_k)*dz[k] / sum_k dz[k]
            == div_h( ubt_exact ),   ubt_exact := sum_k u_k*dz[k]/sum_k dz[k]

(Fz[0] = sum_k div_h[k]*dz[k], and div_h is linear with k-independent metric,
so the column-weighted mean of the divergence IS the divergence of the
column-weighted mean velocity.) Fz[0] is exactly this column leak.

Now advect with the SHEAR velocities

    u'_k = u_k - ubt_exact,      v'_k = v_k - vbt_exact

Then div_h(u'_k) = div_h(u_k) - C for every k, so

    sum_k div_h'(k)*dz[k] = sum_k div_h(k)*dz[k] - C*sum_k dz[k] = 0

    =>  Fz'[0] = 0   (no transport through the free surface)
    =>  heat:  integral adv(T)dV = -sum AREA*Fz'[0]*T[0] = 0   EXACTLY
    =>  mass:  adv(1) = -(div_h' + div_z') = -(div_h' - div_h') = 0 EXACTLY

Both properties hold simultaneously — which no local top-face closure can do
(_fixtest.out: "bt" leaks +38.2 ZJ/yr, "demean" the full +66.6).

This is the standard rigid-lid construction: the barotropic mode's job is to
move eta, and in a fixed-thickness z-column it cannot carry tracer without
creating/destroying it. Physical cost: the depth-mean heat transport is
dropped, which is exactly the part the eta budget already accounts for.

Variants for the gate:
  base   u,v as-is                       (heat leak +66.58 ZJ/yr)
  shear  u-ubt_exact, v-vbt_exact        (predicted 0 / 0)
  btavg  u-_barotropic_velocity(u,v)     (control: the WRONG depth average)
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
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
st = JS.JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)

def heat_rate(adv):
    return float((np.asarray(adv, float) * vol).sum()) * RHO_0 * C_P * 3.1536e7 / 1e21

def shear_vel(u, v, pp):
    wm = pp.wet_mask_z
    L = jnp.sum(pp.dz_node)
    ubt = jnp.sum(u * wm * pp.dz_node, axis=-1) / L
    vbt = jnp.sum(v * wm * pp.dz_node, axis=-1) / L
    return (u - ubt[:, :, None]) * wm, (v - vbt[:, :, None]) * wm

def bt_vel(u, v, pp):
    ubt, vbt = JS._barotropic_velocity(u, v, pp)
    wm = pp.wet_mask_z
    return (u - ubt[:, :, None]) * wm, (v - vbt[:, :, None]) * wm

def evaluate(tag, uu, vv):
    Fz = JS._vertical_transport_iface(uu, vv, p)
    adv = JS._advection_scalar(st.T, uu, vv, Fz, p)
    a1 = np.asarray(JS._advection_scalar(jnp.ones_like(st.T), uu, vv, Fz, p), float)
    f0 = np.asarray(Fz[:, :, 0], float)
    surfm = wet3[:, :, 0] > 0.5
    r = np.array([(np.asarray(adv, float)[:, :, k] * vol[:, :, k]).sum()
                  for k in range(14)]) * RHO_0 * C_P * 3.1536e7 / 1e21
    print("  %-6s heat %+9.4f ZJ/yr | adv(1) max %.3e | Fz[0] rms %.3e | sum area*Fz0 %+.2e"
          % (tag, heat_rate(adv), np.abs(a1[wet3 > 0.5]).max(),
             np.sqrt((f0[surfm] ** 2).mean()), (f0 * AREA * surfm).sum()))
    print("         per level: " + " ".join("%+6.1f" % x for x in r))
    return heat_rate(adv)

print("=" * 96)
print("Is the baroclinic-velocity fix BOTH heat- and mass-conserving at the operator level?")
print("(exact conservation requires heat 0 AND adv(1) ~ 1e-21)")
print("=" * 96)
evaluate("base ", st.u, st.v)
us, vs = shear_vel(st.u, st.v, p)
evaluate("shear", us, vs)
ub, vb = bt_vel(st.u, st.v, p)
evaluate("btavg", ub, vb)

# how much velocity is being removed?
m = wet3 > 0.5
print("\nvelocity removed by the shear projection:")
print("  u   rms over wet %.4e m/s" % np.sqrt((np.asarray(st.u, float)[m] ** 2).mean()))
print("  ubt rms over wet %.4e m/s" % np.sqrt((np.asarray(st.u - us, float)[m] ** 2).mean()))
print("  fraction of u rms in the removed barotropic part: %.1f%%"
      % (100 * np.sqrt((np.asarray(st.u - us, float)[m] ** 2).mean())
         / np.sqrt((np.asarray(st.u, float)[m] ** 2).mean())))

# ── time-integration gate ──
print("\n" + "=" * 96)
print("TIME INTEGRATION: advection-only closed domain, all tracer closures off, no Q.")
print("dH must be EXACTLY 0.")
print("=" * 96)
NOTRACER = dict(kappa_conv=0.0, kappa_h=0.0, kappa_v=0.0, kappa_gm=0.0,
                kappa_redi=0.0, kappa_bi=0.0, nu_h=5e6, nu_bi=2e14)
_, _, _, pN, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **NOTRACER), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=0.0, mode_split=True, dt_bt=300.0, return_params=True)

_orig_vti = JS._vertical_transport_iface
_orig_tend = JS._compute_tracer_tendency
_orig_res = JS._compute_tracer_residual

def heatT(T):
    return float((np.asarray(T, float) * vol).sum()) * RHO_0 * C_P / 1e21

def run(tag, mode, nstep=60):
    if mode == "base":
        JS._vertical_transport_iface = _orig_vti
    else:
        def vti(u, v, pp):
            uu, vv = (shear_vel if mode == "shear" else bt_vel)(u, v, pp)
            return _orig_vti(uu, vv, pp)
        JS._vertical_transport_iface = vti
    s = JS.JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)
    h0 = heatT(T0)
    for k in range(nstep):
        s = JS._step_impl(s, pN)
        if not np.isfinite(np.asarray(s.T)).all():
            print("  %-6s NaN at step %d" % (tag, k + 1)); return
    h1 = heatT(np.asarray(s.T))
    print("  %-6s H0 %.5f  H1 %.5f  dH %+12.6f ZJ  (%.3f ZJ/yr)"
          % (tag, h0, h1, h1 - h0, (h1 - h0) / (nstep * 3600.0 / 3.1536e7)), flush=True)

run("base ", "base")
run("shear", "shear")
JS._vertical_transport_iface = _orig_vti
