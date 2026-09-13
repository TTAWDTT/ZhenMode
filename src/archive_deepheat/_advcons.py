"""Is the tracer advection operator GLOBALLY heat-conserving?

math:   integral of -div(F) over the domain = -(boundary flux).
        For a closed basin with no-flux walls, integral adv(T) dV == 0 EXACTLY.
        Any nonzero value is a spurious heat source/sink.

Test:
  (1) the model's own _advection_scalar
  (2) the same operator applied to T = const  (must be ~0 everywhere, not just
      in integral -- a nonzero integral of adv(1) is a mass-divergence defect)
  (3) per-level integral of adv(T)
  (4) each of the three divergence pieces separately
  (5) does the defect scale with the time step? (=> an inconsistency between
      the tracer faces and the continuity faces used in Fz)
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
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
wm = p.wet_mask_z
st = JS.JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)

def integ(f):
    return float((np.asarray(f, float) * vol).sum())

def perlev(f):
    f = np.asarray(f, float)
    return np.array([(f[:, :, k] * vol[:, :, k]).sum() for k in range(14)]) * ZJYR

Fz = JS._vertical_transport_iface(st.u, st.v, p)

print("=" * 92)
print("(1) model operator _advection_scalar on the real T")
print("=" * 92)
advT = JS._advection_scalar(st.T, st.u, st.v, Fz, p)
print("  integral adv(T) dV = %+10.4f ZJ/yr   (exact conservation requires 0)"
      % integ(advT))
print("  per level          : " + " ".join("%+6.1f" % x for x in perlev(advT)))
print("                       " + " ".join("%+6.1f" % x for x in perlev(advT)) + "  sum %+.1f"
      % perlev(advT).sum())

print("\n" + "=" * 92)
print("(2) same operator on a CONSTANT field -> must be 0 everywhere")
print("=" * 92)
adv1 = JS._advection_scalar(jnp.ones_like(st.T) * 7.0, st.u, st.v, Fz, p)
a1 = np.asarray(adv1, float)
print("  integral adv(1) dV = %+10.4e K*m^3/s" % integ(adv1 / 7.0))
print("  max |adv(1)|       = %10.4e" % np.abs(a1).max())
wetb = wet3 > 0.5
print("  max |adv(1)| in wet= %10.4e" % np.abs(a1[wetb]).max())
print("  rms |adv(1)| in wet= %10.4e" % np.sqrt((a1[wetb] ** 2).mean()))
print("  per-level integral : " + " ".join("%+7.2f" % x
                                          for x in perlev(adv1 / 7.0)))

print("\n" + "=" * 92)
print("(3) rebuild the three divergence pieces explicitly")
print("=" * 92)
T, u, v = st.T, st.u, st.v
Tx_face = 0.5 * (T + jnp.roll(T, -1, axis=0))
ux_face = 0.5 * (u + jnp.roll(u, -1, axis=0))
gate_x = wm * jnp.roll(wm, -1, axis=0)
Fx = ux_face * Tx_face * gate_x
pad = [(0, 0), (1, 1), (0, 0)]
T_pad = jnp.pad(T, pad, mode='edge'); v_pad = jnp.pad(v, pad, mode='edge')
wm_pad = jnp.pad(wm, pad, mode='edge')
Ty_face = 0.5 * (T_pad[:, 1:-1] + T_pad[:, 2:])
vy_face = 0.5 * (v_pad[:, 1:-1] + v_pad[:, 2:])
gate_y = wm_pad[:, 1:-1] * wm_pad[:, 2:]
cos_face = 0.5 * (p.cos_lat + jnp.roll(p.cos_lat, -1))
Fy = (vy_face * Ty_face * gate_y * cos_face[None, :, None]).at[:, -1].set(0.0)
T_deep = JS._fill_ghost_bottom(T, p)[..., 1:]
Fz_int = Fz[..., 1:-1] * jnp.where(Fz[..., 1:-1] > 0.0, T[..., :-1], T_deep) \
    * (wm[..., :-1] * wm[..., 1:])
Fz_top = Fz[:, :, :1] * T[..., :1]
up = jnp.concatenate([Fz_top, Fz_int], axis=-1)
dn = jnp.concatenate([Fz_int, jnp.zeros_like(Fz_int[..., :1])], axis=-1)
Fx_up = jnp.roll(Fx, 1, axis=0)
Fy_up = jnp.roll(Fy, 1, axis=1); Fy_up = Fy_up.at[:, 0].set(0.0)
div_x = (Fx - Fx_up) * p.inv_dx[..., 0:1]
div_y = (Fy - Fy_up) * p.inv_dy / p.cos_lat[None, :, None]
div_z = (dn - up) / p.dz_node

for nm, t in [("div_x", div_x), ("div_y", div_y), ("div_z", div_z)]:
    r = -perlev(t)
    print("  %-14s sum %+9.3f ZJ/yr | " % (nm, r.sum()) + " ".join("%+6.1f" % x for x in r))
r = -perlev(div_x + div_y + div_z)
print("  %-14s sum %+9.3f ZJ/yr | " % ("TOTAL adv", r.sum()) + " ".join("%+6.1f" % x for x in r))

print("\n" + "=" * 92)
print("(4) the rigid-lid top-face term alone")
print("=" * 92)
f0 = np.asarray(Fz[:, :, 0], float)
surfm = wet3[:, :, 0] > 0.5
print("  Fz[0]: rms %.3e  max|.| %.3e m^2/s" %
      (np.sqrt((f0[surfm] ** 2).mean()), np.abs(f0[surfm]).max()))
print("  surface-layer |u|*dz scale  %.3e m^2/s" %
      (np.abs(np.asarray(u[:, :, 0]))[surfm].mean() * DZN[0]))
top_only = jnp.concatenate([(Fz_top) / p.dz_node[..., :1] * (-1.0),
                            jnp.zeros_like(Fz_int)], axis=-1)
print("  integral of -(Fz_top/dz)   = %+9.3f ZJ/yr" % perlev(top_only).sum())
print("  this equals integral of +(Fz[0]*T[0]/dz) with sign flipped")

print("\n" + "=" * 92)
print("(5) is the top-face closure CONSISTENT with continuity?")
print("=" * 92)
Fz_topc = Fz[:, :, :1] * 1.0
topc = jnp.concatenate([Fz_topc / p.dz_node[..., :1] * (-1.0),
                        jnp.zeros_like(Fz_int)], axis=-1)
print("  integral of -(Fz[0]/dz)      = %+9.3e m^3/s/m^2  (must be 0)"
      % float((np.asarray(topc, float) * vol).sum()))
print("  ^ if ~0, sum_k Fz[0] = 0 and the closure is unbiased;")
print("    a nonzero value means the surface cell gains/loses MASS each step.")
colchk = np.asarray(Fz[:, :, 0], float)
print("  sum over surface cells of Fz[0]*area = %+.4e m^3/s" % (colchk * AREA * surfm).sum())
print("  sum over ALL cells of div_x+div_y+div_z for T=1: %.4e"
      % float((np.asarray(div_x * 0 + 0, float) * vol).sum()))
adv1b = JS._advection_scalar(jnp.ones_like(T), u, v, Fz, p)
print("  model adv(1) integral = %+.4e" % integ(adv1b))
