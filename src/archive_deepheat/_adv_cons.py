"""Is the tracer advection operator conservative?

A flux-form advection -div(u T) must integrate to zero over the domain for ANY
T if and only if the discrete velocity field is divergence-free under the same
face/weight discretization. Test by advecting T=1 (wet) everywhere: the result
is -div(u), which must be zero.
Then quantify the spurious source term T*div(u) on the real fields.
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

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
T0 = np.asarray(d["T"], np.float64); S0 = np.asarray(d["S"], np.float64)
u0 = np.asarray(d["u"], np.float64); v0 = np.asarray(d["v"], np.float64)
e0 = np.asarray(d["eta"], np.float64)
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])

phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)
step, _, _, p, terms_fn = JS.make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = 1025.0 * 3992.0 / 1e21 * 8760.0

st = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                  S=jnp.asarray(S0), eta=jnp.asarray(e0))
Fz = JS._vertical_transport_iface(st.u, st.v, p)

def colsum(field):
    f = np.asarray(field)
    return np.array([(f[:, :, k] * vol[:, :, k]).sum() * ZJ for k in range(14)])

print("=" * 92)
print("A. adv(T=1) should be -div(u) = 0 exactly")
print("=" * 92)
ones = jnp.asarray(wet3)
a1 = JS._advection_scalar(ones, st.u, st.v, Fz, p)
print("adv(1) raw stats: min %+.3e max %+.3e  rms %.3e"
      % (float(np.asarray(a1).min()), float(np.asarray(a1).max()),
         float(np.sqrt((np.asarray(a1) ** 2).mean()))))
r = colsum(a1)
print("per-level column sum  : " + " ".join("%+8.2e" % x for x in r))
print("DOMAIN TOTAL          : %+.4f ZJ/yr  (must be 0)" % r.sum())

print("\n" + "=" * 92)
print("B. adv on the real T: level profile and net creation")
print("=" * 92)
aT = JS._advection_scalar(st.T, st.u, st.v, Fz, p)
rT = colsum(aT)
print("adv_T per level (ZJ/yr): " + " ".join("%+8.1f" % x for x in rT))
print("adv_T DOMAIN TOTAL     : %+.4f ZJ/yr" % rT.sum())

print("\n" + "=" * 92)
print("C. where is the divergence? split by term")
print("=" * 92)
wm = p.wet_mask_z
T = st.T; u = st.u; v = st.v
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
Fy = vy_face * Ty_face * gate_y * cos_face[None, :, None]
Fy = Fy.at[:, -1].set(0.0)
Fz_in = Fz
T_deep = JS._fill_ghost_bottom(T, p)[..., 1:]
T_shallow = T[..., :-1]
wet_iface = wm[..., :-1] * wm[..., 1:]
Fz_int = Fz_in[..., 1:-1] * jnp.where(Fz_in[..., 1:-1] > 0.0, T_shallow, T_deep) * wet_iface
Fz_top = Fz_in[:, :, :1] * T[..., :1]
up = jnp.concatenate([Fz_top, Fz_int], axis=-1)
dn = jnp.concatenate([Fz_int, jnp.zeros_like(Fz_int[..., :1])], axis=-1)
Fx_up = jnp.roll(Fx, 1, axis=0)
Fy_up = jnp.roll(Fy, 1, axis=1); Fy_up = Fy_up.at[:, 0].set(0.0)
div_x = (Fx - Fx_up) * p.inv_dx[..., 0:1]
div_y = (Fy - Fy_up) * p.inv_dy / p.cos_lat[None, :, None]
div_z = (dn - up) / p.dz_node
for nm, term in [("div_x", div_x), ("div_y", div_y), ("div_z", div_z)]:
    rr = colsum(term)
    print("%-8s per level: " % nm + " ".join("%+8.1f" % x for x in rr))
    print("%-8s TOTAL    : %+.4f ZJ/yr" % (nm, rr.sum()))

print("\n" + "=" * 92)
print("D. mass divergence of the advecting field (no T): sum of -div(u) weighted")
print("=" * 92)
# mass flux divergence: same faces but flux = velocity only
Fx_m = ux_face * gate_x
Fy_m = vy_face * gate_y * cos_face[None, :, None]
Fy_m = Fy_m.at[:, -1].set(0.0)
upm = jnp.concatenate([Fz_in[:, :, :1], Fz_in[..., 1:-1] * wet_iface], axis=-1)
dnm = jnp.concatenate([Fz_in[..., 1:-1] * wet_iface, jnp.zeros_like(Fz_in[..., :1])], axis=-1)
dxm = (Fx_m - jnp.roll(Fx_m, 1, axis=0)) * p.inv_dx[..., 0:1]
dym = (Fy_m - (lambda z: z.at[:, 0].set(0.0))(jnp.roll(Fy_m, 1, axis=1))) * p.inv_dy / p.cos_lat[None, :, None]
dzm = (dnm - upm) / p.dz_node
massdiv = dxm + dym + dzm
print("mass div stats: min %+.3e max %+.3e rms %.3e" %
      (float(massdiv.min()), float(massdiv.max()), float(jnp.sqrt((massdiv**2).mean()))))
rm = colsum(-massdiv)
print("-massdiv per level: " + " ".join("%+8.1f" % x for x in rm))
print("-massdiv TOTAL    : %+.4f ZJ/yr" % rm.sum())
# mass imbalance per level: this is div(u)*1 which should be 0
print("\nlevel-integrated mass divergence (should be 0 per column):")
print("  max abs column sum of massdiv: %.3e" % float(np.abs(np.asarray(massdiv).sum(axis=2)*wet3[:,:,0]).max()))
