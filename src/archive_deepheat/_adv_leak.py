"""Localize the +66 ZJ/yr spurious source in the tracer advection operator.

adv(1) == 0 exactly => all boundary closures telescope. But adv(T) does NOT
integrate to zero => the advecting velocity is not discretely divergence-free
under the same face/weight discretization. Find which term and which level.
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
ZJYR = 1025.0 * 3992.0 / 1e21 * 3.1536e7

def cs(field):
    f = np.asarray(field)
    return np.array([(f[:, :, k] * vol[:, :, k]).sum() * ZJYR for k in range(14)])

st = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                  S=jnp.asarray(S0), eta=jnp.asarray(e0))
T, u, v = st.T, st.u, st.v
wm = p.wet_mask_z
Fz = JS._vertical_transport_iface(u, v, p)

print("=" * 100)
print("A. the rigid-lid leak Fz[0] (column-integrated horizontal divergence)")
print("=" * 100)
Fz0 = np.asarray(Fz[:, :, 0])
surf = wet3[:, :, 0] > 0.5
m = Fz0 * surf
print("Fz[0] over wet cols: rms %.3e  min %+.3e  max %+.3e   (m^2/s)"
      % (np.sqrt((m ** 2).mean()), m.min(), m.max()))
print("domain sum Fz[0]*area : %+.4e m^3/s  (should be ~0 for steady eta)"
      % (m * AREA * surf).sum())
leak = (m * AREA * surf).sum() / (AREA * surf).sum()
print("area-mean Fz[0]       : %+.4e m^2/s" % leak)
# how big is the leak vs the horizontal transports?
print("typical |u|*dz of surface layer: %.3e m^2/s" % (np.abs(np.asarray(u[:, :, 0])).mean() * DZN[0]))

print("\ntop-face heat source  sum(area*Fz[0]*T0): %+.4f ZJ/yr  (ZJYR units)"
      % float((m * AREA * surf * T0[:, :, 0]).sum() * ZJYR * 1025 * 3992 / 1e21 * 3.1536e7 / (1025 * 3992 / 1e21 * 3.1536e7)))
src = float((m * AREA * surf * T0[:, :, 0]).sum()) * 1025.0 * 3992.0 / 1e21 * 3.1536e7
print("   = %+.4f ZJ/yr" % src)
print("   and with T'=T0-Tbar: %+.4f ZJ/yr"
      % (float((m * AREA * surf * (T0[:, :, 0] - T0[:, :, 0][surf].mean())).sum())
         * 1025.0 * 3992.0 / 1e21 * 3.1536e7))

print("\n" + "=" * 100)
print("B. exact operator decomposition (per level, ZJ/yr)")
print("=" * 100)
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
Fz_int = Fz[..., 1:-1] * jnp.where(Fz[..., 1:-1] > 0.0, T[..., :-1], T_deep) * (wm[..., :-1] * wm[..., 1:])
Fz_top = Fz[:, :, :1] * T[..., :1]
up = jnp.concatenate([Fz_top, Fz_int], axis=-1)
dn = jnp.concatenate([Fz_int, jnp.zeros_like(Fz_int[..., :1])], axis=-1)
div_x = (Fx - jnp.roll(Fx, 1, axis=0)) * p.inv_dx[..., 0:1]
Fy_up = jnp.roll(Fy, 1, axis=1); Fy_up = Fy_up.at[:, 0].set(0.0)
div_y = (Fy - Fy_up) * p.inv_dy / p.cos_lat[None, :, None]
div_z = (dn - up) / p.dz_node

for nm, term in [("div_x", div_x), ("div_y", div_y), ("div_z", div_z),
                 ("div_z TOPFACE only", jnp.concatenate(
                     [(Fz_top - jnp.zeros_like(Fz_top)) / p.dz_node[..., :1],
                      jnp.zeros_like(Fz_int)], axis=-1))]:
    r = -cs(term)
    print("%-20s " % nm + " ".join("%+7.1f" % x for x in r) + " | %+8.2f" % r.sum())

tot = -cs(div_x + div_y + div_z)
print("%-20s " % "TOTAL adv_T" + " ".join("%+7.1f" % x for x in tot) + " | %+8.2f" % tot.sum())

print("\n" + "=" * 100)
print("C. column-integrated mass divergence by level (m^3/s), should be 0 in interior")
print("=" * 100)
massdiv = (div_x + div_y + div_z)
md = np.asarray(massdiv)
colmd = md.sum(axis=2)          # per column
print("column mass-div sum: rms %.3e max %.3e" % (np.sqrt((colmd[surf] ** 2).mean()),
                                                  np.abs(colmd[surf]).max()))
# vertical profile of |column horizontal divergence| at each level
for k in range(14):
    hk = np.asarray(div_x[:, :, k] + div_y[:, :, k])
    print("  k=%2d  |div_h| mean %.3e   |div_z| mean %.3e"
          % (k, np.abs(hk[wet3[:, :, k] > .5]).mean(),
             np.abs(np.asarray(div_z[:, :, k])[wet3[:, :, k] > .5]).mean()))
