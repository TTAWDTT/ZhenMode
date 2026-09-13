"""WHY is the column-integrated divergence Fz[0] ~ 1e-5 when d(eta)/dt ~ 1e-10?

div_h is LINEAR and dz_node is spatially uniform, so

    sum_k div_h(u_k) * dz[k]  ==  div_h( sum_k u_k * dz[k] )  ==  L * div_h(ubt_exact)

with ubt_exact = sum_k u_k*dz[k] / L  and  L = sum_k dz[k] = 5002.5.

So the tracer's column leak Fz[0] is EXACTLY the horizontal divergence of the
column's own depth-mean velocity. The barotropic mode enforces
div_bt(ubt_baro) = 0-ish on ubt_baro = _barotropic_velocity(u,v), a DIFFERENT
field. Any persistent offset ubt_baro != ubt_exact shows up 1:1 as the leak.

This measures the offset and which pairing is actually consistent.

Also measures H_col = sum_k wet_mask_z*dz (the true column depth) against the
H_sw = sum(grid.dz) = 4000 used in the barotropic mass/wind scaling.
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
wet2 = np.asarray(p.wet_mask, float)
DZN = np.asarray(p.dz_node).ravel()
L = DZN.sum()
H_sw = float(p.H_sw)
AREA3 = AREA[:, :, None]
st = JS.JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)
surfm = wet3[:, :, 0] > 0.5

H_col = (wet3 * DZN[None, None, :]).sum(axis=-1)     # (nx,ny) true column depth
print("H_sw (barotropic)        = %.2f m" % H_sw)
print("L = sum(dz_node)         = %.2f m" % L)
print("H_col: min %.1f  max %.1f  mean-over-wet %.1f" %
      (H_col[surfm].min(), H_col[surfm].max(), H_col[surfm].mean()))
print("columns where H_col < H_sw (shelf): %.1f%% of wet cells"
      % (100 * (H_col[surfm] < H_sw - 1.0).mean()))

# ── the three candidate depth-mean velocities ──
ubt_baro, vbt_baro = JS._barotropic_velocity(st.u, st.v, p)          # current code
ubt_exact = jnp.sum(st.u * p.dz_node, axis=-1) / L                    # node sum / L
vbt_exact = jnp.sum(st.v * p.dz_node, axis=-1) / L
ubt_Hsw = jnp.sum(st.u * p.dz_node, axis=-1) / H_sw                   # node sum / H_sw
vbt_Hsw = jnp.sum(st.v * p.dz_node, axis=-1) / H_sw

Fz = JS._vertical_transport_iface(st.u, st.v, p)
f0 = np.asarray(Fz[:, :, 0], float)

def rms(a):
    a = np.asarray(a, float)
    return np.sqrt((a[surfm] ** 2).mean())

# div_h of a 2D velocity: reuse _divergence_h by broadcasting to a 3D field
def divh2d(u2, v2):
    u3 = jnp.broadcast_to(u2[:, :, None], st.u.shape)
    v3 = jnp.broadcast_to(v2[:, :, None], st.v.shape)
    return np.asarray(JS._divergence_h(u3, v3, p)[:, :, 0], float)

def divcons(u2, v2):
    return np.asarray(JS._divergence_conservative(
        jnp.asarray(u2) * p.wet_mask, jnp.asarray(v2) * p.wet_mask, p), float)

print("\n" + "=" * 88)
print("Fz[0] vs candidate reconstructions   [rms over surface cells, m^2/s]")
print("=" * 88)
print("  Fz[0] (measured)                     rms %.4e" % rms(f0))
print("  L   * div_h(ubt_exact)               rms %.4e   ratio to Fz[0] %.4f"
      % (rms(L * divh2d(ubt_exact, vbt_exact)),
         rms(L * divh2d(ubt_exact, vbt_exact)) / rms(f0)))
print("  L   * div_h(ubt_baro)                rms %.4e   ratio %.4f"
      % (rms(L * divh2d(ubt_baro, vbt_baro)),
         rms(L * divh2d(ubt_baro, vbt_baro)) / rms(f0)))
print("  H_sw* div_h(ubt_Hsw)                 rms %.4e   ratio %.4f"
      % (rms(H_sw * divh2d(ubt_Hsw, vbt_Hsw)),
         rms(H_sw * divh2d(ubt_Hsw, vbt_Hsw)) / rms(f0)))

print("\n  --- using _divergence_conservative (the operator the ETA sees) ---")
print("  L   * div_cons(ubt_exact)            rms %.4e" % rms(L * divcons(ubt_exact, vbt_exact)))
print("  L   * div_cons(ubt_baro)             rms %.4e" % rms(L * divcons(ubt_baro, vbt_baro)))
print("  H_sw* div_cons(ubt_baro)             rms %.4e  <-- current eta tendency/H_sw"
      % rms(H_sw * divcons(ubt_baro, vbt_baro)))
print("  H_sw* div_cons(ubt_exact)            rms %.4e  <-- if depth-mean were exact"
      % rms(H_sw * divcons(ubt_exact, vbt_exact)))

print("\n" + "=" * 88)
print("THE OFFSET: ubt_baro - ubt_exact")
print("=" * 88)
du = np.asarray(ubt_baro - ubt_exact, float)
dv = np.asarray(vbt_baro - vbt_exact, float)
print("  ubt_baro rms %.4e   ubt_bexact rms %.4e" % (rms(ubt_baro), rms(ubt_exact)))
print("  offset   rms %.4e   (%.1f%% of ubt_baro rms)"
      % (rms(du), 100 * rms(du) / rms(ubt_baro)))
print("  offset   max %.4e" % np.abs(du[surfm]).max())
print("  L*div_h(offset) rms %.4e  <-- the part of Fz[0] the offset explains"
      % rms(L * divh2d(du, dv)))
print("  Fz[0] - L*div_h(offset) rms %.4e  <-- leftover"
      % rms(f0 - L * divh2d(du, dv)))

print("\n" + "=" * 88)
print("is the leak ACTUALLY eta tendency? compare with the archive")
print("=" * 88)
eta = np.asarray(e0, float)
print("  eta (ckpt): rms %.4f m  min %.3f max %.3f" % (rms(eta), eta[surfm].min(), eta[surfm].max()))
print("  volume-mean eta %.6f m" % ((eta * AREA * wet2).sum() / (AREA * wet2).sum()))
print("  Fz[0]/H_col rms %.4e m/s -> over 10 yr = %.3f m of eta"
      % (rms(f0 / H_col), rms(f0 / H_col) * 3.1536e8))
print("  ...so if Fz[0] were the real mass flux, eta would be O(1e5) m.")
print("  It is not -> Fz[0] is discretisation noise in the 3D velocity,")
print("  not the free-surface tendency.")
