"""Test the column-mean-divergence-removal fix.

Diagnosis so far (see docs/deep-heat-poisoning-root-cause.md):
  integral adv(T) dV == -sum_cells area * Fz[0] * T[0]   (exact identity)
  integral adv(1) dV == -sum_cells area * Fz[0]          (== 0 to 1e-9)

so the operator is pointwise mass-consistent but heat-leaky. Fz[0] is the
COLUMN-INTEGRATED horizontal divergence -- zero for a rigid lid, because
  int_k div_h[k] dz[k] = -(w_top - w_bot) = -w_top.
Since sum(area*Fz[0]) ~ 0 but |Fz[0]| is large, Fz[0] is a local numerical
residual of the 3D field's column divergence, not a physical d(eta)/dt
(which would be ~1e-10 m/s, not 1e-5).

Every LOCAL top-face closure fails because we need simultaneously
  sum area*Fz[0]*T_top = 0   (heat)  and  T_top = 1 gives the mass flux (adv(1)=0)
and sum(area*Fz[0]) ~ 0 makes the required T_top ill-defined.

THE FIX: project the divergence onto the column-mean-free subspace BEFORE
building Fz, i.e. remove the barotropic part of div_h:

    div_h'[k] = div_h[k] - (sum_m div_h[m]*dz[m]) / (sum_m dz[m])

Then sum_k div_h'[k]*dz[k] == 0 => Fz'[0] == 0, the top face carries no
transport at all (w=0 at the surface, the classical rigid-lid condition),
and heat is conserved by construction. The free surface keeps its own
budget (eta from _divergence_conservative); this only makes the TRACER
transport see a rigid-lid-consistent, divergence-free velocity field.

Variants:
  base   current code (Fz from raw _divergence_h, Fz_top = Fz[0]*T[0])
  proj   div_h' = div_h - column mean; Fz'[0] = 0
  projH  same but the column weight is H_sw-normalised dz_norm (control)
  scale  div_h' = div_h * (1 - Fz[0]/(sum div_h dz))  (control, nonlocal)
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
H_sw = float(p.H_sw)
st = JS.JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)

def heat_rate(adv):
    return float((np.asarray(adv, float) * vol).sum()) * RHO_0 * C_P * 3.1536e7 / 1e21

div_h = JS._divergence_h(st.u, st.v, p)
DZNj = p.dz_node                                    # (1,1,nz)
colint = jnp.sum(div_h * DZNj, axis=-1, keepdims=True)      # (nx,ny,1) == Fz[0]
colsum = jnp.sum(DZNj)                                      # 5002.5
div_proj = div_h - colint / colsum                          # zero column integral

def Fz_from(divfield):
    integrand = divfield * DZNj
    Fz_int = jnp.cumsum(integrand[..., ::-1], axis=-1)[..., ::-1]
    return jnp.concatenate([Fz_int, jnp.zeros_like(Fz_int[..., :1])], axis=-1)

Fz_base = JS._vertical_transport_iface(st.u, st.v, p)
Fz_proj = Fz_from(div_proj)

f0b = np.asarray(Fz_base[:, :, 0], float)
f0p = np.asarray(Fz_proj[:, :, 0], float)
surfm = wet3[:, :, 0] > 0.5
print("H_sw = %.1f   sum(dz_node) = %.1f  <-- these differ by %.1f%%"
      % (H_sw, colsum, 100 * (colsum / H_sw - 1)))
print("Fz[0] base: rms %.4e  max %.4e" % (np.sqrt((f0b[surfm] ** 2).mean()),
                                           np.abs(f0b[surfm]).max()))
print("Fz[0] proj: rms %.4e  max %.4e" % (np.sqrt((f0p[surfm] ** 2).mean()),
                                           np.abs(f0p[surfm]).max()))
print("physical d(eta)/dt scale: eta drift ~0.1 m / 10 yr = %.2e m/s"
      % (0.1 / 3.1536e8))

print("\n" + "=" * 74)
print("integral adv(T) dV  [ZJ/yr; exact conservation requires 0]")
print("=" * 74)
for nm, Fz in [("base  (raw _divergence_h)", Fz_base),
               ("proj  (column-mean removed)", Fz_proj)]:
    adv = JS._advection_scalar(st.T, st.u, st.v, Fz, p)
    a1 = np.asarray(JS._advection_scalar(jnp.ones_like(st.T), st.u, st.v, Fz, p), float)
    r = np.array([(np.asarray(adv, float)[:, :, k] * vol[:, :, k]).sum()
                  for k in range(14)]) * RHO_0 * C_P * 3.1536e7 / 1e21
    print("  %-30s heat %+9.4f ZJ/yr | max|adv(1)| %.3e" % (nm, heat_rate(adv),
                                                             np.abs(a1[wet3 > 0.5]).max()))
    print("      per level: " + " ".join("%+6.1f" % x for x in r))

print("\n" + "=" * 74)
print("also check: does the projection change w materially?")
print("=" * 74)
w_base = np.asarray(-Fz_base[..., :-1] * p.wet_mask_z, float)
w_proj = np.asarray(-Fz_proj[..., :-1] * p.wet_mask_z, float)
m = wet3 > 0.5
print("  w base rms %.4e  max %.4e m/s" % (np.sqrt((w_base[m] ** 2).mean()),
                                            np.abs(w_base[m]).max()))
print("  w proj rms %.4e  max %.4e m/s" % (np.sqrt((w_proj[m] ** 2).mean()),
                                            np.abs(w_proj[m]).max()))
print("  rms difference %.4e   (fraction of base rms %.3f%%)"
      % (np.sqrt(((w_base - w_proj)[m] ** 2).mean()),
         100 * np.sqrt(((w_base - w_proj)[m] ** 2).mean()) / np.sqrt((w_base[m] ** 2).mean())))
