"""Is Fz[0] (the column-integrated horizontal divergence of the 3D velocity)
the physical free-surface signal, or mode-split projection noise?

If it is noise, correcting the advecting velocity to have the barotropic
column divergence should remove the +66.58 ZJ/yr spurious source.

Test: build u_c = u + (correction), where the correction is the depth-uniform
delta that makes sum_k div_h(u_c)*dz equal the target column divergence.
Compare three targets:
  T0: leave as is (baseline)
  T1: target = 0 (rigid-lid closure)
  T2: target = the true physical column divergence
Then integrate adv(T) and see the column source.
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
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)
step, _, _, p, _ = JS.make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)

st = JS.JaxStateG(u=jnp.asarray(d["u"], np.float64), v=jnp.asarray(d["v"], np.float64),
                  T=jnp.asarray(d["T"], np.float64), S=jnp.asarray(d["S"], np.float64),
                  eta=jnp.asarray(d["eta"], np.float64))
wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
surf = wet3[:, :, 0] > 0.5
AREAZ = AREA[:, :, None]
ZJYR = 1025.0 * 3992.0 / 1e21 * 3.1536e7

def cs(f):
    f = np.asarray(f)
    return np.array([(f[:, :, k] * wet3[:, :, k] * AREA * DZN[k]).sum() * ZJYR for k in range(14)])

# barotropic divergence operator applied to a depth-uniform correction:
# div_h(corr) is linear, so solve for delta_ubt that yields a target column sum.
divh_u = np.asarray(JS._divergence_h(st.u, st.v, p))
Fz0 = (divh_u * DZN[None, None, :]).sum(axis=2)
print("Fz[0] rms %.4e  (m^2/s)" % np.sqrt((Fz0[surf] ** 2).mean()))

# eta tendency from one step
s1 = step(st); jax.block_until_ready(s1)
deta_dt = (np.asarray(s1.eta) - np.asarray(st.eta)) / float(p.dt)

print("\nWhat the column divergence SHOULD be:")
print("  -deta/dt           rms %.4e   corr with Fz[0] %+.4f"
      % (np.sqrt((deta_dt[surf] ** 2).mean()), np.corrcoef(Fz0[surf], -deta_dt[surf])[0, 1]))
H = np.asarray(p.H_sw, float)
print("  -H*div(ubt)        rms %.4e" % np.sqrt(((H * np.asarray(JS._divergence_conservative(
    *JS._barotropic_velocity(st.u, st.v, p), p)))[surf] ** 2).mean()))

# --- how much of Fz[0] is barotropic vs baroclinic? ---
ubt, vbt = JS._barotropic_velocity(st.u, st.v, p)
# baroclinic residual: u' = u - ubt (uniform in depth)
up = st.u - ubt[:, :, None]
vp = st.v - vbt[:, :, None]
Fz0_bar = np.asarray((JS._divergence_h(up, vp, p) * DZN[None, None, :]).sum(axis=2))
print("\n  Fz[0] from full u,v        rms %.4e" % np.sqrt((Fz0[surf] ** 2).mean()))
print("  Fz[0] from baroclinic u'   rms %.4e" % np.sqrt((Fz0_bar[surf] ** 2).mean()))
# barotropic column divergence: H * div(ubt) computed by the 2D operator
div_bt = np.asarray(JS._divergence_conservative(ubt, vbt, p))
print("  H*div_2d(ubt)              rms %.4e" % np.sqrt(((H * div_bt)[surf] ** 2).mean()))

# --- direct test: what if we simply remove the whole column divergence from u,v? ---
# Build a divergence-free-in-column correction: solve div_h(corr)= -Fz0/H roughly
# Practical: set Fz_top in the advection to 0 by removing the k=0 top-face term.
aT = np.asarray(JS._advection_scalar(st.T, st.u, st.v,
                 JS._vertical_transport_iface(st.u, st.v, p), p))
print("\nadv_T column total WITH top face : %+.4f ZJ/yr" % cs(aT).sum())
print("adv_T per level                  : " + " ".join("%+7.1f" % x for x in cs(aT)))

# Now: mass-consistent advection. Replace the advecting velocity by one whose
# column-integrated divergence is zero, keeping the depth mean of horizontal
# divergence: subtract a linear-in-k profile of horizontal divergence.
# Approximate by distributing the column divergence uniformly: Fz is then
# rebuilt on the corrected velocity.
target = np.zeros_like(Fz0)                # rigid-lid target
delta = target - Fz0                       # m^2/s of extra column transport
# Add a uniform-in-depth horizontal velocity whose divergence gives `delta`.
# div_h is a discrete operator; approximate correction by scaling the local
# depth-integrated horizontal transport. Use the 2D gradient/divergence pair:
corr_eta = np.asarray(JS._gradient_conservative(jnp.asarray(delta), p)) if hasattr(JS, "_gradient_conservative") else None
print("\n(correction via 2D gradient/divergence adjoint pair)")
if corr_eta is not None:
    cx, cy = corr_eta
    # div(-grad(phi)) ~ laplacian; we want div_h(u_c) column = delta/H. Solve
    # with a few Jacobi sweeps.
    phi = np.zeros_like(delta)
    lap = JS._laplacian_h_2d
    b = delta / np.where(H > 0, H, 1.0)
    for it in range(2000):
        L = np.asarray(lap(jnp.zeros_like(jnp.asarray(phi)), jnp.zeros_like(jnp.asarray(phi)), p))
        break
    print("  (Jacobi solve skipped - use direct velocity dump instead)")

# DECISIVE: dump the divergence of the velocity BEFORE vs AFTER the barotropic
# projection inside one step, to see if the projection creates Fz[0].
print("\n" + "=" * 96)
print("PROJECTION TEST: column divergence of the L/N/L output vs the step output")
print("=" * 96)
dt_half = p.dt / 2.0
a = JS._linear_half_step(st, p, dt_half)
b = JS._explicit_full_step(a, p, p.dt)
c = JS._linear_half_step(b, p, dt_half)

def coldiv(s, tag):
    F = np.asarray((JS._divergence_h(s.u, s.v, p) * DZN[None, None, :]).sum(axis=2))
    print("%-28s Fz[0] rms %.4e  max %+.4e" % (tag, np.sqrt((F[surf] ** 2).mean()), np.abs(F[surf]).max()))
    return F

F_in = coldiv(st, "input state")
F_core = coldiv(c, "after L/N/L core")
# barotropic subcycle
F_rho_x, F_rho_y = JS._compute_bt_rho_pgf(c, p)
ubt0, vbt0 = JS._barotropic_velocity(c.u, c.v, p)
eta, ub, vb = c.eta, ubt0, vbt0
for _ in range(int(p.n_subcyc)):
    eta, ub, vb = JS._free_surface_step_fd(eta, ub, vb, p, F_rho_x, F_rho_y, dt_half=p.dt_bt)
proj = JS.JaxStateG(c.u + (ub - ubt0)[:, :, None], c.v + (vb - vbt0)[:, :, None], c.T, c.S, eta)
F_proj = coldiv(proj, "after bt projection")
print("\nchange in Fz[0] from projection: rms %.4e" % np.sqrt(((F_proj - F_core)[surf] ** 2).mean()))
print("corr(F_core, F_proj) = %+.4f" % np.corrcoef(F_core[surf], F_proj[surf])[0, 1])
print("RATIO rms(F_proj)/rms(F_core) = %.3f" % (np.sqrt((F_proj[surf]**2).mean()) / np.sqrt((F_core[surf]**2).mean())))
