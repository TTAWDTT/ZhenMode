"""Stage-by-stage OHC budget of one _step_impl at ckpt_tenyr_ms_gm.

Splits the Strang L/N/L core + barotropic subcycle + polar cap + land mask into
separate contributions so the k=12/13 deep warming can be attributed.
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

step, init_fn, diag, p, terms_fn = JS.make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = 1025.0 * 3992.0 / 1e21 * 8760.0
V = vol.sum()

st = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                  S=jnp.asarray(S0), eta=jnp.asarray(e0))

def dz(Ta, Tb):
    dT = np.asarray(Ta) - np.asarray(Tb)
    rows = np.array([(dT[:, :, k] * vol[:, :, k]).sum() * ZJ for k in range(14)])
    return rows, rows.sum()

HDR = "k   " + " ".join("%7d" % k for k in range(14))

def show(tag, rows, tot):
    print("%-16s" % tag + " ".join("%+7.1f" % r for r in rows) + "  | col %+8.2f" % tot)

print(HDR)
print("-" * 122)

dt_half = p.dt / 2.0
s1 = JS._linear_half_step(st, p, dt_half)
r, t = dz(s1.T, st.T); show("L(dt/2) #1", r, t)
s2 = JS._explicit_full_step(s1, p, p.dt)
r, t = dz(s2.T, s1.T); show("N(dt)", r, t)
s3 = JS._linear_half_step(s2, p, dt_half)
r, t = dz(s3.T, s2.T); show("L(dt/2) #2", r, t)
# barotropic subcycle does not touch T; polar cap + masks do
s4 = JS.JaxStateG(s3.u, s3.v, JS._polar_cap_3d(s3.T, p), s3.S, s3.eta)
r, t = dz(s4.T, s3.T); show("polar_cap_3d", r, t)
s5 = JS.JaxStateG(s4.u, s4.v, s4.T * p.wet_mask_z + st.T * (1.0 - p.wet_mask_z), s4.S, s4.eta)
r, t = dz(s5.T, s4.T); show("land hold", r, t)

TOT = (np.asarray(s5.T) - np.asarray(st.T))
rows = np.array([(TOT[:, :, k] * vol[:, :, k]).sum() * ZJ for k in range(14)])
show("SUM of stages", rows, rows.sum())

real = step(st)
jax.block_until_ready(real)
rr, rt = dz(real.T, st.T)
show("REAL step()", rr, rt)
print("-" * 122)
print("residual (real - sum) column: %+.4f ZJ/yr" % (rt - rows.sum()))

# --- decompose N(dt) tracer residual ---
dT, dS = JS._compute_tracer_residual(s1, p)
dT = np.asarray(dT); dS = np.asarray(dS)
print("\nN-step _compute_tracer_residual per-level (ZJ/yr):")
show("  TOTAL resid", *(lambda x: (x, x.sum()))(np.array([(dT[:, :, k] * vol[:, :, k]).sum() * ZJ * p.dt for k in range(14)])))

tn = terms_fn(s1)
tn = np.asarray(tn)
names = ["adv_T", "diff_h_T", "diff_v_T", "conv_T", "gm_T", "redi_T"]
print("\nterms_fn at L(dt/2) output (ZJ/yr), each x dt:")
for i, nm in enumerate(names):
    rows = np.array([(tn[i][:, :, k] * vol[:, :, k]).sum() * ZJ * p.dt for k in range(14)])
    show("  " + nm, rows, rows.sum())
