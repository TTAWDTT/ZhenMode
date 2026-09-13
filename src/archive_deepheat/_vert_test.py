"""Is the deep warming driven by a spurious vertical velocity?

Fz[0] (column-integrated horizontal divergence of the 3D velocity) has rms
1.28e-5 m^2/s but the true free-surface signal -deta/dt has rms 3.5e-7 --
36x smaller, correlation +0.018.  So the 3D velocity carries a large
non-divergence-free component, and _vertical_transport_iface converts that
straight into vertical mass transport w*area.

Ablations, one step at ckpt_tenyr_ms_gm:
  baseline
  Fz=0        no vertical advective transport (Fz_in zeroed)
  adv=0       u=v=0 (Fz is built from u,v so it also goes to zero)
  conv=0      kappa_conv=0
  Fz=0+conv=0
  kappa_v=0
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
BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)
phys = replace(PhysicsConfig(), **BASE)

# patch source so _advection_scalar accepts a zero Fz without touching u,v
src = open("src/jax_solver_global.py", encoding="utf-8").read()
OLD = "    Fz_int = Fz_in[..., 1:-1] * jnp.where(Fz_in[..., 1:-1] > 0.0, T_shallow, T_deep) * wet_iface"
assert src.count(OLD) == 1
NEW = ("    Fz_int = Fz_in[..., 1:-1] * jnp.where(Fz_in[..., 1:-1] > 0.0, T_shallow, T_deep)"
       " * wet_iface * p.adv_vert_gate")
assert src.count("p.adv_vert_gate") == 0
src2 = src.replace(OLD, NEW.replace("p.adv_vert_gate", "0.0"))
OLD2 = "    Fz_top = Fz_in[:, :, :1] * T[..., :1]"
assert src2.count(OLD2) == 1
src2 = src2.replace(OLD2, "    Fz_top = Fz_in[:, :, :1] * T[..., :1] * 0.0")
open("_js_advgate.py", "w", encoding="utf-8").write(src2)
import _js_advgate as JSA

wet3_ = None

def build(mod, phys_):
    step, _, _, p, terms = mod.make_solver_global(
        g, phys_, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
        mode_split=True, dt_bt=300.0, return_params=True)
    return step, p, terms

step0, p, terms0 = build(JS, phys)
stepG, pG, termsG = build(JSA, phys)
wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
SPY = 8760.0
ZJ = 1025.0 * 3992.0 / 1e21 * SPY
T0 = np.asarray(d["T"], np.float64); S0 = np.asarray(d["S"], np.float64)
st = JS.JaxStateG(u=jnp.asarray(d["u"], np.float64), v=jnp.asarray(d["v"], np.float64),
                  T=jnp.asarray(T0), S=jnp.asarray(S0),
                  eta=jnp.asarray(d["eta"], np.float64))

def cs(f):
    f = np.asarray(f)
    return np.array([(f[:, :, k] * vol[:, :, k]).sum() * ZJ for k in range(14)])

print("k                        " + " ".join("%7d" % k for k in range(14)))
rows = {}
def run(tag, step, s):
    s1 = step(s); jax.block_until_ready(s1)
    r = cs(np.asarray(s1.T) - T0)
    rows[tag] = r
    print("%-22s" % tag + " ".join("%+7.1f" % x for x in r) + " | %+8.2f" % r.sum())

run("baseline", step0, st)
run("Fz=0 (no vert adv)", stepG, st)
zero = JS.JaxStateG(jnp.zeros_like(st.u), jnp.zeros_like(st.v), st.T, st.S, st.eta)
run("u=v=0", step0, zero)

for tag, kw in [("conv=0", dict(kappa_conv=0.0)),
                ("kappa_v=0", dict(kappa_v=0.0)),
                ("gm=redi=0", dict(kappa_gm=0.0, kappa_redi=0.0))]:
    pa = replace(phys, **kw)
    sp, _, _ = build(JS, pa)
    run(tag, sp, st)

sp2, _, _ = build(JS, replace(phys, kappa_conv=0.0))
# conv=0 in the gated module
stepGc, _, _ = build(JSA, replace(phys, kappa_conv=0.0))
run("Fz=0 + conv=0", stepGc, st)

print()
for tag in ["Fz=0 (no vert adv)", "u=v=0", "conv=0", "kappa_v=0", "gm=redi=0", "Fz=0 + conv=0"]:
    print("delta %-20s" % tag + " ".join("%+7.1f" % x for x in (rows[tag] - rows["baseline"]))
          + " | %+8.2f" % (rows[tag].sum() - rows["baseline"].sum()))

print("\n" + "=" * 96)
print("40-step drift for the two strongest levers")
print("=" * 96)
for tag, step in [("baseline", step0), ("Fz=0", stepG)]:
    s = st
    for _ in range(40):
        s = step(s)
    jax.block_until_ready(s)
    Tn = np.asarray(s.T)
    print("%-10s min %.4f max %.4f nan %d  dOHC %+.4f ZJ  deepT(mean k>=11) %.4f"
          % (tag, Tn.min(), Tn.max(), int(np.isnan(Tn).sum()),
             ((Tn - T0) * vol).sum() * 1025 * 3992 / 1e21,
             (Tn[:, :, 11:] * vol[:, :, 11:]).sum() / vol[:, :, 11:].sum()))
print("initial deepT(mean k>=11) %.4f" % ((T0[:, :, 11:] * vol[:, :, 11:]).sum() / vol[:, :, 11:].sum()))
