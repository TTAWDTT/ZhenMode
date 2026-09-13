"""Decisive: which STRUCTURAL (non-closure) step warms the abyss?

Established: k13 dT is invariant (+-3e-5) to every mixing closure AND to a 4x
change in bulk coupling. The surface moves +-0.25 K while k13 does not budge.
That decoupling is not physical -> a numerical source.

Remaining non-closure candidates, tested 200 steps from ckpt_tenyr_ms_gm:
  ALLMIX0        all 7 mixing closures off
  ALLMIX0+NADV   + vertical advective flux Fz_int := 0
  ALLMIX0+NOADV  + both horizontal and vertical advection off
  cap0           polar_cap_rows=0 (the per-step zonal-average filter)
  ALLMIX0+cap0   both
  monopole       mode_split=False (monolithic free surface every step)

Also: direct measurement of the column heat leak from advection, using the
module's own Fz_top and Fz_in[0] (the rigid-lid divergence the barotropic
subcycle is supposed to absorb).
"""
import sys, types
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
ST0 = JS.JaxStateG(u=jnp.asarray(np.asarray(d["u"], np.float64)),
                   v=jnp.asarray(np.asarray(d["v"], np.float64)),
                   T=jnp.asarray(np.asarray(d["T"], np.float64)),
                   S=jnp.asarray(np.asarray(d["S"], np.float64)),
                   eta=jnp.asarray(np.asarray(d["eta"], np.float64)))
T0_np = np.asarray(d["T"], float)
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)
NOMIX = dict(kappa_conv=0.0, kappa_h=0.0, kappa_v=0.0, kappa_gm=0.0,
             kappa_redi=0.0, nu_bi=0.0, kappa_bi=0.0)

SRC = open("src/jax_solver_global.py", encoding="utf-8").read()
def variant(tag, old, new):
    assert SRC.count(old) == 1, "%s: %d" % (tag, SRC.count(old))
    ns = {"__name__": "js_%s" % tag, "__file__": "src/jax_solver_global.py"}
    exec(compile(SRC.replace(old, new), "js_%s.py" % tag, "exec"), ns)
    return types.SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})

# kill only the VERTICAL advective flux (keep horizontal), and vice versa
JS_NOZADV = variant("nozadv", "Fz_int = Fz_in[..., 1:-1] * jnp.where(",
                    "Fz_int = Fz_in[..., 1:-1] * 0.0 * jnp.where(")
JS_NOHADV = variant("nohadv", "Fx = ux_face * Tx_face * gate_x",
                    "Fx = ux_face * Tx_face * gate_x * 0.0")

_, _, _, p_ref, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)
wet3 = np.asarray(p_ref.wet_mask_z, float)
DZN = np.asarray(p_ref.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = RHO_0 * C_P / 1e21
deep_m = (np.arange(14)[None, None, :] >= 10)
up_m = (np.arange(14)[None, None, :] <= 7)

print("%-16s %10s %10s %10s | %11s %11s" % ("arm", "deep dT", "upper dT",
                                            "tot dOHC", "k13 dT", "k0 dT"))
def arm(tag, over=None, mod=JS, nstep=200, **mk):
    phys = replace(PhysicsConfig(), **{**BASE, **(over or {})})
    step, _, _, p, _ = mod.make_solver_global(
        g, phys, 3600.0, T_atm=jnp.asarray(T_atm_np), lambda_bulk=BULK_LAMBDA_DEFAULT,
        mode_split=True, dt_bt=300.0, return_params=True, **mk)
    s = ST0
    for _ in range(nstep):
        s = step(s)
    jax.block_until_ready(s)
    dT = np.asarray(s.T, float) - T0_np
    vv = lambda m: (dT * vol * m).sum() / (vol * m).sum()
    print("%-16s %+10.5f %+10.5f %+10.3f | %+11.5f %+11.5f"
          % (tag, vv(deep_m), vv(up_m), (dT * vol).sum() * ZJ,
             dT[:, :, 13].mean(), dT[:, :, 0].mean()), flush=True)

arm("baseline")
arm("ALLMIX0", NOMIX)
arm("ALLMIX0+NOZADV", NOMIX, mod=JS_NOZADV)
arm("ALLMIX0+NOHADV", NOMIX, mod=JS_NOHADV)
arm("cap0", {}, polar_cap_rows=0)
arm("ALLMIX0+cap0", NOMIX, polar_cap_rows=0)

print("\n=== advective column heat leak ===")
Fz = np.asarray(JS._vertical_transport_iface(ST0.u, ST0.v, p_ref), float)
T = np.asarray(ST0.T, float)
Ttop = T[:, :, 0]
Fz0 = Fz[:, :, 0]
leak = (Fz0 * Ttop * AREA) * RHO_0 * C_P / 1e21 * 8766.0
print("  Fz[:,:,0] rms                = %.4e m^3/s" % np.sqrt((Fz0 ** 2).mean()))
print("  |Fz0| * area, sum            = %.4e m^3/s" % (np.abs(Fz0) * AREA).sum())
print("  top-face heat leak sum       = %+.3f ZJ/yr" % leak.sum())
print("  top-face heat leak |abs| sum = %+.3f ZJ/yr" % (np.abs(leak)).sum())
print("\ndone")
