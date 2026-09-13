"""Final candidates: is the deep warming driven by the wind forcing?

The ablation showed k13 dT is invariant to every MIXING closure. Remaining
time-dependent, multi-step, non-switchable inputs: the wind stress (seasonal,
wind_jit) and the bulk-flux strength.

Arms (200 steps from ckpt_tenyr_ms_gm):
  baseline      production
  wind_off      build with forcing tau=0 (no wind at all)
  bulk20        halve the bulk coupling (lambda 40 -> 20 W/m^2/K)
  bulk80        double it (40 -> 80)
  bulk0         zero it entirely (T_atm := SST)
Reports per-level dT, per-level dOHC contribution, and the k13/k0 signature.

Also reports the momentum/Coriolis consistency: whether the diagnosed vertical
velocity w is consistent with the horizontal divergence it is built from.
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

_, _, _, p_ref, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)
wet3 = np.asarray(p_ref.wet_mask_z, float)
DZN = np.asarray(p_ref.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = RHO_0 * C_P / 1e21
deep_m = (np.arange(14)[None, None, :] >= 10)
up_m = (np.arange(14)[None, None, :] <= 7)

print("%-12s %10s %10s %10s | %11s %11s" % ("arm", "deep dT", "upper dT",
                                            "tot dOHC", "k13 dT", "k0 dT"))
def arm(tag, over=None, lam=BULK_LAMBDA_DEFAULT, Tatm=None, nstep=200):
    phys = replace(PhysicsConfig(), **{**BASE, **(over or {})})
    ta = jnp.asarray(T_atm_np if Tatm is None else Tatm)
    step, _, _, p, _ = JS.make_solver_global(
        g, phys, 3600.0, T_atm=ta, lambda_bulk=lam,
        mode_split=True, dt_bt=300.0, return_params=True)
    s = ST0
    for _ in range(nstep):
        s = step(s)
    jax.block_until_ready(s)
    dT = np.asarray(s.T, float) - T0_np
    vv = lambda m: (dT * vol * m).sum() / (vol * m).sum()
    print("%-12s %+10.5f %+10.5f %+10.3f | %+11.5f %+11.5f"
          % (tag, vv(deep_m), vv(up_m), (dT * vol).sum() * ZJ,
             dT[:, :, 13].mean(), dT[:, :, 0].mean()), flush=True)

arm("baseline")
arm("bulk20", lam=20.0)
arm("bulk80", lam=80.0)
arm("bulk0", Tatm=T0_np[:, :, 0])

print("\n=== vertical velocity / divergence consistency ===")
Fz = JS._vertical_transport_iface(ST0.u, ST0.v, p_ref)
divh = JS._divergence_h(ST0.u, ST0.v, p_ref)
w = np.asarray(Fz, float) / (AREA[:, :, None] / np.asarray(p_ref.dx_2d, float)[:, :, None] /
                             np.asarray(p_ref.dy, float))
print("  Fz (volume transport) rms = %.4e m^3/s" % np.sqrt((np.asarray(Fz, float) ** 2).mean()))
print("  Fz[:, :, 0] (rigid-lid leak) rms = %.4e" % np.sqrt((np.asarray(Fz, float)[:, :, 0] ** 2).mean()))
print("  div_h rms over wet cells = %.4e 1/s" % np.sqrt((np.asarray(divh, float) ** 2).mean()))
# column-wise: sum_k div_h*dz should equal Fz[0]
A2 = np.asarray(p_ref.wet_mask, float)
DZN = np.asarray(p_ref.dz_node).ravel()
colsum = (np.asarray(divh, float) * DZN[None, None, :]).sum(axis=-1)
print("  |sum_k div_h*dz| rms = %.4e  (should equal |Fz0|/cell-area scale)" % np.sqrt((colsum ** 2).mean()))
print("\ndone")
