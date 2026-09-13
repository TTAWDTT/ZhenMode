"""HEAT CONSERVATION TEST — the decisive check.

With the surface bulk flux zeroed (T_atm := current SST) and Q_heat = 0, every
remaining tracer operator (advection, horizontal/vertical diffusion, convective
adjustment, GM/Redi skew flux, biharmonic) redistributes heat WITHIN the ocean.
The total column heat integral must therefore be constant.

Any drift is a conservation defect. Measure it in ZJ/yr, per level, and against
the observed deep-warming signal (~+216 ZJ/yr into k>=10 at the ten-yr ckpt).

Arms:
  conservative   bulk flux zeroed (T_atm = SST), all closures ON
  + diag         same, but report per-level dOHC per step
"""
import sys
from dataclasses import replace
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
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

DT = 3600.0
_, _, _, p_ref, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), DT, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)
wet3 = np.asarray(p_ref.wet_mask_z, float)
DZN = np.asarray(p_ref.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = RHO_0 * C_P / 1e21
SPY_S = 1.0 / (DT * 365.25 * 24)          # per-step -> per-year factor


def run(tag, over=None, Tatm=None, nstep=200, mod=JS):
    phys = replace(PhysicsConfig(), **{**BASE, **(over or {})})
    ta = jnp.asarray(T_atm_np if Tatm is None else Tatm)
    step, _, _, p, _ = mod.make_solver_global(
        g, phys, DT, T_atm=ta, lambda_bulk=BULK_LAMBDA_DEFAULT,
        mode_split=True, dt_bt=300.0, return_params=True)
    s = ST0
    h0 = float((np.asarray(s.T, float) * vol).sum()) * ZJ
    traj = []
    for n in range(nstep):
        s = step(s)
        if (n + 1) % 50 == 0:
            jax.block_until_ready(s)
            h = float((np.asarray(s.T, float) * vol).sum()) * ZJ
            traj.append((n + 1, h))
    jax.block_until_ready(s)
    Tend = np.asarray(s.T, float)
    h1 = float((Tend * vol).sum()) * ZJ
    net = (h1 - h0) / nstep * SPY_S
    print("%-16s H0 %+11.2f  H1 %+11.2f  net %+9.2f ZJ/yr" % (tag, h0, h1, net))
    for n, h in traj:
        print("      step %4d  H %+11.2f  dH %+8.3f ZJ  (%.2f ZJ/yr)"
              % (n, h, h - h0, (h - h0) / n * SPY_S))
    return Tend


print("=== CONSERVATION TEST: surface heat flux zeroed ===")
print("if the model is conservative, net must be ~0\n")
Tatm_zero = T0_np[:, :, 0]                       # T_atm := SST  => bulk flux = 0
Tend = run("bulkoff 200st", Tatm=Tatm_zero, nstep=200)

print("\nper-level heat change (ZJ) over 200 steps, bulk flux off:")
dT = Tend - T0_np
for k in range(14):
    dk = (dT[:, :, k] * vol[:, :, k]).sum() * ZJ
    print("  k%-2d  %+9.4f ZJ   (%+8.2f ZJ/yr)" % (k, dk, dk / 200 * SPY_S))
tot = (dT * vol).sum() * ZJ
print("  TOTAL %+9.4f ZJ   (%+8.2f ZJ/yr)" % (tot, tot / 200 * SPY_S))

print("\n=== control: baseline (bulk ON) 200 steps ===")
run("baseline 200st", nstep=200)
print("\ndone")
