"""Term-by-term T budget at the 10-yr checkpoint, with REAL u,v.

Uses the live solver's own terms_fn (adv, diff_h, diff_v, conv, gm, redi) and
the vertical-diffusion operator, evaluated at results/ckpt_tenyr_ms_gm.npz.
Reports per-level and column-integrated OHC tendency in ZJ/yr for each term.
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")

sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from jax_solver_global import make_solver_global, JaxStateG
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT

RHO_0, C_P = 1025.0, 3992.0
SEC_YR = 3.15576e7
SPY = 8760.0                      # steps per year at dt=3600

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)

d = np.load("results/ckpt_tenyr_ms_gm.npz")
print("ckpt keys", list(d.keys()))
print("cur_step", d["cur_step"])
u0 = jnp.asarray(np.asarray(d["u"], np.float64))
v0 = jnp.asarray(np.asarray(d["v"], np.float64))
T0 = jnp.asarray(np.asarray(d["T"], np.float64))
S0 = jnp.asarray(np.asarray(d["S"], np.float64))
e0 = jnp.asarray(np.asarray(d["eta"], np.float64))
print("u rms %.4e  max %.4e | v rms %.4e max %.4e"
      % (float(jnp.sqrt((u0**2).mean())), float(jnp.abs(u0).max()),
         float(jnp.sqrt((v0**2).mean())), float(jnp.abs(v0).max())))

init = np.load("init_fields_g360x120.npz")
T_atm = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])

phys = replace = None
from dataclasses import replace
phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)

step, init_fn, diag, p, terms_fn = make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)

st = JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)

print("\ncomputing terms_fn ...")
tn = np.asarray(terms_fn(st))            # (6, nx, ny, nz), degC/s
NAMES = ["adv_T", "diff_h_T", "diff_v_T", "conv_T", "gm_T", "redi_T"]
print("terms shape", tn.shape)
print("wet_mask_z shape", np.asarray(p.wet_mask_z).shape)

DZN = np.asarray(p.dz_node).ravel()
print("dz_node", DZN, "sum", DZN.sum())
wet3 = np.asarray(p.wet_mask_z, float)
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
V = vol.sum()
print("ocean vol %.4e m3   wet3 cells %d" % (V, int((wet3 > 0.5).sum())))

W2ZJ = RHO_0 * C_P / 1e21 * SEC_YR * SPY      # degC per step -> ZJ/yr (per cell volume applied next)

print("\nper-level OHC tendency (ZJ/yr), positive = warming:")
print(" k  zc(m)   " + "".join("%11s" % n for n in NAMES) + "      TOTAL")
prof = np.zeros((6, 14))
for k in range(14):
    zc = -np.array([0, 5, 15, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 4000], float)[k]
    row = []
    for t in range(6):
        v = (tn[t][:, :, k] * vol[:, :, k]).sum() * RHO_0 * C_P / 1e21 * SEC_YR * SPY
        prof[t, k] = v
        row.append(v)
    print(" %2d %7.0f  " % (k, zc) + "".join("%+11.3f" % r for r in row) + "  %+11.3f" % sum(row))

print("\ncolumn totals (ZJ/yr):")
for t, n in enumerate(NAMES):
    print("  %-10s %+12.4f" % (n, prof[t].sum()))
print("  %-10s %+12.4f" % ("SUM", prof.sum()))
