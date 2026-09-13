"""Exact per-level tendency budget from _tracer_terms at ckpt_tenyr_ms_gm.

One call, no time stepping: reports dT/dt (K/yr) at each level split into
adv / diff_h / diff_v / conv / gm / redi, volume-meaned over wet cells.
This attributes the bottom-node warming to a specific operator.
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

SPY = 8760.0
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
ST = JS.JaxStateG(u=jnp.asarray(np.asarray(d["u"], np.float64)),
                  v=jnp.asarray(np.asarray(d["v"], np.float64)),
                  T=jnp.asarray(np.asarray(d["T"], np.float64)),
                  S=jnp.asarray(np.asarray(d["S"], np.float64)),
                  eta=jnp.asarray(np.asarray(d["eta"], np.float64)))
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)

_, _, _, p, terms_fn = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)

T3 = np.asarray(terms_fn(ST), float)          # (6, nx, ny, nz) = [adv,diffh,diffv,conv,gm,redi]
wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
NAMES = ["adv", "diff_h", "diff_v", "conv", "gm", "redi"]

print("volume-mean dT/dt per level (K/yr) at ckpt_tenyr_ms_gm")
print("k   depth   " + "".join("%11s" % n for n in NAMES) + "%11s" % "SUM")

depths = [0, -5, -15, -30, -50, -75, -100, -150, -200, -300, -500, -1000, -2000, -4000]
for k in range(14):
    vk = vol[:, :, k]
    den = vk.sum()
    row = [(T3[i][:, :, k] * vk).sum() / den * SPY for i in range(6)]
    print("%-3d %6d " % (k, depths[k]) + "".join("%+11.4f" % x for x in row)
          + "%+11.4f" % sum(row))

print("\nsame, but ONLY over columns whose bottom wet layer is k=13 (open abyss):")
kbot = np.asarray(p.wet_mask_z, float).argmax(axis=-1) * 0 + \
    (14 - 1 - np.asarray(p.wet_mask_z[..., ::-1], float).argmax(axis=-1))
sel = (kbot == 13)
print("  columns:", int(sel.sum()))
for k in (11, 12, 13):
    vk = vol[:, :, k] * sel
    den = vk.sum()
    if den <= 0:
        print("  k%d: no cells" % k); continue
    row = [(T3[i][:, :, k] * vk).sum() / den * SPY for i in range(6)]
    print("  k%-2d " % k + "".join("%+11.4f" % x for x in row) + "%+11.4f" % sum(row))

print("\nsame, but ONLY over columns whose bottom wet layer is < 13 (shelf/sill):")
sel2 = (kbot < 13)
print("  columns:", int(sel2.sum()))
for k in (11, 12, 13):
    vk = vol[:, :, k] * sel2
    den = vk.sum()
    if den <= 0:
        print("  k%d: no cells" % k); continue
    row = [(T3[i][:, :, k] * vk).sum() / den * SPY for i in range(6)]
    print("  k%-2d " % k + "".join("%+11.4f" % x for x in row) + "%+11.4f" % sum(row))
print("\ndone")
