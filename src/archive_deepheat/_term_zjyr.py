"""Direct per-term attribution of the deep-heat budget at ckpt_tenyr_ms_gm.

dOHC(baseline, 1 step) - dOHC(term off, 1 step) = that term's net contribution
to the WHOLE-COLUMN heat budget, in ZJ/yr. One step is enough for a tendency.

Arms:
  baseline        production config, real u,v from the ten-yr checkpoint
  conv=0          kill convective adjustment (kappa_conv 0.05 -> 0)
  conv=0.005      10x weaker convective adjustment
  gm=redi=0       kill isopycnal skew flux
  kappa_v=0       kill explicit vertical diffusion
  topzero         Fz_top = 0 (exact top-face advective conservation)
  advoff          kill the vertical advection flux Fz_int entirely
  bulkoff         T_atm := current SST (zero bulk heat flux)
"""
import sys, types, time
from dataclasses import replace
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT
import jax_solver_global as JS

RHO_0, C_P, SPY = 1025.0, 3992.0, 8760.0
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

# ---- patched module variants -------------------------------------------
SRC = open("src/jax_solver_global.py", encoding="utf-8").read()

def variant(tag, old, new):
    assert SRC.count(old) == 1, "%s: %d matches" % (tag, SRC.count(old))
    ns = {"__name__": "js_%s" % tag, "__file__": "src/jax_solver_global.py"}
    exec(compile(SRC.replace(old, new), "js_%s.py" % tag, "exec"), ns)
    return types.SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})

JS_TOPOZERO = variant("topzero", "Fz_top = Fz_in[:, :, :1] * T[..., :1]",
                      "Fz_top = Fz_in[:, :, :1] * T[..., :1] * 0.0")
# kill the vertical advective flux: Fz_int -> 0 (horizontal advection stays)
JS_ADVOFF = variant("advoff",
                    "Fz_int = Fz_in[..., 1:-1] * jnp.where(",
                    "Fz_int = Fz_in[..., 1:-1] * 0.0 * jnp.where(")
print("patched modules built OK", flush=True)

_, _, _, p_ref, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)
wet3 = np.asarray(p_ref.wet_mask_z, float)
DZN = np.asarray(p_ref.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
vsum = vol.sum()
# deep = k >= 10 (below 500 m); upper = k <= 7 (above 150 m)
deep_m = (np.arange(14)[None, None, :] >= 10)
up_m = (np.arange(14)[None, None, :] <= 7)
vol_d = (vol * deep_m).sum()
vol_u = (vol * up_m).sum()

T0 = np.asarray(ST.T, float)

def measure(tag, over=None, mod=JS, Tatm=None, nstep=1):
    phys = replace(PhysicsConfig(), **{**BASE, **(over or {})})
    ta = jnp.asarray(T_atm_np if Tatm is None else Tatm)
    step, _, _, p, _ = mod.make_solver_global(
        g, phys, 3600.0, T_atm=ta, lambda_bulk=BULK_LAMBDA_DEFAULT,
        mode_split=True, dt_bt=300.0, return_params=True)
    s = ST
    for _ in range(nstep):
        s = step(s)
    jax.block_until_ready(s)
    dT = np.asarray(s.T, float) - T0
    tot = (dT * vol).sum() * RHO_0 * C_P / 1e21 * SPY
    ddeep = (dT * vol * deep_m).sum() * RHO_0 * C_P / 1e21 * SPY
    dup = (dT * vol * up_m).sum() * RHO_0 * C_P / 1e21 * SPY
    return tot, ddeep, dup

t0 = time.time()
bt, bd, bu = measure("baseline")
print("baseline            total %+9.2f  deep %+9.2f  upper %+9.2f ZJ/yr  (%.0f s)"
      % (bt, bd, bu, time.time() - t0), flush=True)

ARMS = [
    ("conv=0",      dict(kappa_conv=0.0), JS, None),
    ("conv=0.005",  dict(kappa_conv=0.005), JS, None),
    ("gm=redi=0",   dict(kappa_gm=0.0, kappa_redi=0.0), JS, None),
    ("kappa_v=0",   dict(kappa_v=0.0), JS, None),
    ("topzero",     {}, JS_TOPOZERO, None),
    ("advoff",      {}, JS_ADVOFF, None),
]
print("\n%-14s %10s %10s %10s | %10s %10s" % ("arm", "total", "deep", "upper",
                                             "d_total", "d_deep"), flush=True)
for tag, over, mod, ta in ARMS:
    t, dd, u = measure(tag, over, mod=mod, Tatm=ta)
    print("%-14s %+10.2f %+10.2f %+10.2f | %+10.2f %+10.2f"
          % (tag, t, dd, u, t - bt, dd - bd), flush=True)

# bulk: T_atm := current SST -> zero heat flux
t, dd, u = measure("bulkoff", Tatm=np.asarray(ST.T, float)[:, :, 0])
print("%-14s %+10.2f %+10.2f %+10.2f | %+10.2f %+10.2f"
      % ("bulkoff", t, dd, u, t - bt, dd - bd), flush=True)
print("\ndone", flush=True)
