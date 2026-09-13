"""Time local CPU stepping + confirm the kappa_v double-application."""
import sys, time
from dataclasses import replace
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from jax_solver_global import make_solver_global, JaxStateG, _d2_dz2
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT

g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)
init = np.load("init_fields_g360x120.npz")
T_atm = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
step, init_fn, diag, p, terms_fn = make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)
st = JaxStateG(u=jnp.asarray(np.asarray(d["u"], np.float64)),
               v=jnp.asarray(np.asarray(d["v"], np.float64)),
               T=jnp.asarray(np.asarray(d["T"], np.float64)),
               S=jnp.asarray(np.asarray(d["S"], np.float64)),
               eta=jnp.asarray(np.asarray(d["eta"], np.float64)))
t0 = time.time(); s = jax.block_until_ready(step(st)); print("first (compile) %.1f s" % (time.time()-t0))
t0 = time.time()
for _ in range(20):
    s = step(s)
s.T.block_until_ready()
dt = (time.time()-t0)/20
print("steady %.3f s/step  -> %.1f steps/s ; 1 model-yr (8760 steps) = %.1f min"
      % (dt, 1/dt, 8760*dt/60))

# --- confirm double application of kappa_v ---
st0 = st
t0 = time.time(); a = np.asarray(step(st0).T); t1 = time.time()
phys2 = replace(phys, kappa_v=0.0)
step2, _, _, p2, _ = make_solver_global(g, phys2, 3600.0, T_atm=jnp.asarray(T_atm),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)
b = np.asarray(step2(st0).T)
wet3 = np.asarray(p.wet_mask_z, float); DZN = np.asarray(p.dz_node).ravel()
AREA = np.asarray(g.dx_2d, float)*float(g.dy)
vol = wet3*AREA[:, :, None]*DZN[None, None, :]
ZJ = 1025.0*3992.0/1e21*8760.0
print("\ndiff_v contribution (direct differencing)  = %+.4f ZJ/yr"
      % (((a-b)*vol).sum()*ZJ))
lap1 = np.asarray(_d2_dz2(jnp.asarray(np.asarray(d["T"], np.float64)), p))
print("kappa_v*_d2_dz2 ONE application           = %+.4f ZJ/yr"
      % (1e-5*(lap1*vol).sum()*ZJ))
print("ratio (should be 2.0 if double-applied)   = %.3f"
      % ((((a-b)*vol).sum()*ZJ) / (1e-5*(lap1*vol).sum()*ZJ)))
