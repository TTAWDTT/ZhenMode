"""Locate the exact source of the _d2_dz2 column leak.

Per-column sums are ~-0.135 K/step*... uniformly negative -> check the bottom
boundary stencil and the dz_node/dz_bnd_bot definition.
"""
import sys
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
T0 = jnp.asarray(np.asarray(d["T"], np.float64)); S0 = jnp.asarray(np.asarray(d["S"], np.float64))
u0 = jnp.asarray(np.asarray(d["u"], np.float64)); v0 = jnp.asarray(np.asarray(d["v"], np.float64))
e0 = jnp.asarray(np.asarray(d["eta"], np.float64))
init = np.load("init_fields_g360x120.npz")
T_atm = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)
step, init_fn, diag, p, terms_fn = make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)
st = JaxStateG(u=u0, v=v0, T=T0, S=S0, eta=e0)

print("dz_node       ", np.round(np.asarray(p.dz_node).ravel(), 4))
print("dz_iface      ", np.round(np.asarray(p.dz_iface), 4) if hasattr(p, 'dz_iface') else "n/a")
print("dz_denom_int  ", np.round(np.asarray(p.dz_denom_interior), 4))
print("dz_bnd_top    ", float(p.dz_bnd_top), " dz_bnd_bot", float(p.dz_bnd_bot))
print("d2z_h0_top    ", float(p.d2z_h0_top), " d2z_h0_bot", float(p.d2z_h0_bot))
try:
    print("d2z_hm        ", np.round(np.asarray(p.d2z_hm), 4))
    print("d2z_hp        ", np.round(np.asarray(p.d2z_hp), 4))
    print("d2z_denom     ", np.round(np.asarray(p.d2z_denom), 4))
except Exception as e:
    print("no d2z arrays", e)

wet = np.asarray(p.wet_mask_z)
# take one wet column with a deep bottom and inspect
kk = int(np.argmax((wet > 0.5).sum(axis=-1).ravel()))
ii, jj = np.unravel_index(kk, (360, 120))
kb = int(np.where(wet[ii, jj] > 0.5)[0].max())
print("\ncolumn (%d,%d) bottom wet k=%d" % (ii, jj, kb))
print("  T column:", np.round(np.asarray(st.T)[ii, jj], 3))
print("  wet     :", wet[ii, jj].astype(int))
lap = np.asarray(_d2_dz2(st.T, p))
print("  d2T     :", np.round(lap[ii, jj], 5))

DZN = np.asarray(p.dz_node).ravel()
print("\n  dz-weighted column sum = %.5e" % (lap[ii, jj] * DZN).sum())
print("  contributions:", np.round(lap[ii, jj] * DZN, 5))

# check what the correct interface-flux form would give
print("\n-- compare with interface-flux (conservative) form --")
Tm = np.asarray(st.T)[ii, jj, :-1]; Tp = np.asarray(st.T)[ii, jj, 1:]
dzif = np.asarray(p.dz_iface).ravel() if hasattr(p, 'dz_iface') else None
if dzif is not None:
    print("  dz_iface len", len(dzif))
    dTdz_i = (Tp - Tm) / dzif
    print("  interface gradients:", np.round(dTdz_i, 6))
iface_wet = (wet[ii, jj, :-1] > 0.5) & (wet[ii, jj, 1:] > 0.5)
print("  iface wet:", iface_wet.astype(int))
print("  note: k=%d..13 are ghost below the bottom wet k=%d" % (kb + 1, kb))
