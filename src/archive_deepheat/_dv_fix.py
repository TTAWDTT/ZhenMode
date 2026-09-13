"""Conservative interface-flux vertical diffusion, plus the double-apply fix.

_d2_dz2 (node form) is used for kappa_v in the N-step residual AND the L
half-step. It is NOT column-conservative: applied to the real checkpoint T it
removes -56.89 ZJ/yr from the ocean. kappa_v*dt is tiny vs the raw operator
(1e-5*3600 = 0.036), but the operator's OWN column leak is large.

Build diff_v in interface-flux form (exactly conservative, mirroring
_conv_flux_tendency) and verify:
  1. column sum == 0 for any T and any mask
  2. the L half-step should NOT apply kappa_v at all (the N-step residual
     subtraction in _compute_tracer_residual is designed to cancel the
     kappa_h term, and _linear_half_step re-adds kappa_h; the kappa_v pair is
     NOT set up this way).
"""
import sys
from dataclasses import replace
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from jax_solver_global import (make_solver_global, JaxStateG, _d2_dz2,
                               _fill_ghost_bottom)
from forcing import air_temp_profile, BULK_LAMBDA_DEFAULT

RHO_0, C_P = 1025.0, 3992.0
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
T0 = np.asarray(d["T"], np.float64)
init = np.load("init_fields_g360x120.npz")
T_atm = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
phys = replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
               kappa_gm=1000.0, kappa_redi=1000.0, kappa_v=1e-5,
               kappa_conv=0.05, gm_slope_max=0.005)
step, init_fn, diag, p, terms_fn = make_solver_global(
    g, phys, 3600.0, T_atm=jnp.asarray(T_atm), lambda_bulk=BULK_LAMBDA_DEFAULT,
    mode_split=True, dt_bt=300.0, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJ = RHO_0*C_P/1e21*3600.0*8760.0

def diff_v_flux(tracer, kappa, p):
    """Conservative interface-flux vertical diffusion (zero flux at top/bottom)."""
    tr = _fill_ghost_bottom(tracer, p)
    Fz_i = -kappa * (tr[..., 1:] - tr[..., :-1]) / p.dz_iface
    wet_if = (p.wet_mask_z[..., :-1] > 0.5) & (p.wet_mask_z[..., 1:] > 0.5)
    Fz_i = jnp.where(wet_if, Fz_i, 0.0)
    up = jnp.concatenate([jnp.zeros_like(Fz_i[..., :1]), Fz_i], axis=-1)
    dn = jnp.concatenate([Fz_i, jnp.zeros_like(Fz_i[..., :1])], axis=-1)
    return (up - dn) / p.dz_node * p.wet_mask_z

# ---- test conservation on random + real data ----
key = jax.random.PRNGKey(0)
for tag, arr in [("real ckpt T", T0),
                 ("random", np.asarray(jax.random.normal(key, (360, 120, 14)))),
                 ("quadratic", np.broadcast_to((1e-6*np.array([0,5,15,30,50,75,100,150,200,300,500,1000,2000,4000],float)**2)[None,None,:], (360,120,14)).copy())]:
    a = jnp.asarray(arr)
    nf = np.asarray(_d2_dz2(a, p))
    fl = np.asarray(diff_v_flux(a, 1.0, p))
    print("%-12s  node-form column sum %+12.5e   flux-form %+12.5e"
          % (tag, (nf*vol).sum(), (fl*vol).sum()))

print("\nnode-form kappa_v contribution : %+.4f ZJ/yr" % (1e-5*(np.asarray(_d2_dz2(jnp.asarray(T0), p))*vol).sum()*ZJ))
print("flux-form kappa_v contribution : %+.4f ZJ/yr" % (1e-5*(np.asarray(diff_v_flux(jnp.asarray(T0), 1e-5, p))*vol).sum()*ZJ))

# per-level comparison
print("\nper-level (ZJ/yr), kappa_v=1e-5:")
nf = 1e-5*np.asarray(_d2_dz2(jnp.asarray(T0), p))
fl = 1e-5*np.asarray(diff_v_flux(jnp.asarray(T0), 1e-5, p))
print("   k    zc      node-form    flux-form")
Z = np.array([0,5,15,30,50,75,100,150,200,300,500,1000,2000,4000], float)
for k in range(14):
    print("  %2d %6.0f  %+11.4f  %+11.4f"
          % (k, -Z[k], (nf[:, :, k]*vol[:, :, k]).sum()*ZJ, (fl[:, :, k]*vol[:, :, k]).sum()*ZJ))
print("  TOTAL      %+11.4f  %+11.4f" % ((nf*vol).sum()*ZJ, (fl*vol).sum()*ZJ))
