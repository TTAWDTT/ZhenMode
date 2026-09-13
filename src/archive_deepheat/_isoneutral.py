"""Isoneutrality test of the skewed GM/Redi operator.

The skew operator is LINEAR in the tracer for fixed slopes.  A correct
isoneutral discretization therefore annihilates density: applying it to
rho' = -alpha*(T-Tref) + beta*(S-Sref) must give ~0.  Any residual is
spurious DIAPYCNAL mixing - the quantity that actually erodes deep
stratification and warms the abyss.

Tests at three states: the ten-year checkpoint and the WOA initial state.
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
init = np.load("init_fields_g360x120.npz")
T_atm_np = air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0])
BASE = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05, gm_slope_max=0.005)
_, _, _, p, _ = JS.make_solver_global(
    g, replace(PhysicsConfig(), **BASE), 3600.0, T_atm=jnp.asarray(T_atm_np),
    lambda_bulk=BULK_LAMBDA_DEFAULT, mode_split=True, dt_bt=300.0, return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
ZJYR = 1025.0 * 3992.0 / 1e21 * 3.1536e7
ALPHA, BETA, TREF, SREF = 2.0e-4, 7.6e-4, 15.0, 35.0

def col(f):
    f = np.asarray(f)
    return np.array([(f[:, :, k] * vol[:, :, k]).sum() for k in range(14)])

STATES = {}
d10 = np.load("results/ckpt_tenyr_ms_gm.npz")
STATES["tenyr-ckpt"] = (np.asarray(d10["T"], float), np.asarray(d10["S"], float))
STATES["WOA init"] = (init["T_init"].astype(float), init["S_init"].astype(float))

for tag, (T0, S0) in STATES.items():
    st = JS.JaxStateG(u=jnp.asarray(d10["u"], float), v=jnp.asarray(d10["v"], float),
                      T=jnp.asarray(T0), S=jnp.asarray(S0),
                      eta=jnp.asarray(d10["eta"], float))
    Sx, Sy = JS._isopycnal_slope(st, p)
    rho = JS._density_anomaly(st.T, st.S, p) * p.wet_mask_z
    kiso = 2000.0
    # 1. operator applied to rho itself -> must be ~0
    t_rho = JS._redi_skew_flux_tendency(rho, Sx, Sy, p, kappa=kiso)
    # 2. operator on T and S, combined into the density tendency
    t_T = JS._redi_skew_flux_tendency(st.T, Sx, Sy, p, kappa=kiso)
    t_S = JS._redi_skew_flux_tendency(st.S, Sx, Sy, p, kappa=kiso)
    t_rho_via_TS = -ALPHA * t_T + BETA * t_S
    tn, tv = col(t_rho), col(t_rho_via_TS)
    m = wet3 > 0.5
    print("=" * 100)
    print(f"STATE: {tag}")
    print("=" * 100)
    print("  operator(rho)      max|.| %.3e  rms %.3e   (must be ~0 if isoneutral)"
          % (np.abs(np.asarray(t_rho))[m].max(), np.asarray(t_rho)[m].std()))
    print("  operator(rho)!=0 cells: %d / %d  (%.2f%%)"
          % (int((np.abs(np.asarray(t_rho)) > 1e-12)[m].sum()), int(m.sum()),
             100.0 * (np.abs(np.asarray(t_rho)) > 1e-12)[m].mean()))
    print("\n  k   z_top     col op(rho) [kg/m3 per s]    col op(rho via T,S)")
    ZTOP = [0, 5, 15, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 4000]
    for k in range(14):
        print("  %2d  %-7.0f   %+.4e                  %+.4e" % (k, ZTOP[k], tn[k], tv[k]))
    print("  SUM            %+.4e                  %+.4e" % (tn.sum(), tv.sum()))
    # 3. the implied diapycnal buoyancy flux, as an equivalent K_dia
    print("\n  equivalent diapycnal diffusivity from op(rho_via_T,S):")
    dCdz = np.asarray(JS._d_dz(JS._fill_ghost_bottom(st.T, p), p))
    # K_dia = w_eff * dz  where w_eff from rho tendency... report K = |t_rho|/|drho/dz| * dz^2
    drho = np.asarray(JS._d_dz(JS._fill_ghost_bottom(rho, p), p))
    wet_i = wet3[:, :, :-1] * wet3[:, :, 1:] > 0.5
    dz_i = np.asarray(p.dz_iface).ravel()
    for k in range(13):
        mm = wet_i[:, :, k]
        if mm.sum() == 0:
            continue
        Kh = -np.asarray(t_rho_via_TS)[:, :, k][mm] / drho[:, :, k][mm] * dz_i[k] ** 2
        print("    k=%2d  mean K_dia_spurious = %+.3e m2/s   (kappa_v=1e-5)"
              % (k, float(np.median(Kh))))
