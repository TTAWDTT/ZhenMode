"""Stage decomposition of the per-level T change, production physics.

_decomp3 showed the six terms_fn terms sum to +7.08 ZJ/yr while the ACTUAL
step is +165.22 ZJ/yr -- the dipole is in MISSING.  Split _step_impl into its
stages and report the per-level dT/yr of each, so the dipole is attributed to
a stage rather than to a bag of left-over terms.

Stages: L1 (linear half-step #1), N (explicit RK2), L2, barotropic subcycle
projection, polar cap, mask/hold.
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "src")
from dataclasses import replace
from grid import make_global_grid, GlobalGridConfig
from config import DEFAULT_CONFIG, PhysicsConfig
from forcing import air_temp_profile
import jax_solver_global as JS

RHO_0, C_P = 1025.0, 3992.0
DT = 3600.0
g = make_global_grid(GlobalGridConfig(lat_max=60.0, ny=120),
                     DEFAULT_CONFIG.bathymetry_file, smooth_passes=30, min_depth=100.0)
d = np.load("results/ckpt_tenyr_ms_gm.npz")
u0 = np.asarray(d["u"], np.float64); v0 = np.asarray(d["v"], np.float64)
T0 = np.asarray(d["T"], np.float64); S0 = np.asarray(d["S"], np.float64)
e0 = np.asarray(d["eta"], np.float64)
init = np.load("init_fields_g360x120.npz")
T_atm_np = np.asarray(air_temp_profile(g, init["T_init"].astype(np.float64)[:, :, 0]), float)

PROD = dict(nu_h=5e6, nu_bi=2e14, kappa_bi=2e14, kappa_gm=1000.0,
            kappa_redi=1000.0, kappa_v=1e-5, kappa_conv=0.05,
            gm_slope_max=0.005)
phys = replace(PhysicsConfig(), **PROD)
_, _, _, p, _ = JS.make_solver_global(
    g, phys, DT, T_atm=jnp.asarray(T_atm_np), lambda_bulk=40.0,
    mode_split=True, dt_bt=300.0, polar_cap_rows=2, polar_cap_taper=3,
    T_init=jnp.asarray(init["T_init"], np.float64),
    S_init=jnp.asarray(init["S_init"], np.float64),
    return_params=True)

wet3 = np.asarray(p.wet_mask_z, float)
DZN = np.asarray(p.dz_node).ravel()
AREA = np.asarray(g.dx_2d, float) * float(g.dy)
vol = wet3 * AREA[:, :, None] * DZN[None, None, :]
YRS = DT / 3.1536e7
RHO_CP = RHO_0 * C_P / 1e21


def per_level(dTstep):
    a = np.asarray(dTstep, float)
    return [float((a[:, :, k] * vol[:, :, k]).sum() / vol[:, :, k].sum()) / YRS
            if vol[:, :, k].sum() > 0 else 0.0 for k in range(14)]


def tot(dTstep):
    return float((np.asarray(dTstep, float) * vol).sum() * RHO_CP / YRS)


s = JS.JaxStateG(u=jnp.asarray(u0), v=jnp.asarray(v0), T=jnp.asarray(T0),
                 S=jnp.asarray(S0), eta=jnp.asarray(e0))
T_start = np.asarray(s.T, float).copy()

stages = []
_prev = [T_start.copy()]


def snap(name, T_now):
    """Record this stage's own increment (T_now - T_prev), not cumulative."""
    a = np.asarray(T_now, float).copy()
    stages.append((name, a - _prev[0]))
    _prev[0] = a


land_u, land_v = s.u, s.v
land_T, land_S = s.T, s.S
dt_half = p.dt / 2.0

s1 = JS._linear_half_step(s, p, dt_half)
snap("L half-step #1", s1.T)
s2 = JS._explicit_full_step(s1, p, p.dt)
snap("N step (RK2)", s2.T)

# barotropic subcycle (does not touch T, but replicate for completeness)
state = s2
F_rho_x, F_rho_y = JS._compute_bt_rho_pgf(state, p)
ubt0, vbt0 = JS._barotropic_velocity(state.u, state.v, p)
ubt, vbt = ubt0, vbt0
eta = state.eta
for _ in range(int(p.n_subcyc)):
    eta, ubt, vbt = JS._free_surface_step_fd(eta, ubt, vbt, p, F_rho_x, F_rho_y,
                                             dt_half=p.dt_bt)
state = JS.JaxStateG(state.u + (ubt - ubt0)[:, :, None],
                     state.v + (vbt - vbt0)[:, :, None],
                     state.T, state.S, eta)
snap("bt subcycle (proj)", state.T)

s4 = JS._linear_half_step(state, p, dt_half)
snap("L half-step #2", s4.T)

T_pc = JS._polar_cap_3d(s4.T, p)
snap("polar cap", T_pc)

wmask = p.wet_mask_z
T_fin = T_pc * wmask + land_T * (1.0 - wmask)
snap("mask/hold", T_fin)

print("=" * 100)
print("Stage decomposition, per-level dT/yr, production physics (ONE step)")
print("=" * 100)
print("  %-20s" % "stage" + "".join("%9s" % ("k%d" % k) for k in range(14)) + "%11s" % "TOT ZJ/yr")
carry = np.zeros_like(T_start)
for name, dT in stages:
    carry = carry + dT
    print("  %-20s" % name + "".join("%+9.4f" % x for x in per_level(dT))
          + "%+11.2f" % tot(dT))
print()
print("  %-20s" % "CUMULATIVE" + "".join("%+9.4f" % x for x in per_level(carry))
      + "%+11.2f" % tot(carry))
T_ref_fin = np.asarray(T_fin, float)
print("  %-20s" % "ACTUAL full step"
      + "".join("%+9.4f" % x for x in per_level(T_ref_fin - T_start))
      + "%+11.2f" % tot(T_ref_fin - T_start))
