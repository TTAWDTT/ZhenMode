"""Resolve the conv contradiction: measure the conv heat rate AS APPLIED by the
actual compiled solver step, on the ACTUAL run state (L-step-modified), vs the
offline reconstruction from snapshots.

Run: python _conv_resolve.py
"""
import numpy as np
import dataclasses
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from grid import make_global_grid
from config import GlobalGridConfig, PhysicsConfig
import jax_solver_global as jsg

gcfg = dataclasses.replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, r'C:\Users\zhen.luo\Desktop\ETOPO_2022_v1_r3600x1800_surface.nc',
                        smooth_passes=30, min_depth=100.0)

d = np.load('../results/global_gm90d_bi2e14_lam120_dt120.npz')
T_init = d['T_init']; S_init = d['S_init']

# Solver params must match the winning 60d/90d config exactly.
physics = dataclasses.replace(PhysicsConfig(), nu_h=5e6, nu_bi=2e14, kappa_bi=2e14,
                              kappa_gm=1000.0, gm_slope_max=0.01)
_, init_state, _, params = jsg.make_solver_global(
    grid, physics, 120.0,
    forcing=(jnp.zeros((360, 120)), jnp.zeros((360, 120)), jnp.zeros((360, 120))),
    eos_type='linear', T_atm=None, lambda_bulk=0.0,
    sponge_days=3.0, sponge_cells=8,
    T_init=jnp.array(T_init), S_init=jnp.array(S_init),
    polar_cap_rows=2, polar_cap_taper=3, return_params=True)

# ── Load d80 snapshot state (the 3-field diag format: T,u,v) ──
sn = np.load('results/global_diag90_blob_3d/snap_00032.npy')
T3, u3, v3 = sn[0], sn[1], sn[2]
S_run = S_init * params.wet_mask_z + (1.0 - params.wet_mask_z) * physics.S_ref
S_run = np.array(S_run)
st = jsg.JaxStateG(jnp.array(u3), jnp.array(v3), jnp.array(T3), jnp.array(S_run),
                   jnp.zeros((360, 120)))

# ── 1. Conv tendency AS the solver computes it (its own state reconstruction) ──
# Mirror _compute_tracer_tendency's conv block exactly, but WITHOUT the final
# wet_mask_z multiplication so we can see the RAW term.
rho_prime = jsg._density_anomaly(st.T, st.S, params) * params.wet_mask_z
wet_iface = (params.wet_mask_z[..., :-1] > 0.5) & (params.wet_mask_l0[...] if False else (params.wet_mask_z[..., 1:] > 0.5))
unstable = (rho_prime[..., :-1] > rho_prime[..., 1:]) & wet_iface
conv_mask_3d = jnp.any(unstable, axis=-1, keepdims=True)
d2T = jsg._d2_dz2(st.T, params)
conv_T_raw = params.kappa_conv * conv_mask_3d * d2T * params.wet_mask_z

# ── 2. The N-step residual actually applied: dT/dt_applied = (residual) = full tendency - kappa_h*lap ──
dT_full, dS_full = jsg._compute_tracer_tendency(st, params)
dT_res, _ = jsg._compute_tracer_residual(st, params)

# ── 3. Volume-weighted global integrals (W) → per-area W/m2 using ocean area ──
area = np.array(grid.area) if hasattr(grid, 'area') else None
if area is None:
    R = 6.371e6
    dlon = np.radians(1.0)
    dlat = np.radians(1.0)
    lon = np.array(grid.lon); lat = np.array(grid.lat)
    area = (R**2) * dlon * np.cos(np.radians(lat))[None, :] * dlat * np.ones((grid.nx, grid.ny))
wlev = np.array(grid.dz_ctrl_vol) if hasattr(grid, 'dz_ctrl_vol') else None
if wlev is None:
    z = np.array(grid.z); h = np.abs(np.diff(z))
    wlev = np.empty(14); wlev[0] = 0.5*h[0]; wlev[-1] = 0.5*h[-1]
    wlev[1:-1] = 0.5*(h[:-1] + h[1:])
wet3 = np.array(params.wet_mask_z)
V = area[:, :, None] * wlev[None, None, :] * wet3   # control volumes m3
Vtot = V.sum()
Ao = area[wet3[:, :, 0] > 0.5].sum()

def _g(x):
    return float(np.sum(np.array(x) * V)) / Vtot    # volume-mean

def _gW(x):
    # W/m2 over ocean area: W = rho0*cp*sum(conv_T*V); /Ao
    return float(RHO0C * np.sum(np.array(x) * V)) / Ao

RHO0C = 1027.0 * 3900.0
print('Vtot = %.4e m3   Ao = %.4e m2' % (Vtot, Ao))
print('conv_T (solver state):  %+.1f W/m2   (volume-mean K/s: %+.3e)' % (
    _gW(conv_T_raw), _g(conv_T_raw)))
print('dT_full (full tendency): %+.1f W/m2' % _gW(dT_full))
print('dT_res  (N-step applied): %+.1f W/m2' % _gW(dT_res))
# conv contribution INSIDE the residual = conv_T (residual subtracts kappa_h*lap only)
# so the N-step applies conv fully once per full dt (Strang: N(dt) applied once).
# L-step applies kappa_h*lap twice at dt/2; N applies conv once at dt.
# time-mean conv heating = conv W/m2  (1x per step, dt=120s)
print()
print('snap-state unstable ifaces: %d  conv cells: %d' % (
    int(np.array(unstable).sum()), int(np.array(conv_mask_3d).sum())))

# ── 4. Repeat at the NEXT snapshot (d82.5) and compare dT/dt measured between snaps ──
sn33 = np.load('results/global_diag90_blob_3d/snap_00033.npy')
T33 = sn33[0]
dTdt_meas = (T33 - T3) / (2.5 * 86400.0)
# per-term split of the MEASURED dT/dt by correlating with each diagnosed term
conv_term = np.array(conv_T_raw)
Vb = V
num = float(np.sum(conv_term * dTdt_meas * Vb))
den = float(np.sum(conv_term * conv_term * Vb))
print('regression conv_coeff (1 if conv is the only term acting): %.4f' % (num / den))
# measured total heating
print('measured dT/dt heating d80->d82.5: %+.1f W/m2' % _gW(dTdt_meas))
