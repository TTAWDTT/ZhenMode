"""
Long-integration stability diagnostic on real WOA fields (no forcing).

The 2h stability scan only ran 24 steps at dt=300, but the original
blowup from report 5.3 appeared at step ~30-40 of a ~100-step run
(8 hours).  This script reproduces the slow-growing instability over
long integration and tracks the energy partition through each Strang
sub-step, plus term-targeted diagnostics to isolate which physical
tendency drives the growth.

Run:  python src/diag_long_run.py  (> logs/diag_long_run.log 2>&1)
"""
import sys
import os

import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG, RHO_0, G_EARTH
from grid import make_grid
from jax_solver import (
    make_solver, JaxState, _compute_params,
    _linear_half_step, _explicit_full_step,
    _compute_momentum_tendency, _compute_tracer_tendency,
    _compute_hydrostatic_pressure, _density_anomaly,
)
from woa_data import get_initial_fields

# JIT sub-step functions with params bound in a closure (so JIT treats
# params — including string fields like bottom_friction — as static,
# mirroring make_solver's @jax.jit). Returns (half_step, full_step), each
# taking (state, dt).
def _make_steps(params):
    @jax.jit
    def half_step(state, dt_h):
        return _linear_half_step(state, params, dt_h)

    @jax.jit
    def full_step(state, dt_f):
        return _explicit_full_step(state, params, dt_f)

    return half_step, full_step

grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
physics = DEFAULT_CONFIG.physics
nx, ny, nz = grid.nx, grid.ny, grid.nz
dx, dy = grid.dx, grid.dy
dz_sum = float(np.sum(grid.dz))

T_init, S_init = get_initial_fields(grid)
print(f"WOA T range: [{T_init.min():.2f}, {T_init.max():.2f}] C")
print(f"WOA S range: [{S_init.min():.2f}, {S_init.max():.2f}] PSU")


def total_energy(state):
    """Total (kinetic + potential) integrated energy."""
    ke = 0.5 * RHO_0 * jnp.sum(state.u ** 2 + state.v ** 2) * dz_sum
    pe = 0.5 * RHO_0 * G_EARTH * jnp.sum(state.eta ** 2) * dx * dy
    return float(ke + pe)


def domain_T_range(state):
    return (float(state.T.min()), float(state.T.max()))


# ── Reproduce long-run instability (no forcing, dt=300) ──
print("\n" + "=" * 72)
print("REPRO: WOA no-forcing, dt=300, 150 steps (12.5 h)")
print("=" * 72)
dt = 300.0
params = _compute_params(grid, physics, dt, forcing=None)
state = JaxState(
    u=jnp.zeros((nx, ny, nz)),
    v=jnp.zeros((nx, ny, nz)),
    T=jnp.array(T_init), S=jnp.array(S_init),
    eta=jnp.zeros((nx, ny)),
)

E0 = total_energy(state)
half_step, full_step = _make_steps(params)
blow_step = None
for i in range(1, 151):
    s_L1 = half_step(state, dt / 2.0)
    s_N = full_step(s_L1, dt)
    state = half_step(s_N, dt / 2.0)

    mu = float(jnp.max(jnp.abs(state.u)))
    Tlo, Thi = domain_T_range(state)
    E = total_energy(state)
    dE_L1 = total_energy(s_L1) - E0 if i == 1 else total_energy(s_L1) - E0
    nan = bool(jnp.isnan(state.u).any() or jnp.isnan(state.T).any())
    e_ratio = (E / E0) if E0 > 0 else float('nan')

    if i <= 5 or i % 10 == 0 or nan:
        print(f"  step {i:3d}: max|u|={mu:9.4e}  T=[{Tlo:7.2f},{Thi:7.2f}]  "
              f"E={E:9.4e}  E/E0={e_ratio:8.4f}  {'NaN!' if nan else ''}")

    if nan or mu > 1e6:
        blow_step = i
        print(f"  -> BLOWUP at step {i}")
        break

print(f"  RESULT: blow_step = {blow_step}, "
      f"(report 5.3 saw blowup ~step 30-40)")


# ── Term-targeted: which tendency grows? (single-step sensitivity) ──
print("\n" + "=" * 72)
print("TERM-TARGETED: magnitude of each tendency at t=0 (WOA no-forcing)")
print("=" * 72)
state0 = JaxState(
    u=jnp.zeros((nx, ny, nz)),
    v=jnp.zeros((nx, ny, nz)),
    T=jnp.array(T_init), S=jnp.array(S_init),
    eta=jnp.zeros((nx, ny)),
)

# Tracer tendencies with individual terms
dTdt, dSdt = _compute_tracer_tendency(state0, params)
print(f"  full dT/dt: max|.|={float(jnp.max(jnp.abs(dTdt))):.4e} °C/s")
print(f"       dT/dt x dt(300) = {float(jnp.max(jnp.abs(dTdt)))*300:.4e} °C/step")

# Momentum tendencies
dudt, dvdt = _compute_momentum_tendency(state0, params)
du_amp = float(jnp.max(jnp.abs(dudt)))
print(f"  full du/dt: max|.|={du_amp:.4e} m/s^2")
print(f"       du/dt x dt(300) = {du_amp*300:.4e} m/s/step")

# Decompose tracer tendency into advection / diffusion / heat (JIT'd,
# params bound in closure so string fields like bottom_friction are static)
from jax_solver import _compute_vertical_velocity, _advection_scalar, _laplacian_h, _d2_dz2

@jax.jit
def _term_decomp_params(t, u, v, w):
    adv_T = _advection_scalar(t, u, v, w, params)
    diff_h_T = params.kappa_h * _laplacian_h(t, params)
    diff_v_T = params.kappa_v * _d2_dz2(t, params)
    return adv_T, diff_h_T, diff_v_T

@jax.jit
def _jw_params(state):
    return _compute_vertical_velocity(state, params)

w = _jw_params(state0)
adv_T, diff_h_T, diff_v_T = _term_decomp_params(state0.T, state0.u, state0.v, w)
heat_T = params.Q_heat_2d[:, :, None] / (RHO_0 * 3992.0 * params.dz_surface) * params.surface_mask
print(f"\n  Tracer terms (max|.|):")
print(f"    advection = {float(jnp.max(jnp.abs(adv_T))):.4e}")
print(f"    diff_h    = {float(jnp.max(jnp.abs(diff_h_T))):.4e}")
print(f"    diff_v    = {float(jnp.max(jnp.abs(diff_v_T))):.4e}")
print(f"    heat      = {float(jnp.max(jnp.abs(heat_T))):.4e}")
print(f"  advection x dt(300) = {float(jnp.max(jnp.abs(adv_T)))*300:.4e} °C/step")


# ── Does WOA T field have a strong vertical mean gradient that,
#     combined with w from imbalance, advects T vertically? ──
print("\n" + "=" * 72)
print("VERTICAL: baroclinic initial imbalance check")
print("=" * 72)
# Baroclinic PGF magnitude from initial density
rho_prime = _density_anomaly(T_init, S_init, params)
pgf_x, pgf_y = _compute_momentum_tendency(state0, params)
print(f"  initial rho' range: [{float(rho_prime.min()):.2f}, "
      f"{float(rho_prime.max()):.2f}] kg/m^3")
print(f"  initial full du/dt max = {float(jnp.max(jnp.abs(dudt))):.4e}")
print(f"  initial full dv/dt max = {float(jnp.max(jnp.abs(dvdt))):.4e}")
