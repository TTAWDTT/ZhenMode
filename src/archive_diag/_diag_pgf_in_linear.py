"""Test: does moving ALL density PGF into the linear half-step (removing it from
the RK2 residual) kill the slow growth?

Currently the density PGF is split:
  - barotropic part (bt_rho_pgf) => _linear_half_step free-surface forcing
  - depth-varying part (full PGF - barotropic) => _compute_momentum_residual
The dt-independence suggests the SPLIT itself is the error. This test keeps the
FULL 3D density PGF in the linear step (as a body force on the 3D velocity,
added right after Coriolis) and ZEROES it in the residual, so the density force
is handled in ONE place coherently with Coriolis + free surface.

Implementation: monkeypatch _compute_momentum_residual to subtract the FULL 3D
density PGF (not just barotropic) so the residual carries NO density PGF; and
add the full 3D density PGF as a force in a custom linear half-step.
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
import jax_solver_global as G
from jax_solver_global import (make_solver_global, JaxStateG, RHO_0, G_EARTH,
                               _step_impl, _compute_momentum_residual,
                               _compute_momentum_tendency, _linear_half_step,
                               _compute_hydrostatic_pressure, _compute_pressure_gradient,
                               _coriolis_rotation_2d, _d_dx, _d_dy, _laplacian_h,
                               _d2_dz2, _biharmonic_h, _compute_bt_rho_pgf,
                               _free_surface_step_fd, _density_anomaly, FDPhysParams)
from forcing import air_temp_profile, heat_flux_meridional
from woa_data import get_initial_fields

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])
wm = np.array(grid.wet_mask) > 0.5
dz = np.array(grid.dz); dz_norm = dz/dz.sum(); H=4000.0

physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

wm_j = jnp.array(grid.wet_mask); dz_norm_j = jnp.array(dz_norm)
@jax.jit
def energy_j(state):
    e=state.eta; u=state.u; v=state.v
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1); vbt=jnp.sum(va*dz_norm_j,-1)
    return (0.5*H*jnp.sum((ubt**2+vbt**2)*wm_j)+0.5*G_EARTH*H*jnp.sum(e**2*wm_j))
@jax.jit
def diag_j(state):
    return (jnp.max(jnp.abs(state.eta)), jnp.max(jnp.abs(state.u)),
            jnp.isfinite(state.eta).all())

# --- Custom momentum residual: subtract the FULL 3D density PGF so residual
#     carries NO density force (only advection + vert diff + bottom friction). ---
def _residual_no_density_pgf(state, p):
    dudt, dvdt = _compute_momentum_tendency(state, p)
    dudt = dudt - p.nu_h * _laplacian_h(state.u, p)
    dvdt = dvdt - p.nu_h * _laplacian_h(state.v, p)
    if p.nu_bi > 0.0:
        dudt = dudt + p.nu_bi * _biharmonic_h(state.u, p)
        dvdt = dvdt + p.nu_bi * _biharmonic_h(state.v, p)
    dudt = dudt - p.f[:, :, None] * state.v
    dvdt = dvdt + p.f[:, :, None] * state.u
    # subtract the FULL 3D PGF (eta barotropic + density baroclinic)
    pressure = _compute_hydrostatic_pressure(state, p)
    pgf_x = -_d_dx(pressure, p) / RHO_0
    pgf_y = -_d_dy(pressure, p) / RHO_0
    dudt = dudt - pgf_x
    dvdt = dvdt - pgf_y
    # barotropic wind
    dudt = dudt - p.tau_x_2d[:, :, None] / (RHO_0 * p.H_sw)
    dvdt = dvdt - p.tau_y_2d[:, :, None] / (RHO_0 * p.H_sw)
    dudt = dudt * p.wet_mask_z
    dvdt = dvdt * p.wet_mask_z
    return dudt, dvdt

# --- Custom linear half-step: adds the FULL 3D density PGF as a body force
#     (coherent with Coriolis + free surface), so the density force is handled
#     in ONE place. Applied as a forward-Euler force after Coriolis, before FS. ---
def _linear_half_step_with_full_pgf(state, p, dt_half):
    u = state.u + p.nu_h * _laplacian_h(state.u, p) * dt_half
    v = state.v + p.nu_h * _laplacian_h(state.v, p) * dt_half
    T = state.T + p.kappa_h * _laplacian_h(state.T, p) * dt_half
    S = state.S + p.kappa_h * _laplacian_h(state.S, p) * dt_half
    u = u + p.nu_v * _d2_dz2(state.u, p) * dt_half
    v = v + p.nu_v * _d2_dz2(state.v, p) * dt_half
    T = T + p.kappa_v * _d2_dz2(state.T, p) * dt_half
    S = S + p.kappa_v * _d2_dz2(state.S, p) * dt_half
    u = u * p.wet_mask_z; v = v * p.wet_mask_z
    T = T * p.wet_mask_z; S = S * p.wet_mask_z
    decay = jnp.exp(-p.sponge_rate * dt_half)
    u = u * decay; v = v * decay
    T = p.T_clim_3d + (T - p.T_clim_3d) * decay
    S = p.S_clim_3d + (S - p.S_clim_3d) * decay
    # Coriolis rotation (exact)
    u, v = _coriolis_rotation_2d(u, v, p.f, dt_half)
    # FULL 3D density PGF as a forward-Euler body force (coherent location)
    pgf_x, pgf_y = _compute_pressure_gradient(state, p)
    u = u + pgf_x * dt_half
    v = v + pgf_y * dt_half
    u = u * p.wet_mask_z; v = v * p.wet_mask_z
    # free surface (barotropic density PGF forcing, as before)
    F_rho_x, F_rho_y = _compute_bt_rho_pgf(state, p)
    eta, u, v = _free_surface_step_fd(state.eta, u, v, p, F_rho_x, F_rho_y, dt_half)
    return JaxStateG(u, v, T, S, eta)

# disable advection to isolate the density-PGF splitting effect
G._advection_flux_form = lambda u, v, w, p: (jnp.zeros_like(u), jnp.zeros_like(v))
G._advection_scalar = lambda T, u, v, w, p: jnp.zeros_like(T)

# patched step: use full-PGF linear step + no-density-PGF residual
def _step_patched(state, p):
    dt_half = p.dt / 2.0
    state = _linear_half_step_with_full_pgf(state, p, dt_half)
    # nonlinear step with patched residual (no density PGF)
    state = G._explicit_full_step(state, p, p.dt)
    state = _linear_half_step_with_full_pgf(state, p, dt_half)
    return state
# patch the residual used inside _explicit_full_step
G._compute_momentum_residual = _residual_no_density_pgf

step_patched = jax.jit(lambda s: _step_patched(s, params))

def run(label, step_fn, n=1200):
    state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    E0 = float(energy_j(state))
    print(f"\n=== {label} ===  E0={E0:.4e}")
    print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10}")
    for k in range(n):
        state = step_fn(state)
        if (k+1)%200==0:
            E=float(energy_j(state)); me, mu, fin = diag_j(state)
            print(f"{k+1:>5} {E:>12.4e} {float(me):>10.4e} {float(mu):>10.4e}")
            if not bool(fin): print(f"  NaN step {k+1}"); break

# baseline (current split) for reference
step_base = jax.jit(lambda s: _step_impl(s, params))
run("BASELINE (split: bt_rho in linear, depth-varying in residual), advection OFF", step_base)
run("PATCHED (full 3D density PGF in linear step, NONE in residual), advection OFF", step_patched)
print("\nDONE.")
