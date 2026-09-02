"""DECISIVE: is the 3D baroclinic PGF in the momentum RESIDUAL (explicit RK2)
the injector? Patch _compute_momentum_residual to subtract the FULL 3D PGF
(eta + density), so the residual carries NO pressure gradient at all — only
advection + vertical diffusion + bottom friction. The full PGF then lives ONLY
in the linear half-step (as the barotropic part) — which is stable alone.
If this stabilizes => the residual's baroclinic PGF under RK2 is the injector,
and the fix is to handle the 3D PGF in the linear/implicit step, not the
explicit residual.
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
                               _step_impl, _compute_momentum_tendency,
                               _laplacian_h, _biharmonic_h, _compute_hydrostatic_pressure,
                               _d_dx, _d_dy, _compute_bt_rho_pgf)
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
            jnp.sum(state.eta*wm_j), jnp.isfinite(state.eta).all())

# Patched residual: subtract the FULL 3D PGF (eta + density) so residual has
# NO pressure gradient (only advection + vert diff + bottom friction).
def _residual_no_pgf(state, p):
    dudt, dvdt = _compute_momentum_tendency(state, p)
    dudt = dudt - p.nu_h * _laplacian_h(state.u, p)
    dvdt = dvdt - p.nu_h * _laplacian_h(state.v, p)
    if p.nu_bi > 0.0:
        dudt = dudt + p.nu_bi * _biharmonic_h(state.u, p)
        dvdt = dvdt + p.nu_bi * _biharmonic_h(state.v, p)
    dudt = dudt - p.f[:, :, None] * state.v
    dvdt = dvdt + p.f[:, :, None] * state.u
    # FULL 3D PGF (eta barotropic + density baroclinic)
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
G._compute_momentum_residual = _residual_no_pgf

step_nopgf = jax.jit(lambda s: _step_impl(s, params))

def run(label, step_fn, n=1400):
    state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    print(f"\n=== {label} ===")
    print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10}")
    for k in range(n):
        state = step_fn(state)
        if (k+1) in (200, 600, 1000, 1400):
            E=float(energy_j(state)); me, mu, se, fin = diag_j(state)
            print(f"{k+1:>5} {E:>12.4e} {float(me):>10.4e} {float(mu):>10.4e}")
            if not bool(fin): print(f"  NaN step {k+1}"); break

run("FULL step, residual has NO PGF (full 3D PGF only in linear step)", step_nopgf)
print("\nDONE.")
