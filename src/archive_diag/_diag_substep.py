"""Cheaper explicit-path test: sub-step the free-surface FB step (smaller internal dt).

The spectral particular solution is exact; the explicit FB is its first-order
approx. In the dt->0 limit the explicit FB converges to the exact forced-SW
solution (with within-step balancing via many small steps). If sub-stepping
stabilizes F_rho, it's an explicit-only fix (no Helmholtz) god may prefer.
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
import jax_solver_global as JSG
from jax_solver_global import make_solver_global, JaxStateG, RHO_0, G_EARTH
from forcing import air_temp_profile, heat_flux_meridional
from woa_data import get_initial_fields

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
T_init, S_init = get_initial_fields(grid)
T_init = np.array(T_init); S_init = np.array(S_init)
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])
wm = np.array(grid.wet_mask) > 0.5
step, init_state_fn, _ = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0)

_orig_lhs = JSG._linear_half_step

def make_lhs(nsub):
    def _lhs_sub(state, p, dt_half):
        u = state.u + p.nu_h * JSG._laplacian_h(state.u, p) * dt_half
        v = state.v + p.nu_h * JSG._laplacian_h(state.v, p) * dt_half
        T = state.T + p.kappa_h * JSG._laplacian_h(state.T, p) * dt_half
        S = state.S + p.kappa_h * JSG._laplacian_h(state.S, p) * dt_half
        u = u * p.wet_mask_z; v = v * p.wet_mask_z; T = T * p.wet_mask_z; S = S * p.wet_mask_z
        decay = jnp.exp(-p.sponge_rate * dt_half)
        u = u * decay; v = v * decay
        T = p.T_clim_3d + (T - p.T_clim_3d) * decay; S = p.S_clim_3d + (S - p.S_clim_3d) * decay
        u, v = JSG._coriolis_rotation_2d(u, v, p.f, dt_half)
        F_rho_x, F_rho_y = JSG._compute_bt_rho_pgf(state, p)
        # sub-step the free-surface FB step: nsub steps of dt_half/nsub
        eta = state.eta
        sub_dt = dt_half / nsub
        for _ in range(nsub):
            eta, u, v = JSG._free_surface_step_fd(eta, u, v, p, F_rho_x, F_rho_y, sub_dt)
        return JaxStateG(u, v, T, S, eta)
    return _lhs_sub

for nsub in [1, 5, 20]:
    JSG._linear_half_step = make_lhs(nsub)
    state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    print(f"\n=== free-surface sub-stepped x{nsub} (cap off, no wind, F_rho ON, r_bot=1e-3) ===")
    print(f"{'stp':>5} {'max|eta|':>10} {'max|u|':>10} {'max|T|':>9}")
    for k in range(2000):
        state = step(state)
        if (k+1)%400==0:
            e=np.array(state.eta); u=np.array(state.u); T=np.array(state.T)
            print(f"{k+1:>5} {float(np.max(np.abs(e))):>10.4e} {float(np.max(np.abs(u))):>10.4e} {float(np.max(np.abs(T))):>9.3f}")
        if not np.isfinite(np.array(state.eta)).all():
            print(f"  NaN step {k+1}"); break
    else:
        print(f"  STABLE through 2000 steps")
JSG._linear_half_step = _orig_lhs
