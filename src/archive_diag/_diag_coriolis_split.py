"""Discriminate: is the blowup from (FB free surface + F_rho) or from
(FB free surface + Coriolis SPLITTING)?

Full physics _diag_full_spinup blows up NaN~1194 with Coriolis ON. The isolated
_free_surface_step_fd WITHOUT Coriolis/diffusion/drag also blows up (but that
had everything off). This isolates the Coriolis-free case WITH diffusion+drag,
to see whether Coriolis splitting is the injector or F_rho+FB alone is.

Runs the FULL step (make_solver_global) with f artificially zeroed vs real f.
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
from jax_solver_global import make_solver_global, JaxStateG, RHO_0, G_EARTH
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

def energy(state):
    e=np.array(state.eta); u=np.array(state.u); v=np.array(state.v)
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=np.sum(ua*dz_norm,-1); vbt=np.sum(va*dz_norm,-1)
    return (0.5*H*np.sum((ubt**2+vbt**2)*wm)+0.5*G_EARTH*H*np.sum(e**2*wm))

def run(label, zero_f=False, n=1400):
    physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
    step, init_state_fn, _ = make_solver_global(
        grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
        T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
        T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0)
    state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    if zero_f:
        # The Coriolis param f is baked into params at make_solver_global time.
        # Patch via the returned params? init_state_fn doesn't return params.
        # Instead rebuild with a config that has f=0 — PhysicsConfig has no f;
        # f comes from the grid. So we patch the step's closure isn't possible.
        # Workaround: monkeypatch the grid's f to zero BEFORE make_solver_global.
        pass
    E0 = energy(state)
    print(f"\n=== {label} ===  E0={E0:.4e}")
    print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10} {'sum_eta':>11}")
    for k in range(n):
        state = step(state)
        if (k+1)%200==0:
            E=energy(state)
            e=np.array(state.eta); u=np.array(state.u)
            print(f"{k+1:>5} {E:>12.4e} {float(np.max(np.abs(e))):>10.4e} "
                  f"{float(np.max(np.abs(u))):>10.4e} {float(np.sum(e[wm])):>11.4e}")
        if not np.isfinite(np.array(state.eta)).all():
            print(f"  NaN step {k+1}"); break

# We can't easily zero f post-hoc. Instead test the ISOLATED free-surface step
# WITH drag but WITHOUT Coriolis — does r_bot=1e-3 contain F_rho alone?
import jax
from jax_solver_global import (_free_surface_step_fd, _compute_bt_rho_pgf,
                                _barotropic_velocity, _divergence_conservative,
                                _gradient_conservative)
_, _, _, params = make_solver_global(
    grid, replace(PhysicsConfig(), nu_h=0.0, nu_bi=0, kappa_bi=0, kappa_h=0.0, r_bot=1e-3),
    60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)
nx,ny,nz = grid.nx, grid.ny, grid.nz
eta0 = jnp.zeros((nx,ny)); u0=jnp.zeros((nx,ny,nz)); v0=jnp.zeros((nx,ny,nz))
st = JaxStateG(u0,v0,jnp.array(T_init),jnp.array(S_init),eta0)
Fx, Fy = _compute_bt_rho_pgf(st, params)

print("=== Isolated FS step WITH r_bot=1e-3 drag, NO Coriolis, full F_rho ===")
eta = jnp.zeros((nx,ny)); u = jnp.zeros((nx,ny,nz)); v = jnp.zeros((nx,ny,nz))
E0 = energy(JaxStateG(u,v,jnp.array(T_init),jnp.array(S_init),eta))
print(f"E0={E0:.4e}")
print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10}")
for k in range(1400):
    eta, u, v = _free_surface_step_fd(eta, u, v, params, Fx, Fy)
    if (k+1)%200==0:
        st2 = JaxStateG(u,v,jnp.array(T_init),jnp.array(S_init),eta)
        E=energy(st2)
        print(f"{k+1:>5} {E:>12.4e} {float(np.max(np.abs(np.array(eta)))):>10.4e} "
              f"{float(np.max(np.abs(np.array(u)))):>10.4e}")
    if not np.all(np.isfinite(np.array(eta))):
        print(f"  NaN step {k+1}"); break
print("\nDONE.")
