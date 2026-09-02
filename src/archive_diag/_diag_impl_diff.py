"""Test: is the discriminator the DIFFUSION step (spectral exact exp(-nu*k^2*dt)
strongly damps grid-scale; FD explicit fwd-Euler weakly damps at CFL)?
Replace FD's explicit diffusion with implicit (Crank-Nicolson) diffusion in the
linear half-step and test no-adv injection. If implicit diffusion equilibrates
where explicit doesn't, the diffusion discretization is the discriminator.
"""
import os, sys
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, 'src')
import jax
import jax.numpy as jnp
import numpy as np
import jax.scipy.sparse.linalg as jsp
from dataclasses import replace
from config import DEFAULT_CONFIG, PhysicsConfig, GlobalGridConfig
from grid import make_global_grid
import jax_solver_global as G
from jax_solver_global import (make_solver_global, JaxStateG, RHO_0, G_EARTH,
                               _step_impl, _laplacian_h, _d2_dz2)
from forcing import air_temp_profile, heat_flux_meridional

bathy = DEFAULT_CONFIG.bathymetry_file
gcfg = replace(GlobalGridConfig(), lat_max=60.0, ny=120)
grid = make_global_grid(gcfg, bathy, smooth_passes=30, min_depth=100.0)
Q_heat = heat_flux_meridional(grid, Q0=0.0)
d = np.load('src/_cache_init.npz')
T_init = d['T']; S_init = d['S']
tau_x = np.zeros((grid.nx, grid.ny)); tau_y = np.zeros((grid.nx, grid.ny))
T_atm = air_temp_profile(grid, T_init[:, :, 0])
dz = np.array(grid.dz); dz_norm = dz/dz.sum(); H=4000.0

physics = replace(PhysicsConfig(), nu_h=1e3, nu_bi=0, kappa_bi=0, r_bot=1e-3)
step, init_state_fn, _, params = make_solver_global(
    grid, physics, 60.0, forcing=(tau_x, tau_y, Q_heat), eos_type='linear',
    T_atm=T_atm, lambda_bulk=0.0, sponge_days=0.0, sponge_cells=0,
    T_init=T_init, S_init=S_init, polar_cap_rows=0, polar_cap_taper=0,
    return_params=True)

# Implicit (CN) horizontal diffusion on the full 3D field (block-diagonal in z).
def _impl_diff_h(field3d, p, nu, dt):
    """CN implicit horizontal diffusion: (I - a*L)u_new = (I + a*L)u_old, a=nu*dt/2.
    Operates on the full (nx,ny,nz) field; L is the _laplacian_h (per-z, 3D-aware)."""
    if nu == 0.0:
        return field3d
    wmz = p.wet_mask_z
    a = 0.5 * nu * dt
    rhs = (field3d + a * _laplacian_h(field3d, p)) * wmz
    def op(x):
        return wmz * (x - a * _laplacian_h(x, p))
    sol, info = jsp.cg(op, rhs, maxiter=80, tol=1e-7)
    return sol * wmz

# Patch _linear_half_step to use implicit horizontal diffusion (keep FB free surface)
_orig_linear = G._linear_half_step
def _linear_impl_diff(state, p, dt_half):
    # Implicit CN horizontal diffusion (replace the explicit fwd-Euler lines)
    u = _impl_diff_h(state.u, p, p.nu_h, dt_half)
    v = _impl_diff_h(state.v, p, p.nu_h, dt_half)
    T = _impl_diff_h(state.T, p, p.kappa_h, dt_half)
    S = _impl_diff_h(state.S, p, p.kappa_h, dt_half)
    # vertical diffusion explicit (unchanged)
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
    u, v = G._coriolis_rotation_2d(u, v, p.f, dt_half)
    F_rho_x, F_rho_y = G._compute_bt_rho_pgf(state, p)
    eta, u, v = G._free_surface_step_fd(state.eta, u, v, p, F_rho_x, F_rho_y, dt_half)
    return JaxStateG(u, v, T, S, eta)
G._linear_half_step = _linear_impl_diff

# no-advection test
def _zero_adv_u(u, v, w, p):
    return jnp.zeros_like(v), jnp.zeros_like(v)
def _zero_adv_s(T, u, v, w, p):
    return jnp.zeros_like(T)
G._advection_flux_form = _zero_adv_u
G._advection_scalar = _zero_adv_s
step_impl = jax.jit(lambda s: _step_impl(s, params))

wm_j = jnp.array(grid.wet_mask); dz_norm_j = jnp.array(dz_norm); wm_3d = wm_j[:,:,None]
@jax.jit
def energy_j(state):
    e=state.eta; u=state.u; v=state.v
    ua=0.5*(u[...,:-1]+u[...,1:]); va=0.5*(v[...,:-1]+v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1); vbt=jnp.sum(va*dz_norm_j,-1)
    return (0.5*H*jnp.sum((ubt**2+vbt**2)*wm_j)+0.5*G_EARTH*H*jnp.sum(e**2*wm_j))
@jax.jit
def diag_j(state):
    ua=0.5*(state.u[...,:-1]+state.u[...,1:]); va=0.5*(state.v[...,:-1]+state.v[...,1:])
    ubt=jnp.sum(ua*dz_norm_j,-1,keepdims=True); vbt=jnp.sum(va*dz_norm_j,-1,keepdims=True)
    kebc=0.5*H*jnp.sum(((ua-ubt)**2+(va-vbt)**2)*wm_3d)
    return (jnp.max(jnp.abs(state.eta)), jnp.max(jnp.abs(state.u)), kebc,
            jnp.isfinite(state.eta).all())

state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
print("=== IMPLICIT (CN) diffusion + FB free-surface, NO advection ===")
print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10} {'KE_bc':>12}")
for k in range(800):
    state = step_impl(state)
    if (k+1) in (200,400,800):
        E=float(energy_j(state)); me, mu, kebc, fin = diag_j(state)
        print(f"{k+1:>5} {E:>12.4e} {float(me):>10.4e} {float(mu):>10.4e} {float(kebc):>12.4e}")
        if not bool(fin): print(f"  NaN step {k+1}"); break
print("\nDONE.")
