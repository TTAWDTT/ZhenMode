"""Test the semi-implicit Crank-Nicolson free-surface step as a drop-in
replacement for _free_surface_step_fd. Validates the phase-error hypothesis:
if CN equilibrates the baroclinic adjustment (bounded KE_bc, no blowup) where
FB grows unbounded, the fix is confirmed.
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
                               _step_impl, _barotropic_velocity,
                               _divergence_conservative, _gradient_conservative,
                               _polar_cap_weights)
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

# --- Semi-implicit Crank-Nicolson free-surface step ---
def _cap(field2d, p):
    nc = p.polar_cap_rows
    if nc <= 0:
        return field2d
    nt = p.polar_cap_taper
    wts = _polar_cap_weights(nc, nt); nb = nc + nt
    wts_b = wts.reshape(1, nb)
    def _band(field, wm_band):
        s = field * wm_band
        wsum = jnp.maximum(jnp.sum(wm_band, axis=0, keepdims=True), 1.0)
        zmean = jnp.sum(s, axis=0, keepdims=True) / wsum
        zmean = jnp.broadcast_to(zmean, (p.nx, nb)) * wm_band
        return wts_b * zmean + (1.0 - wts_b) * (field * wm_band)
    south = _band(field2d[:, :nb], p.wet_mask[:, :nb])
    north = _band(field2d[:, -nb:], p.wet_mask[:, -nb:])
    return jnp.concatenate([south, field2d[:, nb:-nb], north], axis=1)

def _lap_consistent(eta, p):
    gx, gy = _gradient_conservative(eta, p)
    return _divergence_conservative(gx, gy, p)

def _free_surface_step_cn(eta, u, v, p, F_rho_x=None, F_rho_y=None, dt_half=None):
    """Semi-implicit Crank-Nicolson shallow-water free-surface step.

    Trapezoidal (CN) coupling of (eta, ubt, vbt) with implicit linear bottom
    drag. The spectral solver integrates this SW system EXACTLY (matrix exp,
    correct phase => baroclinic adjustment equilibrates). The explicit FB step
    has phase error => never equilibrates => unbounded baroclinic KE growth.
    CN is 2nd-order with correct phase (the FD analogue of the matrix exp).

    Substituting the CN momentum eqn into continuity gives a Helmholtz eqn for
    eta^{n+1}:  (I - (dt/2)^2 g H_sw L) eta^{n+1} = rhs,  L = div(grad).
    Solved by CG (operator is SPD on the wet domain; -L is PSD).
    """
    if dt_half is None:
        dt_half = p.dt / 2.0
    ubt, vbt = _barotropic_velocity(u, v, p)
    F_x = jnp.zeros_like(ubt); F_y = jnp.zeros_like(vbt)
    if F_rho_x is not None:
        F_x = F_x + F_rho_x; F_y = F_y + F_rho_y
    F_x = F_x + p.tau_x_2d / (RHO_0 * p.H_sw)
    F_y = F_y + p.tau_y_2d / (RHO_0 * p.H_sw)

    r_bt = p.r_bot if p.bottom_friction == 'linear' else 0.0
    drag = 1.0 / (1.0 + 0.5 * r_bt * dt_half)   # CN drag denominator (1/(1+r*a))
    drag_old = (1.0 - 0.5 * r_bt * dt_half) * drag  # (1-r*a)/(1+r*a) old-vel coef
    a = 0.5 * dt_half                            # CN time weight (=dt_half/2)
    wm = p.wet_mask
    ubt_m = ubt * wm; vbt_m = vbt * wm
    div_n = _divergence_conservative(ubt_m, vbt_m, p)
    lap_n = _lap_consistent(eta, p)
    div_F = _divergence_conservative(F_x * wm, F_y * wm, p)
    # CN: U^{n+1}=drag*[(1-r*a)*U^n + a*(-g grad(eta^n+eta^{n+1}) + 2F)]
    # Continuity (CN): eta^{n+1}=eta^n - a*H*(div U^{n+1}+div U^n)
    # => [I + a^2*g*H*drag*L] eta^{n+1}
    #    = eta^n - a*H*[(1+drag_old)*div U^n + 2*a*drag*div F] - a^2*g*H*drag*L(eta^n)
    coef = a * a * G_EARTH * p.H_sw * drag
    rhs = (eta - a * p.H_sw * (1.0 + drag_old) * div_n
           - 2.0 * coef * div_F - coef * lap_n) * wm
    def helmholtz_op(e):
        return wm * (e + coef * _lap_consistent(e, p))
    eta_new, info = jsp.cg(helmholtz_op, rhs, maxiter=150, tol=1e-8)
    eta_new = eta_new * wm
    sw_decay = jnp.exp(-p.sponge_rate_2d * dt_half)
    eta_new = eta_new * sw_decay
    eta_new = _cap(eta_new, p)
    grad_eta_x, grad_eta_y = _gradient_conservative(eta_new, p)
    ubt_new = drag_old * ubt + drag * a * (-G_EARTH * grad_eta_x + 2.0 * F_x)
    vbt_new = drag_old * vbt + drag * a * (-G_EARTH * grad_eta_y + 2.0 * F_y)
    ubt_new = ubt_new * wm * sw_decay; vbt_new = vbt_new * wm * sw_decay
    ubt_new = _cap(ubt_new, p); vbt_new = _cap(vbt_new, p)
    delta_ubt = (ubt_new - ubt)[:, :, None]
    delta_vbt = (vbt_new - vbt)[:, :, None]
    u_new = u + delta_ubt; v_new = v + delta_vbt
    v_new = v_new * p.interior_mask_z
    return eta_new, u_new, v_new

# Monkeypatch: replace FB with CN in the linear half-step path
G._free_surface_step_fd = _free_surface_step_cn
step_cn = jax.jit(lambda s: _step_impl(s, params))

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

def run(label, step_fn, n=1400):
    state = init_state_fn(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    print(f"\n=== {label} ===")
    print(f"{'stp':>5} {'E':>12} {'max|eta|':>10} {'max|u|':>10} {'KE_bc':>12}")
    for k in range(n):
        state = step_fn(state)
        if (k+1) in (200,400,800,1000,1200,1400):
            E=float(energy_j(state)); me, mu, kebc, fin = diag_j(state)
            print(f"{k+1:>5} {E:>12.4e} {float(me):>10.4e} {float(mu):>10.4e} {float(kebc):>12.4e}")
            if not bool(fin): print(f"  NaN step {k+1}"); break

# First: no-advection (the hypothesis test — should EQUILIBRATE like spectral)
def _zero_adv_u(u, v, w, p):
    z = jnp.zeros_like(v); return z, z
def _zero_adv_s(T, u, v, w, p):
    return jnp.zeros_like(T)
G._advection_flux_form = _zero_adv_u
G._advection_scalar = _zero_adv_s
step_cn_noadv = jax.jit(lambda s: _step_impl(s, params))
run("CN free-surface, NO advection (hypothesis: should equilibrate)", step_cn_noadv, n=800)
print("\nDONE.")
