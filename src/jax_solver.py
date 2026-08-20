"""
JAX Ocean Solver — JIT-compiled hydrostatic primitive equations.

XLA compiler fuses all element-wise operations into optimized kernels,
eliminates intermediate array allocations, and enables GPU acceleration
with zero code change.

Key optimizations vs numpy version:
  1. @jax.jit on entire step() — XLA fuses all operators into one kernel
  2. Spectral-space derivative chaining: FFT once, apply ik, IFFT —
     saves ~50% FFT calls vs numpy version (which FFTs twice per derivative)
  3. Pre-computed decay factors, wavenumbers, z-grid coefficients —
     all baked into XLA graph as constants
  4. Purely functional — no array mutation, no Python overhead per step

Usage:
    from jax_solver import make_solver
    step_fn, init_state, diag_fn = make_solver(grid, physics, dt=300.0)
    state = init_state()
    for _ in range(n_steps):
        state = step_fn(state)
    rho, p, w = diag_fn(state)
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from collections import namedtuple

from config import RHO_0, ALPHA_T, BETA_S, C_P, G_EARTH


# ── State ────────────────────────────────────────────────────────────
# Namedtuple = automatic JAX pytree, no registration needed.

JaxState = namedtuple('JaxState', ['u', 'v', 'T', 'S', 'eta'])


# ── Solver parameters ────────────────────────────────────────────────
# All pre-computed static data. When captured in @jax.jit closure,
# XLA treats these as constants and constant-folds them.

SolverParams = namedtuple('SolverParams', [
    # Spectral (reshaped for 3D broadcasting)
    'kx', 'ky', 'k2',              # (nx,1,1), (1,ny,1), (nx,ny,1)
    'dealias_2d',                  # (nx,ny,1) combined 2/3 dealias mask
    'decay_u', 'decay_T',          # (nx,ny,1) diffusion decay for dt/2
    # Grid
    'f',                           # (nx,ny) full Coriolis
    'f0',                          # scalar
    'nx', 'ny', 'nz',              # ints (static)
    # Vertical grid
    'dz_denom_interior',           # (1,1,nz-2) for d_dz
    'dz_bnd_top', 'dz_bnd_bot',   # scalars for d_dz boundaries
    'd2z_hm', 'd2z_hp', 'd2z_denom',  # (1,1,nz-2) for d2_dz2
    'd2z_h0_top', 'd2z_h0_bot',   # scalars for d2_dz2 boundaries
    'dz_3d',                       # (1,1,nz-1) layer thicknesses
    'dz_surface',                  # scalar |z[0]-z[1]|
    # Forcing masks
    'surface_mask',                # (1,1,nz) 1 at k=0
    'bottom_mask',                 # (1,1,nz) 1 at k=-1
    # Physics scalars
    'nu_h', 'nu_v', 'kappa_h', 'kappa_v',
    'T_ref', 'S_ref',
    'tau_x', 'tau_y', 'Q_heat', 'r_bot',
    # Free surface (2D spectral)
    'kx_2d', 'ky_2d',           # (nx,1), (1,ny) for 2D ops
    'omega_sw',                  # (nx,ny) shallow water frequency
    'sw_sin', 'sw_cos',         # (nx,ny) Rodrigues coefficients for dt/2
    'H_mean',                    # scalar, mean ocean depth [m] (bathymetry)
    'H_sw',                      # scalar, effective SW depth = sum(dz) [m]
    'dz_norm',                   # (1,1,nz-1) dz/H_sw for barotropic vel
    # Time
    'dt',
])


def _compute_params(grid, physics, dt):
    """Pre-compute all static JAX arrays from grid and physics."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    dt_half = dt / 2.0

    # ── Wavenumbers ──
    kx_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    kx = kx_1d.reshape(nx, 1, 1)
    ky = ky_1d.reshape(1, ny, 1)
    k2 = kx ** 2 + ky ** 2  # (nx, ny, 1)

    # ── Dealias mask (2/3 rule) ──
    cx = nx // 3
    cy = ny // 3
    mx = jnp.zeros(nx).at[:cx].set(1.0).at[-cx:].set(1.0)
    my = jnp.zeros(ny).at[:cy].set(1.0).at[-cy:].set(1.0)
    dealias_2d = (mx[:, None] * my[None, :])[:, :, None]  # (nx, ny, 1)

    # ── Diffusion decay factors (for linear half-step, dt/2) ──
    decay_u = jnp.exp(-physics.nu_h * k2 * dt_half)
    decay_T = jnp.exp(-physics.kappa_h * k2 * dt_half)

    # ── Coriolis ──
    f = jnp.array(grid.f)
    f0 = float(grid.f0)

    # ── Vertical grid coefficients ──
    z = jnp.array(grid.z)
    dz = jnp.array(grid.dz)
    dz_3d = dz.reshape(1, 1, -1)

    # d_dz interior: (u[k+1] - u[k-1]) / (dz_up + dz_dn)
    dz_up = z[1:-1] - z[:-2]
    dz_dn = z[2:] - z[1:-1]
    dz_denom_interior = (dz_up + dz_dn).reshape(1, 1, -1)
    dz_bnd_top = float(z[1] - z[0])
    dz_bnd_bot = float(z[-1] - z[-2])

    # d2_dz2 interior: non-uniform second derivative
    hm = jnp.abs(z[:-2] - z[1:-1])
    hp = jnp.abs(z[2:] - z[1:-1])
    d2z_denom = (hm * hp * (hm + hp) / 2.0).reshape(1, 1, -1)
    d2z_hm = hm.reshape(1, 1, -1)
    d2z_hp = hp.reshape(1, 1, -1)
    d2z_h0_top = float(jnp.abs(z[1] - z[0]))
    d2z_h0_bot = float(jnp.abs(z[-1] - z[-2]))

    dz_surface = float(jnp.abs(z[0] - z[1]))

    # ── Forcing masks ──
    surface_mask = jnp.zeros(nz).at[0].set(1.0).reshape(1, 1, -1)
    bottom_mask = jnp.zeros(nz).at[-1].set(1.0).reshape(1, 1, -1)

    # ── Free surface parameters ──
    # Mean ocean depth (bathymetry) and effective SW depth
    H_mean = float(grid.depth[grid.ocean_mask].mean())
    H_sw = float(jnp.sum(dz))   # z-grid vertical span = sum(dz)

    # 2D wavenumbers for 2D spectral ops (no trailing dimension)
    kx_2d = kx_1d.reshape(nx, 1)
    ky_2d = ky_1d.reshape(1, ny)
    k2_2d = kx_2d ** 2 + ky_2d ** 2  # (nx, ny)

    # Shallow water frequency: omega = sqrt(g * H_sw * k^2)
    # H_sw = sum(dz) ensures consistency between barotropic velocity
    # extraction, wave frequency, and projection back to 3D fields.
    omega_sw = jnp.sqrt(G_EARTH * H_sw * k2_2d)  # (nx, ny)

    # Rodrigues coefficients for half-step (dt/2)
    sw_sin = jnp.sin(omega_sw * dt_half)   # (nx, ny)
    sw_cos = jnp.cos(omega_sw * dt_half)   # (nx, ny)

    # Normalized layer thicknesses for barotropic velocity
    # dz_norm sums to 1, so ubt = sum(u_avg * dz_norm) is a true
    # depth-average over the z-grid span, consistent with projection.
    dz_norm = dz.reshape(1, 1, -1) / H_sw   # (1, 1, nz-1)

    return SolverParams(
        kx=kx, ky=ky, k2=k2,
        dealias_2d=dealias_2d,
        decay_u=decay_u, decay_T=decay_T,
        f=f, f0=f0, nx=nx, ny=ny, nz=nz,
        dz_denom_interior=dz_denom_interior,
        dz_bnd_top=dz_bnd_top, dz_bnd_bot=dz_bnd_bot,
        d2z_hm=d2z_hm, d2z_hp=d2z_hp, d2z_denom=d2z_denom,
        d2z_h0_top=d2z_h0_top, d2z_h0_bot=d2z_h0_bot,
        dz_3d=dz_3d, dz_surface=dz_surface,
        surface_mask=surface_mask, bottom_mask=bottom_mask,
        nu_h=physics.nu_h, nu_v=physics.nu_v,
        kappa_h=physics.kappa_h, kappa_v=physics.kappa_v,
        T_ref=physics.T_ref, S_ref=physics.S_ref,
        tau_x=physics.tau_x, tau_y=physics.tau_y,
        Q_heat=physics.Q_heat, r_bot=physics.r_bot,
        kx_2d=kx_2d, ky_2d=ky_2d,
        omega_sw=omega_sw, sw_sin=sw_sin, sw_cos=sw_cos,
        H_mean=H_mean, H_sw=H_sw, dz_norm=dz_norm,
        dt=dt,
    )


def _init_state(grid, physics):
    """Create initial rest state as JAX arrays."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    return JaxState(
        u=jnp.zeros((nx, ny, nz)),
        v=jnp.zeros((nx, ny, nz)),
        T=jnp.full((nx, ny, nz), physics.T_ref),
        S=jnp.full((nx, ny, nz), physics.S_ref),
        eta=jnp.zeros((nx, ny)),
    )


# ── Spectral operators ───────────────────────────────────────────────
# Module-level pure functions. When called inside @jax.jit with params
# captured from closure, XLA treats all params as constants.

def _d_dx(u, p):
    """Spectral du/dx = IFFT(i*kx * FFT(u))."""
    u_hat = jnp.fft.fft(u, axis=0)
    return jnp.real(jnp.fft.ifft(1j * p.kx * u_hat, axis=0))


def _d_dy(u, p):
    """Spectral du/dy = IFFT(i*ky * FFT(u))."""
    u_hat = jnp.fft.fft(u, axis=1)
    return jnp.real(jnp.fft.ifft(1j * p.ky * u_hat, axis=1))


def _laplacian_h(u, p):
    """Horizontal Laplacian = IFFT(-k^2 * FFT(u))."""
    u_hat = jnp.fft.fft2(u, axes=(0, 1))
    return jnp.real(jnp.fft.ifft2(-p.k2 * u_hat, axes=(0, 1)))


def _divergence_h(u, v, p):
    """du/dx + dv/dy."""
    return _d_dx(u, p) + _d_dy(v, p)


def _d_dx_2d(eta, p):
    """Spectral d(eta)/dx for 2D field (nx, ny)."""
    eta_hat = jnp.fft.fft(eta, axis=0)
    return jnp.real(jnp.fft.ifft(1j * p.kx_2d * eta_hat, axis=0))


def _d_dy_2d(eta, p):
    """Spectral d(eta)/dy for 2D field (nx, ny)."""
    eta_hat = jnp.fft.fft(eta, axis=1)
    return jnp.real(jnp.fft.ifft(1j * p.ky_2d * eta_hat, axis=1))


def _d_dz(u, p):
    """Vertical first derivative, non-uniform grid. Concatenate boundaries."""
    du_interior = (u[..., 2:] - u[..., :-2]) / p.dz_denom_interior
    du_top = (u[..., 1:2] - u[..., 0:1]) / p.dz_bnd_top
    du_bot = (u[..., -1:] - u[..., -2:-1]) / p.dz_bnd_bot
    return jnp.concatenate([du_top, du_interior, du_bot], axis=-1)


def _d2_dz2(u, p):
    """Vertical second derivative, non-uniform grid."""
    d2u_interior = (
        u[..., 2:] * p.d2z_hm + u[..., :-2] * p.d2z_hp
        - u[..., 1:-1] * (p.d2z_hm + p.d2z_hp)
    ) / p.d2z_denom
    d2u_top = (u[..., 2:3] - 2 * u[..., 1:2] + u[..., 0:1]) / (p.d2z_h0_top ** 2)
    d2u_bot = (u[..., -3:-2] - 2 * u[..., -2:-1] + u[..., -1:]) / (p.d2z_h0_bot ** 2)
    return jnp.concatenate([d2u_top, d2u_interior, d2u_bot], axis=-1)


def _advection_flux_form(u, v, p):
    """
    Flux-form advection with spectral-space derivative chaining.

    Optimization: FFT each flux once, apply dealias + derivative operator
    (ik) in spectral space, then IFFT. This saves ~50% FFT calls vs the
    numpy version which FFTs twice per derivative (once for dealias,
    once for derivative).

    3 FFT + 4 IFFT  (vs 7 FFT + 7 IFFT in numpy version)
    """
    uu = u * u
    uv = u * v
    vv = v * v

    uu_hat = jnp.fft.fft2(uu, axes=(0, 1)) * p.dealias_2d
    uv_hat = jnp.fft.fft2(uv, axes=(0, 1)) * p.dealias_2d
    vv_hat = jnp.fft.fft2(vv, axes=(0, 1)) * p.dealias_2d

    d_dx_uu = jnp.real(jnp.fft.ifft2(1j * p.kx * uu_hat, axes=(0, 1)))
    d_dy_uv = jnp.real(jnp.fft.ifft2(1j * p.ky * uv_hat, axes=(0, 1)))
    d_dx_uv = jnp.real(jnp.fft.ifft2(1j * p.kx * uv_hat, axes=(0, 1)))
    d_dy_vv = jnp.real(jnp.fft.ifft2(1j * p.ky * vv_hat, axes=(0, 1)))

    adv_u = -(d_dx_uu + d_dy_uv)
    adv_v = -(d_dx_uv + d_dy_vv)
    return adv_u, adv_v


def _advection_scalar(T, u, v, p):
    """
    Scalar advection with spectral-space derivative chaining.

    2 FFT + 2 IFFT  (vs 4 FFT + 4 IFFT in numpy version)
    """
    uT_hat = jnp.fft.fft2(u * T, axes=(0, 1)) * p.dealias_2d
    vT_hat = jnp.fft.fft2(v * T, axes=(0, 1)) * p.dealias_2d

    d_dx_uT = jnp.real(jnp.fft.ifft2(1j * p.kx * uT_hat, axes=(0, 1)))
    d_dy_vT = jnp.real(jnp.fft.ifft2(1j * p.ky * vT_hat, axes=(0, 1)))

    return -(d_dx_uT + d_dy_vT)


def _linear_step_diffusion(u, decay):
    """Exact diffusion: u(t+dt) = IFFT(decay * FFT(u)). Decay pre-computed."""
    u_hat = jnp.fft.fft2(u, axes=(0, 1))
    return jnp.real(jnp.fft.ifft2(decay * u_hat, axes=(0, 1)))


# ── Physics ──────────────────────────────────────────────────────────

def _density_anomaly(T, S, p):
    """rho' = rho - rho_0 [kg/m^3]."""
    return RHO_0 * (-ALPHA_T * (T - p.T_ref) + BETA_S * (S - p.S_ref))


def _compute_hydrostatic_pressure(state, p):
    """Full hydrostatic pressure via cumulative trapezoidal integration."""
    rho_prime = _density_anomaly(state.T, state.S, p)
    rho_avg = 0.5 * (rho_prime[..., :-1] + rho_prime[..., 1:])
    dp = G_EARTH * rho_avg * p.dz_3d  # (nx, ny, nz-1)

    p_bc = jnp.zeros_like(state.T)
    p_bc = p_bc.at[..., 1:].set(jnp.cumsum(dp, axis=-1))

    p_bt = RHO_0 * G_EARTH * state.eta[:, :, None]
    return p_bt + p_bc


def _compute_pressure_gradient(state, p):
    """Horizontal pressure gradient force per unit mass."""
    pressure = _compute_hydrostatic_pressure(state, p)
    pgf_x = -_d_dx(pressure, p) / RHO_0
    pgf_y = -_d_dy(pressure, p) / RHO_0
    return pgf_x, pgf_y


def _compute_momentum_tendency(state, p):
    """du/dt, dv/dt for hydrostatic primitive equations."""
    adv_u, adv_v = _advection_flux_form(state.u, state.v, p)

    f_3d = p.f[:, :, None]
    cor_u = f_3d * state.v
    cor_v = -f_3d * state.u

    pgf_x, pgf_y = _compute_pressure_gradient(state, p)

    diff_h_u = p.nu_h * _laplacian_h(state.u, p)
    diff_h_v = p.nu_h * _laplacian_h(state.v, p)

    diff_v_u = p.nu_v * _d2_dz2(state.u, p)
    diff_v_v = p.nu_v * _d2_dz2(state.v, p)

    wind_factor = 1.0 / (RHO_0 * p.dz_surface)
    wind_u = p.tau_x * wind_factor * p.surface_mask
    wind_v = p.tau_y * wind_factor * p.surface_mask

    bot_u = -p.r_bot * state.u * p.bottom_mask
    bot_v = -p.r_bot * state.v * p.bottom_mask

    dudt = adv_u + cor_u + pgf_x + diff_h_u + diff_v_u + wind_u + bot_u
    dvdt = adv_v + cor_v + pgf_y + diff_h_v + diff_v_v + wind_v + bot_v
    return dudt, dvdt


def _compute_tracer_tendency(state, p):
    """dT/dt, dS/dt for tracer transport."""
    adv_T = _advection_scalar(state.T, state.u, state.v, p)
    adv_S = _advection_scalar(state.S, state.u, state.v, p)

    diff_h_T = p.kappa_h * _laplacian_h(state.T, p)
    diff_h_S = p.kappa_h * _laplacian_h(state.S, p)

    diff_v_T = p.kappa_v * _d2_dz2(state.T, p)
    diff_v_S = p.kappa_v * _d2_dz2(state.S, p)

    heat_factor = p.Q_heat / (RHO_0 * C_P * p.dz_surface)
    heat_T = heat_factor * p.surface_mask

    dTdt = adv_T + diff_h_T + diff_v_T + heat_T
    dSdt = adv_S + diff_h_S + diff_v_S
    return dTdt, dSdt


def _compute_vertical_velocity(state, p):
    """Diagnose w from horizontal continuity. w=0 at bottom."""
    div_h = _divergence_h(state.u, state.v, p)
    div_avg = 0.5 * (div_h[..., :-1] + div_h[..., 1:])
    integrand = div_avg * p.dz_3d

    w = jnp.zeros_like(state.u)
    w = w.at[..., :-1].set(-jnp.cumsum(integrand[..., ::-1], axis=-1)[..., ::-1])
    return w


# ── Integrator (Strang splitting IMEX) ───────────────────────────────

def _coriolis_rotation(u, v, f0, dt):
    """Exact f-plane Coriolis rotation."""
    angle = f0 * dt
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    u_new =  cos_a * u + sin_a * v
    v_new = -sin_a * u + cos_a * v
    return u_new, v_new


def _barotropic_velocity(u, v, p):
    """Depth-averaged (barotropic) horizontal velocity.

    ubt = sum(0.5*(u[k]+u[k+1]) * dz_norm, axis=-1)

    Returns:
        ubt, vbt: (nx, ny) barotropic velocities
    """
    u_avg = 0.5 * (u[..., :-1] + u[..., 1:])
    v_avg = 0.5 * (v[..., :-1] + v[..., 1:])
    ubt = jnp.sum(u_avg * p.dz_norm, axis=-1)
    vbt = jnp.sum(v_avg * p.dz_norm, axis=-1)
    return ubt, vbt


def _free_surface_step(eta, u, v, p):
    """Exact linear shallow water step in spectral space (no Coriolis).

    Solves the coupled (eta, ubt, vbt) system per wavenumber:
        d(eta)/dt  = -H * (ikx*ubt + iky*vbt)     [continuity]
        d(ubt)/dt  = -g * ikx * eta               [x-momentum]
        d(vbt)/dt  = -g * iky * eta               [y-momentum]

    via matrix exponential:  expm(A*dt) = I + s*A + c2*A^2
    where s = sin(w*dt)/w,  c2 = (1-cos(w*dt))/w^2,  w = sqrt(g*H*k^2).

    The k=0 mode (omega=0) is identity - mean mass and momentum conserved.
    Barotropic velocity changes are projected uniformly back to 3D fields.
    """
    ubt, vbt = _barotropic_velocity(u, v, p)

    # Spectral space
    eta_hat = jnp.fft.fft2(eta)
    ubt_hat = jnp.fft.fft2(ubt)
    vbt_hat = jnp.fft.fft2(vbt)

    # Safe-division coefficients (k=0 mode -> 0, leaving mean unchanged)
    k2_2d = p.kx_2d ** 2 + p.ky_2d ** 2
    sin_div_w = jnp.where(p.omega_sw > 0.0, p.sw_sin / p.omega_sw, 0.0)
    omc_div_k2 = jnp.where(k2_2d > 0.0, (1.0 - p.sw_cos) / k2_2d, 0.0)

    ikx = 1j * p.kx_2d   # (nx, 1)
    iky = 1j * p.ky_2d   # (1, ny)

    # eta:  eta_new = cos(w dt) * eta  -  H_sw * sin(w dt)/w * (ikx*ubt + iky*vbt)
    eta_hat_new = (p.sw_cos * eta_hat
                   - p.H_sw * sin_div_w * (ikx * ubt_hat + iky * vbt_hat))

    # ubt:  ubt + s*(-g*ikx)*eta  -  (1-cos)/k^2 * (kx^2*ubt + kx*ky*vbt)
    ubt_hat_new = (ubt_hat
                   - G_EARTH * ikx * sin_div_w * eta_hat
                   - omc_div_k2 * (p.kx_2d ** 2 * ubt_hat + p.kx_2d * p.ky_2d * vbt_hat))

    # vbt:  vbt + s*(-g*iky)*eta  -  (1-cos)/k^2 * (kx*ky*ubt + ky^2*vbt)
    vbt_hat_new = (vbt_hat
                   - G_EARTH * iky * sin_div_w * eta_hat
                   - omc_div_k2 * (p.kx_2d * p.ky_2d * ubt_hat + p.ky_2d ** 2 * vbt_hat))

    # Back to physical space
    eta_new = jnp.real(jnp.fft.ifft2(eta_hat_new))
    ubt_new = jnp.real(jnp.fft.ifft2(ubt_hat_new))
    vbt_new = jnp.real(jnp.fft.ifft2(vbt_hat_new))

    # Project barotropic delta back to 3D velocity (uniform over depth)
    delta_ubt = (ubt_new - ubt)[:, :, None]
    delta_vbt = (vbt_new - vbt)[:, :, None]
    u_new = u + delta_ubt
    v_new = v + delta_vbt

    return eta_new, u_new, v_new


def _linear_half_step(state, p, dt_half):
    """Linear half-step: diffusion + Coriolis + free surface (all exact)."""
    u = _linear_step_diffusion(state.u, p.decay_u)
    v = _linear_step_diffusion(state.v, p.decay_u)
    T = _linear_step_diffusion(state.T, p.decay_T)
    S = _linear_step_diffusion(state.S, p.decay_T)
    u, v = _coriolis_rotation(u, v, p.f0, dt_half)
    eta, u, v = _free_surface_step(state.eta, u, v, p)
    return JaxState(u, v, T, S, eta)


def _explicit_full_step(state, p, dt):
    """Nonlinear tendencies via explicit Euler, minus linear parts."""
    dudt, dvdt = _compute_momentum_tendency(state, p)
    dTdt, dSdt = _compute_tracer_tendency(state, p)

    # Subtract linear parts (handled by _linear_half_step)
    dudt = dudt - p.nu_h * _laplacian_h(state.u, p)
    dvdt = dvdt - p.nu_h * _laplacian_h(state.v, p)
    dTdt = dTdt - p.kappa_h * _laplacian_h(state.T, p)
    dSdt = dSdt - p.kappa_h * _laplacian_h(state.S, p)
    dudt = dudt - p.f0 * state.v
    dvdt = dvdt + p.f0 * state.u

    # Subtract barotropic PGF: -g*grad(eta) is handled by _free_surface_step
    bt_pgf_x = -G_EARTH * _d_dx_2d(state.eta, p)
    bt_pgf_y = -G_EARTH * _d_dy_2d(state.eta, p)
    dudt = dudt - bt_pgf_x[:, :, None]
    dvdt = dvdt - bt_pgf_y[:, :, None]

    u_new = state.u + dudt * dt
    v_new = state.v + dvdt * dt
    T_new = state.T + dTdt * dt
    S_new = state.S + dSdt * dt
    return JaxState(u_new, v_new, T_new, S_new, state.eta)


def _step_impl(state, p):
    """Strang splitting: L(dt/2) -> N(dt) -> L(dt/2)."""
    dt_half = p.dt / 2.0
    state = _linear_half_step(state, p, dt_half)
    state = _explicit_full_step(state, p, p.dt)
    state = _linear_half_step(state, p, dt_half)
    return state


# ── Public API ───────────────────────────────────────────────────────

def make_solver(grid, physics, dt):
    """
    Create a JIT-compiled ocean solver.

    All static parameters (wavenumbers, decay factors, z-grid coefficients,
    physics constants) are pre-computed and baked into the XLA computation
    graph as constants. The returned step_fn takes only a JaxState and
    returns a new JaxState — no Python overhead per step.

    Args:
        grid: OceanGrid (from grid.py)
        physics: PhysicsConfig (from config.py)
        dt: time step [s]

    Returns:
        step_fn: JIT-compiled (state: JaxState) -> JaxState
        init_state: () -> JaxState (rest state)
        diagnostics: JIT-compiled (state) -> (rho, pressure, w)
    """
    params = _compute_params(grid, physics, dt)

    @jax.jit
    def step(state):
        return _step_impl(state, params)

    @jax.jit
    def diagnostics(state):
        rho = RHO_0 * (1.0 - ALPHA_T * (state.T - params.T_ref)
                        + BETA_S * (state.S - params.S_ref))
        pressure = _compute_hydrostatic_pressure(state, params)
        w = _compute_vertical_velocity(state, params)
        return rho, pressure, w

    def init_state():
        return _init_state(grid, physics)

    return step, init_state, diagnostics


# ── Benchmark ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import DEFAULT_CONFIG
    from grid import make_grid

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    dt = 300.0

    print("=== JAX Ocean Solver Benchmark ===")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}")
    print()

    step_fn, init_state, diag_fn = make_solver(grid, physics, dt)

    state = init_state()
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    state = JaxState(
        u=state.u + jax.random.normal(key1, state.u.shape) * 0.01,
        v=state.v + jax.random.normal(key2, state.v.shape) * 0.01,
        T=state.T + jax.random.normal(key3, state.T.shape) * 0.01,
        S=state.S,
        eta=state.eta,
    )

    # Warmup (includes JIT compilation)
    print("Compiling JIT...")
    t0 = time.perf_counter()
    state = step_fn(state)
    jax.block_until_ready(state.u)
    t1 = time.perf_counter()
    print(f"Compilation + first step: {t1-t0:.2f}s")
    print()

    # Timed
    n = 10
    t0 = time.perf_counter()
    for _ in range(n):
        state = step_fn(state)
    jax.block_until_ready(state.u)
    t1 = time.perf_counter()
    ms_per_step = (t1 - t0) / n * 1000
    print(f"Steps: {n}")
    print(f"Time/step: {ms_per_step:.1f} ms")
    print(f"Simulated time: {n * dt / 86400:.2f} days in {t1-t0:.2f}s wall time")
