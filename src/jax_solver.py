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
    'k4',                          # (nx,ny,1) k4 = k2^2 for biharmonic
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
    'nu_bi', 'kappa_bi',           # biharmonic horizontal viscosity/diffusivity
    'T_ref', 'S_ref',
    # Surface forcing (2D fields, nx x ny)
    'tau_x_2d', 'tau_y_2d', 'Q_heat_2d', 'r_bot', 'cd', 'bottom_friction',
    # Free surface (2D spectral)
    'kx_2d', 'ky_2d',           # (nx,1), (1,ny) for 2D ops
    'omega_sw',                  # (nx,ny) shallow water frequency
    'sw_sin', 'sw_cos',         # (nx,ny) Rodrigues coefficients for dt/2
    'H_mean',                    # scalar, mean ocean depth [m] (bathymetry)
    'H_sw',                      # scalar, effective SW depth = sum(dz) [m]
    'dz_norm',                   # (1,1,nz-1) dz/H_sw for barotropic vel
    # Barotropic wind forcing (for forced free surface step)
    'F_bt_x_hat', 'F_bt_y_hat', # (nx,ny) FFT of barotropic wind forcing
    'sw_sin_div_w',             # (nx,ny) sin(w*dt/2)/w, k=0 limit = dt/2
    # Time
    'dt',
    # EOS
    'eos_type',                   # 'linear' or 'unesco'
    # Smagorinsky
    'smag_cs',                    # Smagorinsky constant (0 = disabled)
    'dx',                         # grid spacing for Smagorinsky length scale
])


def _compute_params(grid, physics, dt, forcing=None, eos_type='linear'):
    """Pre-compute all static JAX arrays from grid and physics.

    Args:
        grid: OceanGrid
        physics: PhysicsConfig
        dt: time step [s]
        forcing: optional (tau_x_2d, tau_y_2d, Q_heat_2d) tuple of
            (nx, ny) arrays. If None, defaults to zero forcing (rest).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    dt_half = dt / 2.0

    # ── Wavenumbers ──
    kx_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    kx = kx_1d.reshape(nx, 1, 1)
    ky = ky_1d.reshape(1, ny, 1)
    k2 = kx ** 2 + ky ** 2  # (nx, ny, 1)
    k4 = k2 ** 2            # (nx, ny, 1) for biharmonic (scale-selective)

    # ── Dealias mask (2/3 rule) ──
    cx = nx // 3
    cy = ny // 3
    mx = jnp.zeros(nx).at[:cx].set(1.0).at[-cx:].set(1.0)
    my = jnp.zeros(ny).at[:cy].set(1.0).at[-cy:].set(1.0)
    dealias_2d = (mx[:, None] * my[None, :])[:, :, None]  # (nx, ny, 1)

    # ── Diffusion decay factors (for linear half-step, dt/2) ──
    # Combined Laplacian (k^2) + biharmonic (k^4). Biharmonic damps
    # grid-scale modes ∝ k^4 far more than large scales, so it kills the
    # baroclinic eddy-instability blowup while leaving basin flow intact.
    decay_u = jnp.exp(-(physics.nu_h * k2 + physics.nu_bi * k4) * dt_half)
    decay_T = jnp.exp(-(physics.kappa_h * k2 + physics.kappa_bi * k4) * dt_half)

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

    # ── Surface forcing (2D fields) ──
    # forcing is a (tau_x_2d, tau_y_2d, Q_heat_2d) tuple of (nx, ny) arrays,
    # or None for rest-state (zero) forcing. Convert to jnp arrays so they
    # are captured as XLA constants in the JIT closure.
    if forcing is None:
        tau_x_2d = jnp.zeros((nx, ny))
        tau_y_2d = jnp.zeros((nx, ny))
        Q_heat_2d = jnp.zeros((nx, ny))
    else:
        tau_x_2d, tau_y_2d, Q_heat_2d = (jnp.array(f) for f in forcing)

    # ── Barotropic wind forcing (for forced free surface step) ──
    # F_bt = tau / (rho_0 * H_sw) is the depth-uniform body force.
    # The free surface step solves the forced shallow water equation
    # exactly, preventing the Strang splitting resonance that occurs
    # when wind forcing is only in the explicit step.
    F_bt_x = tau_x_2d / (RHO_0 * H_sw)
    F_bt_y = tau_y_2d / (RHO_0 * H_sw)
    F_bt_x_hat = jnp.fft.fft2(F_bt_x)
    F_bt_y_hat = jnp.fft.fft2(F_bt_y)

    # sin(omega*dt/2)/omega with k=0 limit = 0 (mean mode handled separately)
    sw_sin_div_w = jnp.where(omega_sw > 0.0, sw_sin / omega_sw, 0.0)

    return SolverParams(
        kx=kx, ky=ky, k2=k2, k4=k4,
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
        nu_bi=physics.nu_bi, kappa_bi=physics.kappa_bi,
        T_ref=physics.T_ref, S_ref=physics.S_ref,
        tau_x_2d=tau_x_2d, tau_y_2d=tau_y_2d,
        Q_heat_2d=Q_heat_2d, r_bot=physics.r_bot,
        omega_sw=omega_sw, sw_sin=sw_sin, sw_cos=sw_cos,
        kx_2d=kx_2d, ky_2d=ky_2d,
        H_mean=H_mean, H_sw=H_sw, dz_norm=dz_norm,
        F_bt_x_hat=F_bt_x_hat, F_bt_y_hat=F_bt_y_hat,
        sw_sin_div_w=sw_sin_div_w,
        dt=dt,
        eos_type=eos_type,
        cd=physics.cd,
        bottom_friction=physics.bottom_friction,
        smag_cs=physics.smag_cs,
        dx=dx,
    )


def _init_state(grid, physics, T_init=None, S_init=None):
    """Create initial state as JAX arrays.

    If T_init/S_init are provided (e.g., from WOA climatology), they
    are used as the initial temperature/salinity fields. Otherwise,
    uniform T_ref/S_ref is used (rest state).

    Velocity and eta are always initialized to zero.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if T_init is not None:
        T = jnp.array(T_init)
        S = jnp.array(S_init)
    else:
        T = jnp.full((nx, ny, nz), physics.T_ref)
        S = jnp.full((nx, ny, nz), physics.S_ref)
    return JaxState(
        u=jnp.zeros((nx, ny, nz)),
        v=jnp.zeros((nx, ny, nz)),
        T=T,
        S=S,
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


def _biharmonic_h(u, p):
    """Horizontal biharmonic = IFFT(k^4 * FFT(u)).

    ∇⁴ operator. Biharmonic dissipation is scale-selective: it damps
    grid-scale modes ∝ k⁴ but leaves large-scale flow nearly untouched,
    so it kills the baroclinic eddy-instability blowup without smearing
    the basin-scale circulation. Note the sign: ∇⁴ = ∇²(∇²) has spectrum
    +k⁴. The Schr operator convolution (u - nu_bi*∇⁴u) gives decay
    exp(-nu_bi*k⁴*dt) in the linear half-step.
    """
    u_hat = jnp.fft.fft2(u, axes=(0, 1))
    return jnp.real(jnp.fft.ifft2(p.k4 * u_hat, axes=(0, 1)))


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


def _dealias_h(field, p):
    """Apply 2/3 dealiasing rule to a 3D field (horizontal axes only).

    Nonlinear products in physical space (e.g. u * du/dx) generate
    spurious high-wavenumber energy via aliasing.  This FFTs the
    product, zeros the upper 1/3 of wavenumbers, and IFFTs back.
    """
    field_hat = jnp.fft.fft2(field, axes=(0, 1))
    field_hat = field_hat * p.dealias_2d
    return jnp.real(jnp.fft.ifft2(field_hat, axes=(0, 1)))


def _advection_flux_form(u, v, w, p):
    """
    3D advective-form momentum advection (dealiased).

    adv_u = -(u*du/dx + v*du/dy + w*du/dz)
    adv_v = -(u*dv/dx + v*dv/dy + w*dv/dz)

    The advective form (rather than flux form) avoids spurious
    source terms proportional to div_h(u) that arise in the flux
    form d(uu)/dx + d(uv)/dy = u*du/dx + v*du/dy + u*div_h when
    vertical advection is absent.  In a 3D hydrostatic model,
    div_h(u) != 0 (balanced by dw/dz), so the flux form creates a
    non-physical source proportional to horizontal divergence.

    Horizontal derivatives: spectral (exact).
    Vertical derivative: non-uniform finite difference.
    Nonlinear products: dealiased via 2/3 rule (horizontal axes).
    """
    du_dx = _d_dx(u, p)
    du_dy = _d_dy(u, p)
    dv_dx = _d_dx(v, p)
    dv_dy = _d_dy(v, p)

    du_dz = _d_dz(u, p)
    dv_dz = _d_dz(v, p)

    adv_u = -(u * du_dx + v * du_dy + w * du_dz)
    adv_v = -(u * dv_dx + v * dv_dy + w * dv_dz)
    return _dealias_h(adv_u, p), _dealias_h(adv_v, p)


def _advection_scalar(T, u, v, w, p):
    """
    3D advective-form scalar advection (dealiased).

    adv_T = -(u*dT/dx + v*dT/dy + w*dT/dz)

    Advective form (rather than flux form) avoids spurious source
    terms proportional to T*div_h(u) that arise in the flux form
    d(uT)/dx + d(vT)/dy = u*dT/dx + v*dT/dy + T*div_h when vertical
    advection is absent.  In a 3D hydrostatic model, div_h(u) != 0
    (balanced by dw/dz), so the flux form creates a non-physical
    source proportional to horizontal divergence.

    Horizontal derivatives: spectral (exact).
    Vertical derivative: non-uniform finite difference.
    Nonlinear products: dealiased via 2/3 rule (horizontal axes).
    """
    dT_dx = _d_dx(T, p)
    dT_dy = _d_dy(T, p)
    dT_dz = _d_dz(T, p)

    adv_T = -(u * dT_dx + v * dT_dy + w * dT_dz)
    return _dealias_h(adv_T, p)


def _linear_step_diffusion(u, decay):
    """Exact diffusion: u(t+dt) = IFFT(decay * FFT(u)). Decay pre-computed."""
    u_hat = jnp.fft.fft2(u, axes=(0, 1))
    return jnp.real(jnp.fft.ifft2(decay * u_hat, axes=(0, 1)))


# ── Physics ──────────────────────────────────────────────────────────

# ── UNESCO 1980 nonlinear EOS (JAX-compatible) ─────────────────────
# Reference: Fofonoff & Millard (1983), UNESCO Tech. Papers in Marine
# Science No. 44. Valid at atmospheric pressure (0 dbar gauge).
# All functions use jnp for JIT compatibility.

def _rho_smow_jax(T):
    """Density of Standard Mean Ocean Water (pure water) [kg/m^3]."""
    return (
        999.842594
        + 6.793952e-2 * T
        - 9.095290e-3 * T**2
        + 1.001685e-4 * T**3
        - 1.120083e-6 * T**4
        + 6.536332e-9 * T**5
    )


def _b_jax(T):
    """Salinity coefficient B(T) for UNESCO EOS."""
    return (
        8.24493e-1
        - 4.0899e-3 * T
        + 7.6438e-5 * T**2
        - 8.2467e-7 * T**3
        + 5.3875e-9 * T**4
    )


def _c_jax(T):
    """Salinity coefficient C(T) for UNESCO EOS."""
    return -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T**2


def _density_unesco_jax(T, S):
    """UNESCO 1980 nonlinear EOS at 1 atm, JAX-compatible.

    rho(T,S) = rho_smow(T) + B(T)*S + C(T)*S^(3/2) + D*S^2
    """
    _D = 4.8314e-4
    smow = _rho_smow_jax(T)
    S_safe = jnp.maximum(S, 0.0)
    S_sqrt = jnp.sqrt(S_safe)
    return smow + _b_jax(T) * S + _c_jax(T) * S_sqrt * S + _D * S**2


def _density_anomaly(T, S, p):
    """rho' = rho - rho_0 [kg/m^3]. Branches on eos_type."""
    if p.eos_type == 'unesco':
        return _density_unesco_jax(T, S) - RHO_0
    else:
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


def _compute_bt_rho_pgf(state, p):
    """Barotropic (depth-averaged) PGF from density anomalies.

    The full hydrostatic PGF has both barotropic and baroclinic
    components.  The barotropic component (depth-averaged) drives
    the fast external mode and must be handled by the exact SW
    solver in the linear step, not the explicit nonlinear step.

    Computes:  F = -1/rho_0 * grad( depth_mean(p_bc) )
    where p_bc is the baroclinic pressure from density anomalies.

    Returns:
        bt_rho_pgf_x, bt_rho_pgf_y: (nx, ny) 2D barotropic forcing
    """
    rho_prime = _density_anomaly(state.T, state.S, p)
    rho_avg = 0.5 * (rho_prime[..., :-1] + rho_prime[..., 1:])
    dp = G_EARTH * rho_avg * p.dz_3d

    p_bc = jnp.zeros_like(state.T)
    p_bc = p_bc.at[..., 1:].set(jnp.cumsum(dp, axis=-1))

    p_bc_avg = jnp.sum(
        0.5 * (p_bc[..., :-1] + p_bc[..., 1:]) * p.dz_norm, axis=-1
    )

    bt_rho_pgf_x = -_d_dx_2d(p_bc_avg, p) / RHO_0
    bt_rho_pgf_y = -_d_dy_2d(p_bc_avg, p) / RHO_0
    return bt_rho_pgf_x, bt_rho_pgf_y


def _compute_momentum_tendency(state, p):
    """du/dt, dv/dt for hydrostatic primitive equations."""
    w = _compute_vertical_velocity(state, p)
    adv_u, adv_v = _advection_flux_form(state.u, state.v, w, p)

    f_3d = p.f[:, :, None]
    cor_u = f_3d * state.v
    cor_v = -f_3d * state.u

    pgf_x, pgf_y = _compute_pressure_gradient(state, p)

    diff_h_u = p.nu_h * _laplacian_h(state.u, p)
    diff_h_v = p.nu_h * _laplacian_h(state.v, p)

    # Smagorinsky subgrid closure (state-dependent viscosity)
    if p.smag_cs > 0:
        dudx = _d_dx(state.u, p)
        dvdy = _d_dy(state.v, p)
        dudy = _d_dy(state.u, p)
        dvdx = _d_dx(state.v, p)
        def_strain = jnp.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2)
        nu_smg = (p.smag_cs * p.dx)**2 * def_strain
        diff_h_u = diff_h_u + nu_smg * _laplacian_h(state.u, p)
        diff_h_v = diff_h_v + nu_smg * _laplacian_h(state.v, p)

    diff_v_u = p.nu_v * _d2_dz2(state.u, p)
    diff_v_v = p.nu_v * _d2_dz2(state.v, p)

    wind_factor = 1.0 / (RHO_0 * p.dz_surface)
    wind_u = p.tau_x_2d[:, :, None] * wind_factor * p.surface_mask
    wind_v = p.tau_y_2d[:, :, None] * wind_factor * p.surface_mask

    # Bottom friction
    if p.bottom_friction == 'quadratic':
        speed = jnp.sqrt(state.u**2 + state.v**2)
        bot_u = -p.cd * speed * state.u * p.bottom_mask
        bot_v = -p.cd * speed * state.v * p.bottom_mask
    else:
        bot_u = -p.r_bot * state.u * p.bottom_mask
        bot_v = -p.r_bot * state.v * p.bottom_mask

    dudt = adv_u + cor_u + pgf_x + diff_h_u + diff_v_u + wind_u + bot_u
    dvdt = adv_v + cor_v + pgf_y + diff_h_v + diff_v_v + wind_v + bot_v
    return dudt, dvdt


def _compute_tracer_tendency(state, p):
    """dT/dt, dS/dt for tracer transport."""
    w = _compute_vertical_velocity(state, p)
    adv_T = _advection_scalar(state.T, state.u, state.v, w, p)
    adv_S = _advection_scalar(state.S, state.u, state.v, w, p)

    diff_h_T = p.kappa_h * _laplacian_h(state.T, p)
    diff_h_S = p.kappa_h * _laplacian_h(state.S, p)

    diff_v_T = p.kappa_v * _d2_dz2(state.T, p)
    diff_v_S = p.kappa_v * _d2_dz2(state.S, p)

    heat_factor = 1.0 / (RHO_0 * C_P * p.dz_surface)
    heat_T = p.Q_heat_2d[:, :, None] * heat_factor * p.surface_mask

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


def _free_surface_step(eta, u, v, p, F_rho_x=None, F_rho_y=None):
    """Exact linear shallow water step in spectral space (no Coriolis).

    Solves the coupled (eta, ubt, vbt) system per wavenumber:
        d(eta)/dt  = -H * (ikx*ubt + iky*vbt)     [continuity]
        d(ubt)/dt  = -g * ikx * eta + F_x          [x-momentum]
        d(vbt)/dt  = -g * iky * eta + F_y          [y-momentum]

    via matrix exponential:  expm(A*dt) = I + s*A + c2*A^2
    where s = sin(w*dt)/w,  c2 = (1-cos(w*dt))/w^2,  w = sqrt(g*H*k^2).

    The k=0 mode (omega=0) is identity - mean mass and momentum conserved.
    Barotropic velocity changes are projected uniformly back to 3D fields.

    Optional forcing F = (0, F_x, F_y) includes both wind and density-
    driven barotropic PGF.  The particular solution is the exact
    integral of expm(A*t)*F from 0 to dt_half.
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

    # ── Particular solution for barotropic forcing ──
    # Forced SW system dx/dt = A*x + F with F = (0, F_x, F_y).
    # F includes wind + density-driven barotropic PGF.
    # Exact particular solution: integral of expm(A*t) from 0 to dt_half,
    # multiplied by F. Simplified using omega^2 = g*H_sw*k^2:
    #   eta_part  = -(1-cos)/(g*k^2) * div_F
    #   ubt_part  = dt_half*F_x + (dt_half - sin/w)/k^2 * ikx*div_F
    #   vbt_part  = dt_half*F_y + (dt_half - sin/w)/k^2 * iky*div_F
    # k=0 mode: div_F=0, so only ubt/vbt accelerate by dt_half*F.
    dt_half = p.dt / 2.0

    # Combine wind + density barotropic forcing
    F_x_hat = p.F_bt_x_hat
    F_y_hat = p.F_bt_y_hat
    if F_rho_x is not None:
        F_x_hat = F_x_hat + jnp.fft.fft2(F_rho_x)
        F_y_hat = F_y_hat + jnp.fft.fft2(F_rho_y)

    div_F = ikx * F_x_hat + iky * F_y_hat
    c3_div_k2 = jnp.where(k2_2d > 0.0, (dt_half - sin_div_w) / k2_2d, 0.0)

    eta_hat_new = eta_hat_new - (omc_div_k2 / G_EARTH) * div_F
    ubt_hat_new = ubt_hat_new + dt_half * F_x_hat + c3_div_k2 * ikx * div_F
    vbt_hat_new = vbt_hat_new + dt_half * F_y_hat + c3_div_k2 * iky * div_F

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
    """Linear half-step: diffusion + Coriolis + free surface (all exact).

    The free surface step includes both wind and density-driven
    barotropic PGF as forcing, ensuring the fast external mode is
    driven consistently by all barotropic forces.
    """
    u = _linear_step_diffusion(state.u, p.decay_u)
    v = _linear_step_diffusion(state.v, p.decay_u)
    T = _linear_step_diffusion(state.T, p.decay_T)
    S = _linear_step_diffusion(state.S, p.decay_T)
    u, v = _coriolis_rotation(u, v, p.f0, dt_half)

    # Compute density-driven barotropic PGF from current T/S state
    F_rho_x, F_rho_y = _compute_bt_rho_pgf(state, p)
    eta, u, v = _free_surface_step(state.eta, u, v, p, F_rho_x, F_rho_y)
    return JaxState(u, v, T, S, eta)



def _compute_tracer_residual(state, p):
    """Tracer tendency residual (linear parts subtracted).

    Returns (dTdt, dSdt) with horizontal diffusion removed
    (handled by _linear_half_step).  What remains:
      - advection + vertical diffusion + heat flux
    """
    dTdt, dSdt = _compute_tracer_tendency(state, p)
    dTdt = dTdt - p.kappa_h * _laplacian_h(state.T, p)
    dSdt = dSdt - p.kappa_h * _laplacian_h(state.S, p)
    dTdt = dTdt - p.kappa_bi * _biharmonic_h(state.T, p)
    dSdt = dSdt - p.kappa_bi * _biharmonic_h(state.S, p)
    return dTdt, dSdt


def _compute_momentum_residual(state, p):
    """Momentum tendency residual (linear parts subtracted).

    Returns (dudt, dvdt) with horizontal diffusion, Coriolis,
    barotropic PGF (from both eta and density), and barotropic
    wind removed (handled by _linear_half_step).  What remains:
      - advection + pure baroclinic PGF + vertical diffusion
        + baroclinic wind + bottom friction

    The baroclinic PGF depends on T/S via density anomaly,
    enabling forward-backward coupling.
    """
    dudt, dvdt = _compute_momentum_tendency(state, p)
    dudt = dudt - p.nu_h * _laplacian_h(state.u, p)
    dvdt = dvdt - p.nu_h * _laplacian_h(state.v, p)
    dudt = dudt - p.nu_bi * _biharmonic_h(state.u, p)
    dvdt = dvdt - p.nu_bi * _biharmonic_h(state.v, p)
    dudt = dudt - p.f0 * state.v
    dvdt = dvdt + p.f0 * state.u
    # Barotropic PGF from free surface (eta)
    bt_pgf_x = -G_EARTH * _d_dx_2d(state.eta, p)
    bt_pgf_y = -G_EARTH * _d_dy_2d(state.eta, p)
    dudt = dudt - bt_pgf_x[:, :, None]
    dvdt = dvdt - bt_pgf_y[:, :, None]
    # Barotropic PGF from density anomaly (depth-averaged baroclinic PGF)
    bt_rho_pgf_x, bt_rho_pgf_y = _compute_bt_rho_pgf(state, p)
    dudt = dudt - bt_rho_pgf_x[:, :, None]
    dvdt = dvdt - bt_rho_pgf_y[:, :, None]
    # Barotropic wind forcing
    bt_wind_x = p.tau_x_2d / (RHO_0 * p.H_sw)
    bt_wind_y = p.tau_y_2d / (RHO_0 * p.H_sw)
    dudt = dudt - bt_wind_x[:, :, None]
    dvdt = dvdt - bt_wind_y[:, :, None]
    return dudt, dvdt



def _explicit_full_step(state, p, dt):
    """Forward-backward RK2 for nonlinear tendencies.

    The baroclinic PGF-Tracer coupling produces internal gravity waves
    with purely imaginary eigenvalues.  Symmetric RK2 has |lambda|>1
    for such modes (4th-order growth), causing blowup.

    Forward-backward coupling breaks the oscillation: tracers are
    updated first (using old velocity), then momentum uses the updated
    tracers for the baroclinic PGF.  This shifts eigenvalues from the
    imaginary axis to the left half-plane, giving neutral stability
    (|lambda|=1) for the linear internal wave modes.

    Scheme (2nd-order predictor-corrector):
      Predictor:
        T_pred = T + dt * R_T(state)          [forward: old velocity]
        u_pred = u + dt * R_u(state_T_pred)   [backward: new T for PGF]
      Corrector:
        T_new = T + 0.5*dt * (R_T(state) + R_T(state_pred))
        u_new = u + 0.5*dt * (R_u(state) + R_u(state_T_new))
    """
    # ── Predictor ──
    # Forward: tracer using old velocity
    dT1, dS1 = _compute_tracer_residual(state, p)
    T_pred = state.T + dT1 * dt
    S_pred = state.S + dS1 * dt
    state_T = JaxState(state.u, state.v, T_pred, S_pred, state.eta)

    # Backward: momentum using predicted T for baroclinic PGF
    du1, dv1 = _compute_momentum_residual(state_T, p)
    u_pred = state.u + du1 * dt
    v_pred = state.v + dv1 * dt
    state_pred = JaxState(u_pred, v_pred, T_pred, S_pred, state.eta)

    # ── Corrector ──
    # Forward: tracer using predicted velocity
    dT2, dS2 = _compute_tracer_residual(state_pred, p)
    T_new = state.T + 0.5 * (dT1 + dT2) * dt
    S_new = state.S + 0.5 * (dS1 + dS2) * dt
    state_T_new = JaxState(state.u, state.v, T_new, S_new, state.eta)

    # Backward: momentum using corrected T for baroclinic PGF
    du2, dv2 = _compute_momentum_residual(state_T_new, p)
    u_new = state.u + 0.5 * (du1 + du2) * dt
    v_new = state.v + 0.5 * (dv1 + dv2) * dt
    return JaxState(u_new, v_new, T_new, S_new, state.eta)


def _step_impl(state, p):
    """Strang splitting: L(dt/2) -> N(dt) -> L(dt/2)."""
    dt_half = p.dt / 2.0
    state = _linear_half_step(state, p, dt_half)
    state = _explicit_full_step(state, p, p.dt)
    state = _linear_half_step(state, p, dt_half)
    return state


# ── Public API ───────────────────────────────────────────────────────

def make_solver(grid, physics, dt, forcing=None, eos_type='linear'):
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
        forcing: optional (tau_x_2d, tau_y_2d, Q_heat_2d) tuple of
            (nx, ny) numpy arrays for spatially-varying surface forcing.
            If None, defaults to zero forcing (rest state).
        eos_type: 'linear' (default) or 'unesco' for nonlinear EOS.

    Returns:
        step_fn: JIT-compiled (state: JaxState) -> JaxState
        init_state: (T_init=None, S_init=None) -> JaxState
            If T_init/S_init are provided (e.g., from WOA climatology),
             they override uniform T_ref/S_ref.
        diagnostics: JIT-compiled (state) -> (rho, pressure, w)
    """
    params = _compute_params(grid, physics, dt, forcing=forcing, eos_type=eos_type)

    @jax.jit
    def step(state):
        return _step_impl(state, params)

    @jax.jit
    def diagnostics(state):
        rho_prime = _density_anomaly(state.T, state.S, params)
        rho = RHO_0 + rho_prime
        pressure = _compute_hydrostatic_pressure(state, params)
        w = _compute_vertical_velocity(state, params)
        return rho, pressure, w

    def init_state(T_init=None, S_init=None):
        return _init_state(grid, physics, T_init, S_init)

    return step, init_state, diagnostics


# ── Benchmark ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import DEFAULT_CONFIG
    from grid import make_grid
    from forcing import wind_stress_gyre, heat_flux_meridional

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    dt = 300.0

    # 2D surface forcing (Stommel gyre wind + meridional heat flux)
    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)
    forcing = (tau_x, tau_y, Q_heat)

    print("=== JAX Ocean Solver Benchmark ===")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}")
    print(f"Forcing: wind_stress_gyre(tau0=0.1), heat_flux_meridional(Q0=50)")
    print()

    step_fn, init_state, diag_fn = make_solver(grid, physics, dt, forcing=forcing)

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
