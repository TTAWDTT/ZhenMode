"""
PyTorch Ocean Solver — faithful port of the JAX solver.

Mirrors src/jax_solver.py physics-for-physics and op-for-op so the two
backends can be cross-validated bit-for-bit-equivalently (within float
tolerance). Key mapping:

    jnp.fft.fft(a, axis=0)      -> torch.fft.fft(a, dim=0)
    jnp.fft.fft2(a, axes=(0,1)) -> torch.fft.fft2(a, dim=(0,1))
    arr.at[idx].set(v)          -> _scatter_set(arr, idx, v)
    jnp.cumsum(x, axis=-1)      -> torch.cumsum(x, dim=-1)
    @jax.jit                    -> eager torch (optionally torch.compile)

All arrays are torch.float64 (matching JAX x64). No autograd is used in
the step itself (the linear solve and RK2 are explicit), which keeps the
step cheap and lets torch.compile fuse the elementwise graph.

Usage:
    from torch_solver import make_solver
    step_fn, init_state, diag_fn = make_solver(grid, physics, dt=300.0)
    state = init_state()
    for _ in range(n_steps):
        state = step_fn(state)
    rho, p, w = diag_fn(state)
"""
import torch
import numpy as np
from collections import namedtuple

from config import RHO_0, ALPHA_T, BETA_S, C_P, G_EARTH


# ── State ────────────────────────────────────────────────────────────
TorchState = namedtuple('TorchState', ['u', 'v', 'T', 'S', 'eta'])


# ── Solver parameters (mirrors JAX SolverParams) ─────────────────────
SolverParams = namedtuple('SolverParams', [
    'kx', 'ky', 'k2',              # (nx,1,1), (1,ny,1), (nx,ny,1)
    'k4',                          # (nx,ny,1) k4 = k2^2 for biharmonic
    'dealias_2d',                  # (nx,ny,1) combined 2/3 dealias mask
    'decay_u', 'decay_T',          # (nx,ny,1) diffusion decay for dt/2
    # Grid
    'f',                           # (nx,ny) full Coriolis
    'f0',                          # scalar
    'nx', 'ny', 'nz',              # ints
    # Vertical grid
    'dz_denom_interior',           # (1,1,nz-2)
    'dz_bnd_top', 'dz_bnd_bot',
    'd2z_hm', 'd2z_hp', 'd2z_denom',
    'd2z_h0_top', 'd2z_h0_bot',
    'dz_3d',                       # (1,1,nz-1)
    'dz_surface',
    # Forcing masks
    'surface_mask',                # (1,1,nz)
    'bottom_mask',                 # (1,1,nz)
    # Physics scalars
    'nu_h', 'nu_v', 'kappa_h', 'kappa_v',
    'nu_bi', 'kappa_bi',
    'T_ref', 'S_ref',
    # Surface forcing
    'tau_x_2d', 'tau_y_2d', 'Q_heat_2d', 'r_bot', 'cd', 'bottom_friction',
    # Free surface (2D spectral)
    'kx_2d', 'ky_2d',
    'omega_sw',
    'sw_sin', 'sw_cos',
    'H_mean', 'H_sw',
    'dz_norm',
    'F_bt_x_hat', 'F_bt_y_hat',
    'sw_sin_div_w',
    # Time
    'dt',
    # EOS
    'eos_type',
    # Smagorinsky
    'smag_cs',
    'dx',
])


# ── Small helpers ────────────────────────────────────────────────────
def _t(x):
    """Convert array-like to a torch float64 tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(torch.float64)
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


def _scatter_set(arr, index, value):
    """Equivalent of jnp arr.at[index].set(value): returns a copy."""
    out = arr.clone()
    out[index] = value
    return out


def _compute_params(grid, physics, dt, forcing=None, eos_type='linear'):
    """Pre-compute all static torch arrays from grid and physics."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy = grid.dx, grid.dy
    dt_half = dt / 2.0

    # ── Wavenumbers ──
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx = _t(kx_1d).view(nx, 1, 1)
    ky = _t(ky_1d).view(1, ny, 1)
    k2 = kx ** 2 + ky ** 2            # (nx, ny, 1)
    k4 = k2 ** 2

    # ── Dealias mask (2/3 rule) ──
    cx, cy = nx // 3, ny // 3
    mx = torch.zeros(nx, dtype=torch.float64)
    mx[:cx] = 1.0
    mx[-cx:] = 1.0
    my = torch.zeros(ny, dtype=torch.float64)
    my[:cy] = 1.0
    my[-cy:] = 1.0
    dealias_2d = (mx[:, None] * my[None, :])[:, :, None]

    # ── Diffusion decay factors (linear half-step, dt/2) ──
    decay_u = torch.exp(-(physics.nu_h * k2 + physics.nu_bi * k4) * dt_half)
    decay_T = torch.exp(-(physics.kappa_h * k2 + physics.kappa_bi * k4) * dt_half)

    # ── Coriolis ──
    f = _t(grid.f)
    f0 = float(grid.f0)

    # ── Vertical grid coefficients ──
    z = _t(grid.z)
    dz = _t(grid.dz)
    dz_3d = dz.view(1, 1, -1)

    dz_up = z[1:-1] - z[:-2]
    dz_dn = z[2:] - z[1:-1]
    dz_denom_interior = (dz_up + dz_dn).view(1, 1, -1)
    dz_bnd_top = float(z[1] - z[0])
    dz_bnd_bot = float(z[-1] - z[-2])

    hm = (z[:-2] - z[1:-1]).abs()
    hp = (z[2:] - z[1:-1]).abs()
    d2z_denom = (hm * hp * (hm + hp) / 2.0).view(1, 1, -1)
    d2z_hm = hm.view(1, 1, -1)
    d2z_hp = hp.view(1, 1, -1)
    d2z_h0_top = float((z[1] - z[0]).abs())
    d2z_h0_bot = float((z[-1] - z[-2]).abs())

    dz_surface = float((z[0] - z[1]).abs())

    # ── Forcing masks ──
    surface_mask = torch.zeros(nz, dtype=torch.float64).view(1, 1, -1)
    surface_mask[..., 0] = 1.0
    bottom_mask = torch.zeros(nz, dtype=torch.float64).view(1, 1, -1)
    bottom_mask[..., -1] = 1.0

    # ── Free surface parameters ──
    H_mean = float(grid.depth[grid.ocean_mask].mean())
    H_sw = float(torch.sum(dz))

    kx_2d = _t(kx_1d).view(nx, 1)
    ky_2d = _t(ky_1d).view(1, ny)
    k2_2d = kx_2d ** 2 + ky_2d ** 2

    omega_sw = torch.sqrt(G_EARTH * H_sw * k2_2d)
    sw_sin = torch.sin(omega_sw * dt_half)
    sw_cos = torch.cos(omega_sw * dt_half)

    dz_norm = dz.view(1, 1, -1) / H_sw

    # ── Surface forcing ──
    if forcing is None:
        tau_x_2d = torch.zeros(nx, ny, dtype=torch.float64)
        tau_y_2d = torch.zeros(nx, ny, dtype=torch.float64)
        Q_heat_2d = torch.zeros(nx, ny, dtype=torch.float64)
    else:
        tau_x_2d, tau_y_2d, Q_heat_2d = (_t(f) for f in forcing)

    F_bt_x = tau_x_2d / (RHO_0 * H_sw)
    F_bt_y = tau_y_2d / (RHO_0 * H_sw)
    F_bt_x_hat = torch.fft.fft2(F_bt_x)
    F_bt_y_hat = torch.fft.fft2(F_bt_y)

    sw_sin_div_w = torch.where(omega_sw > 0.0, sw_sin / omega_sw,
                               torch.zeros_like(omega_sw))

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
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if T_init is not None:
        T = _t(T_init)
        S = _t(S_init)
    else:
        T = torch.full((nx, ny, nz), physics.T_ref, dtype=torch.float64)
        S = torch.full((nx, ny, nz), physics.S_ref, dtype=torch.float64)
    return TorchState(
        u=torch.zeros((nx, ny, nz), dtype=torch.float64),
        v=torch.zeros((nx, ny, nz), dtype=torch.float64),
        T=T,
        S=S,
        eta=torch.zeros((nx, ny), dtype=torch.float64),
    )


# ── Spectral operators ───────────────────────────────────────────────
def _d_dx(u, p):
    u_hat = torch.fft.fft(u, dim=0)
    return torch.fft.ifft(1j * p.kx * u_hat, dim=0).real


def _d_dy(u, p):
    u_hat = torch.fft.fft(u, dim=1)
    return torch.fft.ifft(1j * p.ky * u_hat, dim=1).real


def _laplacian_h(u, p):
    u_hat = torch.fft.fft2(u, dim=(0, 1))
    return torch.fft.ifft2(-p.k2 * u_hat, dim=(0, 1)).real


def _biharmonic_h(u, p):
    u_hat = torch.fft.fft2(u, dim=(0, 1))
    return torch.fft.ifft2(p.k4 * u_hat, dim=(0, 1)).real


def _divergence_h(u, v, p):
    return _d_dx(u, p) + _d_dy(v, p)


def _d_dx_2d(eta, p):
    eta_hat = torch.fft.fft(eta, dim=0)
    return torch.fft.ifft(1j * p.kx_2d * eta_hat, dim=0).real


def _d_dy_2d(eta, p):
    eta_hat = torch.fft.fft(eta, dim=1)
    return torch.fft.ifft(1j * p.ky_2d * eta_hat, dim=1).real


def _d_dz(u, p):
    du_interior = (u[..., 2:] - u[..., :-2]) / p.dz_denom_interior
    du_top = (u[..., 1:2] - u[..., 0:1]) / p.dz_bnd_top
    du_bot = (u[..., -1:] - u[..., -2:-1]) / p.dz_bnd_bot
    return torch.cat([du_top, du_interior, du_bot], dim=-1)


def _d2_dz2(u, p):
    d2u_interior = (
        u[..., 2:] * p.d2z_hm + u[..., :-2] * p.d2z_hp
        - u[..., 1:-1] * (p.d2z_hm + p.d2z_hp)
    ) / p.d2z_denom
    d2u_top = (u[..., 2:3] - 2 * u[..., 1:2] + u[..., 0:1]) / (p.d2z_h0_top ** 2)
    d2u_bot = (u[..., -3:-2] - 2 * u[..., -2:-1] + u[..., -1:]) / (p.d2z_h0_bot ** 2)
    return torch.cat([d2u_top, d2u_interior, d2u_bot], dim=-1)


def _dealias_h(field, p):
    field_hat = torch.fft.fft2(field, dim=(0, 1))
    field_hat = field_hat * p.dealias_2d
    return torch.fft.ifft2(field_hat, dim=(0, 1)).real


def _advection_flux_form(u, v, w, p):
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
    dT_dx = _d_dx(T, p)
    dT_dy = _d_dy(T, p)
    dT_dz = _d_dz(T, p)
    adv_T = -(u * dT_dx + v * dT_dy + w * dT_dz)
    return _dealias_h(adv_T, p)


def _linear_step_diffusion(u, decay):
    u_hat = torch.fft.fft2(u, dim=(0, 1))
    return torch.fft.ifft2(decay * u_hat, dim=(0, 1)).real


# ── Physics ──────────────────────────────────────────────────────────
def _rho_smow_t(T):
    return (
        999.842594
        + 6.793952e-2 * T
        - 9.095290e-3 * T ** 2
        + 1.001685e-4 * T ** 3
        - 1.120083e-6 * T ** 4
        + 6.536332e-9 * T ** 5
    )


def _b_t(T):
    return (
        8.24493e-1
        - 4.0899e-3 * T
        + 7.6438e-5 * T ** 2
        - 8.2467e-7 * T ** 3
        + 5.3875e-9 * T ** 4
    )


def _c_t(T):
    return -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T ** 2


def _density_unesco_t(T, S):
    _D = 4.8314e-4
    smow = _rho_smow_t(T)
    S_safe = torch.clamp(S, min=0.0)
    S_sqrt = torch.sqrt(S_safe)
    return smow + _b_t(T) * S + _c_t(T) * S_sqrt * S + _D * S ** 2


def _density_anomaly(T, S, p):
    if p.eos_type == 'unesco':
        return _density_unesco_t(T, S) - RHO_0
    else:
        return RHO_0 * (-ALPHA_T * (T - p.T_ref) + BETA_S * (S - p.S_ref))


def _compute_hydrostatic_pressure(state, p):
    rho_prime = _density_anomaly(state.T, state.S, p)
    rho_avg = 0.5 * (rho_prime[..., :-1] + rho_prime[..., 1:])
    dp = G_EARTH * rho_avg * p.dz_3d
    p_bc = torch.zeros_like(state.T)
    p_bc[..., 1:] = torch.cumsum(dp, dim=-1)
    p_bt = RHO_0 * G_EARTH * state.eta[:, :, None]
    return p_bt + p_bc


def _compute_pressure_gradient(state, p):
    pressure = _compute_hydrostatic_pressure(state, p)
    pgf_x = -_d_dx(pressure, p) / RHO_0
    pgf_y = -_d_dy(pressure, p) / RHO_0
    return pgf_x, pgf_y


def _compute_bt_rho_pgf(state, p):
    rho_prime = _density_anomaly(state.T, state.S, p)
    rho_avg = 0.5 * (rho_prime[..., :-1] + rho_prime[..., 1:])
    dp = G_EARTH * rho_avg * p.dz_3d
    p_bc = torch.zeros_like(state.T)
    p_bc[..., 1:] = torch.cumsum(dp, dim=-1)
    p_bc_avg = torch.sum(
        0.5 * (p_bc[..., :-1] + p_bc[..., 1:]) * p.dz_norm, dim=-1)
    bt_rho_pgf_x = -_d_dx_2d(p_bc_avg, p) / RHO_0
    bt_rho_pgf_y = -_d_dy_2d(p_bc_avg, p) / RHO_0
    return bt_rho_pgf_x, bt_rho_pgf_y


def _compute_momentum_tendency(state, p, lap_u=None, lap_v=None):
    w = _compute_vertical_velocity(state, p)
    adv_u, adv_v = _advection_flux_form(state.u, state.v, w, p)
    f_3d = p.f[:, :, None]
    cor_u = f_3d * state.v
    cor_v = -f_3d * state.u
    pgf_x, pgf_y = _compute_pressure_gradient(state, p)

    if lap_u is None:
        lap_u = _laplacian_h(state.u, p)
    if lap_v is None:
        lap_v = _laplacian_h(state.v, p)
    diff_h_u = p.nu_h * lap_u
    diff_h_v = p.nu_h * lap_v
    if p.smag_cs > 0:
        dudx = _d_dx(state.u, p)
        dvdy = _d_dy(state.v, p)
        dudy = _d_dy(state.u, p)
        dvdx = _d_dx(state.v, p)
        def_strain = torch.sqrt((dudx - dvdy) ** 2 + (dudy + dvdx) ** 2)
        nu_smg = (p.smag_cs * p.dx) ** 2 * def_strain
        diff_h_u = diff_h_u + nu_smg * _laplacian_h(state.u, p)
        diff_h_v = diff_h_v + nu_smg * _laplacian_h(state.v, p)

    diff_v_u = p.nu_v * _d2_dz2(state.u, p)
    diff_v_v = p.nu_v * _d2_dz2(state.v, p)

    wind_factor = 1.0 / (RHO_0 * p.dz_surface)
    wind_u = p.tau_x_2d[:, :, None] * wind_factor * p.surface_mask
    wind_v = p.tau_y_2d[:, :, None] * wind_factor * p.surface_mask

    if p.bottom_friction == 'quadratic':
        speed = torch.sqrt(state.u ** 2 + state.v ** 2)
        bot_u = -p.cd * speed * state.u * p.bottom_mask
        bot_v = -p.cd * speed * state.v * p.bottom_mask
    else:
        bot_u = -p.r_bot * state.u * p.bottom_mask
        bot_v = -p.r_bot * state.v * p.bottom_mask

    dudt = adv_u + cor_u + pgf_x + diff_h_u + diff_v_u + wind_u + bot_u
    dvdt = adv_v + cor_v + pgf_y + diff_h_v + diff_v_v + wind_v + bot_v
    return dudt, dvdt


def _compute_tracer_tendency(state, p, lap_T=None, lap_S=None):
    w = _compute_vertical_velocity(state, p)
    adv_T = _advection_scalar(state.T, state.u, state.v, w, p)
    adv_S = _advection_scalar(state.S, state.u, state.v, w, p)
    # Optional CSE: pass precomputed laplacian to avoid double FFTs.
    if lap_T is None:
        lap_T = _laplacian_h(state.T, p)
    if lap_S is None:
        lap_S = _laplacian_h(state.S, p)
    diff_h_T = p.kappa_h * lap_T
    diff_h_S = p.kappa_h * lap_S
    diff_v_T = p.kappa_v * _d2_dz2(state.T, p)
    diff_v_S = p.kappa_v * _d2_dz2(state.S, p)
    heat_factor = 1.0 / (RHO_0 * C_P * p.dz_surface)
    heat_T = p.Q_heat_2d[:, :, None] * heat_factor * p.surface_mask
    dTdt = adv_T + diff_h_T + diff_v_T + heat_T
    dSdt = adv_S + diff_h_S + diff_v_S
    return dTdt, dSdt


def _compute_vertical_velocity(state, p):
    div_h = _divergence_h(state.u, state.v, p)
    div_avg = 0.5 * (div_h[..., :-1] + div_h[..., 1:])
    integrand = div_avg * p.dz_3d
    w = torch.zeros_like(state.u)
    cum_bottom_up = torch.cumsum(torch.flip(integrand, dims=[-1]), dim=-1)
    w[..., :-1] = -torch.flip(cum_bottom_up, dims=[-1])
    return w


# ── Integrator (Strang splitting IMEX) ───────────────────────────────
def _coriolis_rotation(u, v, f0, dt):
    angle = torch.as_tensor(f0, dtype=torch.float64) * dt
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    u_new = cos_a * u + sin_a * v
    v_new = -sin_a * u + cos_a * v
    return u_new, v_new


def _barotropic_velocity(u, v, p):
    u_avg = 0.5 * (u[..., :-1] + u[..., 1:])
    v_avg = 0.5 * (v[..., :-1] + v[..., 1:])
    ubt = torch.sum(u_avg * p.dz_norm, dim=-1)
    vbt = torch.sum(v_avg * p.dz_norm, dim=-1)
    return ubt, vbt


def _free_surface_step(eta, u, v, p, F_rho_x=None, F_rho_y=None):
    ubt, vbt = _barotropic_velocity(u, v, p)

    eta_hat = torch.fft.fft2(eta)
    ubt_hat = torch.fft.fft2(ubt)
    vbt_hat = torch.fft.fft2(vbt)

    k2_2d = p.kx_2d ** 2 + p.ky_2d ** 2
    sin_div_w = torch.where(p.omega_sw > 0.0, p.sw_sin / p.omega_sw,
                            torch.zeros_like(p.omega_sw))
    omc_div_k2 = torch.where(k2_2d > 0.0, (1.0 - p.sw_cos) / k2_2d,
                             torch.zeros_like(k2_2d))

    ikx = 1j * p.kx_2d
    iky = 1j * p.ky_2d

    eta_hat_new = (p.sw_cos * eta_hat
                   - p.H_sw * sin_div_w * (ikx * ubt_hat + iky * vbt_hat))

    ubt_hat_new = (ubt_hat
                   - G_EARTH * ikx * sin_div_w * eta_hat
                   - omc_div_k2 * (p.kx_2d ** 2 * ubt_hat
                                   + p.kx_2d * p.ky_2d * vbt_hat))

    vbt_hat_new = (vbt_hat
                   - G_EARTH * iky * sin_div_w * eta_hat
                   - omc_div_k2 * (p.kx_2d * p.ky_2d * ubt_hat
                                   + p.ky_2d ** 2 * vbt_hat))

    dt_half = p.dt / 2.0

    F_x_hat = p.F_bt_x_hat
    F_y_hat = p.F_bt_y_hat
    if F_rho_x is not None:
        F_x_hat = F_x_hat + torch.fft.fft2(F_rho_x)
        F_y_hat = F_y_hat + torch.fft.fft2(F_rho_y)

    div_F = ikx * F_x_hat + iky * F_y_hat
    c3_div_k2 = torch.where(k2_2d > 0.0, (dt_half - sin_div_w) / k2_2d,
                            torch.zeros_like(k2_2d))

    eta_hat_new = eta_hat_new - (omc_div_k2 / G_EARTH) * div_F
    ubt_hat_new = ubt_hat_new + dt_half * F_x_hat + c3_div_k2 * ikx * div_F
    vbt_hat_new = vbt_hat_new + dt_half * F_y_hat + c3_div_k2 * iky * div_F

    eta_new = torch.fft.ifft2(eta_hat_new).real
    ubt_new = torch.fft.ifft2(ubt_hat_new).real
    vbt_new = torch.fft.ifft2(vbt_hat_new).real

    delta_ubt = (ubt_new - ubt)[:, :, None]
    delta_vbt = (vbt_new - vbt)[:, :, None]
    u_new = u + delta_ubt
    v_new = v + delta_vbt

    return eta_new, u_new, v_new


def _linear_half_step(state, p, dt_half):
    u = _linear_step_diffusion(state.u, p.decay_u)
    v = _linear_step_diffusion(state.v, p.decay_u)
    T = _linear_step_diffusion(state.T, p.decay_T)
    S = _linear_step_diffusion(state.S, p.decay_T)
    u, v = _coriolis_rotation(u, v, p.f0, dt_half)
    F_rho_x, F_rho_y = _compute_bt_rho_pgf(state, p)
    eta, u, v = _free_surface_step(state.eta, u, v, p, F_rho_x, F_rho_y)
    return TorchState(u, v, T, S, eta)


def _compute_tracer_residual(state, p):
    # CSE: compute laplacian once, reuse for the tendency's +diff and the
    # residual's subtraction (net horizontal diffusion cancels). Biharmonic
    # stays in the residual (it is not part of the tendency).
    lap_T = _laplacian_h(state.T, p)
    lap_S = _laplacian_h(state.S, p)
    dTdt, dSdt = _compute_tracer_tendency(state, p, lap_T=lap_T, lap_S=lap_S)
    dTdt = dTdt - p.kappa_h * lap_T
    dSdt = dSdt - p.kappa_h * lap_S
    dTdt = dTdt - p.kappa_bi * _biharmonic_h(state.T, p)
    dSdt = dSdt - p.kappa_bi * _biharmonic_h(state.S, p)
    return dTdt, dSdt


def _compute_momentum_residual(state, p):
    # CSE: compute laplacian once, reuse for the tendency's +diff and the
    # residual's subtraction (net horizontal diffusion cancels). Biharmonic
    # stays in the residual (it is not part of the tendency).
    lap_u = _laplacian_h(state.u, p)
    lap_v = _laplacian_h(state.v, p)
    dudt, dvdt = _compute_momentum_tendency(state, p, lap_u=lap_u, lap_v=lap_v)
    dudt = dudt - p.nu_h * lap_u
    dvdt = dvdt - p.nu_h * lap_v
    dudt = dudt - p.nu_bi * _biharmonic_h(state.u, p)
    dvdt = dvdt - p.nu_bi * _biharmonic_h(state.v, p)
    dudt = dudt - p.f0 * state.v
    dvdt = dvdt + p.f0 * state.u
    bt_pgf_x = -G_EARTH * _d_dx_2d(state.eta, p)
    bt_pgf_y = -G_EARTH * _d_dy_2d(state.eta, p)
    dudt = dudt - bt_pgf_x[:, :, None]
    dvdt = dvdt - bt_pgf_y[:, :, None]
    bt_rho_pgf_x, bt_rho_pgf_y = _compute_bt_rho_pgf(state, p)
    dudt = dudt - bt_rho_pgf_x[:, :, None]
    dvdt = dvdt - bt_rho_pgf_y[:, :, None]
    bt_wind_x = p.tau_x_2d / (RHO_0 * p.H_sw)
    bt_wind_y = p.tau_y_2d / (RHO_0 * p.H_sw)
    dudt = dudt - bt_wind_x[:, :, None]
    dvdt = dvdt - bt_wind_y[:, :, None]
    return dudt, dvdt


def _explicit_full_step(state, p, dt):
    # Predictor
    dT1, dS1 = _compute_tracer_residual(state, p)
    T_pred = state.T + dT1 * dt
    S_pred = state.S + dS1 * dt
    state_T = TorchState(state.u, state.v, T_pred, S_pred, state.eta)

    du1, dv1 = _compute_momentum_residual(state_T, p)
    u_pred = state.u + du1 * dt
    v_pred = state.v + dv1 * dt
    state_pred = TorchState(u_pred, v_pred, T_pred, S_pred, state.eta)

    # Corrector
    dT2, dS2 = _compute_tracer_residual(state_pred, p)
    T_new = state.T + 0.5 * (dT1 + dT2) * dt
    S_new = state.S + 0.5 * (dS1 + dS2) * dt
    state_T_new = TorchState(state.u, state.v, T_new, S_new, state.eta)

    du2, dv2 = _compute_momentum_residual(state_T_new, p)
    u_new = state.u + 0.5 * (du1 + du2) * dt
    v_new = state.v + 0.5 * (dv1 + dv2) * dt
    return TorchState(u_new, v_new, T_new, S_new, state.eta)


def _step_impl(state, p):
    dt_half = p.dt / 2.0
    state = _linear_half_step(state, p, dt_half)
    state = _explicit_full_step(state, p, p.dt)
    state = _linear_half_step(state, p, dt_half)
    return state


# ── Public API ───────────────────────────────────────────────────────
def make_solver(grid, physics, dt, forcing=None, eos_type='linear',
                compile=False):
    """
    Create a PyTorch ocean solver.

    Mirrors jax_solver.make_solver. If compile=True, torch.compile is
    applied to the full step (falling back to eager if unavailable).

    Returns: step_fn, init_state, diagnostics
    """
    params = _compute_params(grid, physics, dt, forcing=forcing,
                             eos_type=eos_type)

    def step(state):
        return _step_impl(state, params)

    if compile:
        try:
            step = torch.compile(step)
        except Exception as e:  # noqa: BLE001
            print(f"torch.compile unavailable, using eager: {e}")

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

    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)
    forcing = (tau_x, tau_y, Q_heat)

    print("=== PyTorch Ocean Solver Benchmark ===")
    print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
    print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}")
    print(f"Forcing: wind_stress_gyre(tau0=0.1), heat_flux_meridional(Q0=50)")
    print()

    step_fn, init_state, diag_fn = make_solver(grid, physics, dt,
                                               forcing=forcing, compile=False)
    state = init_state()
    g = torch.Generator().manual_seed(42)
    state = TorchState(
        u=state.u + torch.randn(state.u.shape, generator=g) * 0.01,
        v=state.v + torch.randn(state.v.shape, generator=g) * 0.01,
        T=state.T + torch.randn(state.T.shape, generator=g) * 0.01,
        S=state.S,
        eta=state.eta,
    )

    print("Warmup...")
    t0 = time.perf_counter()
    state = step_fn(state)
    t1 = time.perf_counter()
    print(f"First step: {t1-t0:.2f}s")
    print()

    n = 10
    t0 = time.perf_counter()
    for _ in range(n):
        state = step_fn(state)
    t1 = time.perf_counter()
    ms_per_step = (t1 - t0) / n * 1000
    print(f"Steps: {n}")
    print(f"Time/step: {ms_per_step:.1f} ms")
    print(f"Simulated time: {n * dt / 86400:.2f} days in {t1-t0:.2f}s wall time")
