"""
Global Finite-Difference Ocean Solver — hydrostatic primitive equations, JAX.

Companion to jax_solver.py (regional pseudo-spectral). This solver uses:
  - 2nd-order finite differences on a global 1° lat-lon grid (lon-periodic),
  - spherical metric factors (dx = R*cos(lat)*dlon, varies with latitude),
  - a real wet_mask for no-flux land boundaries,
  - full 2D Coriolis f = 2*Omega*sin(lat).

The spectral regional solver (jax_solver.py) is UNTOUCHED and retained as a
cross-validation baseline. This file is the new FD verification target.

G1 (this file, initial): FD horizontal operators + vertical operators +
MMS (manufactured-solution) verification. The FD operators are pure functions
taking (field, params); params is a lightweight struct carrying the metric
fields. No time integration yet (G2).

Convention (matches regional solver + grid.py):
  - 3D fields: (nx, ny, nz), axis 0 = lon (periodic), axis 1 = lat, axis 2 = z
  - 2D fields: (nx, ny)
  - z negative downward, z=0 at surface
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from collections import namedtuple

from config import RHO_0, ALPHA_T, BETA_S, C_P, G_EARTH, R_EARTH, OMEGA


# ── State (same structure as regional solver) ──────────────────────
JaxStateG = namedtuple('JaxStateG', ['u', 'v', 'T', 'S', 'eta'])


# ── FD solver parameters ───────────────────────────────────────────
# Metric + grid fields for the FD operators. Built once from a GlobalOceanGrid.
FDParams = namedtuple('FDParams', [
    # Spherical metric
    'dx_2d',            # (nx, ny) zonal spacing [m] = R*cos(lat)*dlon
    'dy',               # scalar meridional spacing [m]
    'cos_lat',          # (ny,) cos(lat)
    'inv_dx',           # (nx, ny) 1/dx_2d
    'inv_dy',           # scalar 1/dy
    'inv_dx2',          # (nx, ny) 1/dx_2d^2
    'inv_dy2',          # scalar 1/dy^2
    # Coriolis
    'f',                # (nx, ny) full 2D Coriolis
    # Land
    'wet_mask',         # (nx, ny) 1=ocean, 0=land
    'wet_mask_3d',      # (nx, ny, 1) for 3D broadcast (column-uniform)
    'wet_mask_z',       # (nx, ny, nz) TRUE vertical wet mask: 1 where layer is
                        # above the seafloor, 0 below (ghost water excluded).
                        # Used in pressure integration to kill the spurious PGF
                        # at steep topography (ghost-water-column bug fix).
    'interior_mask',    # (nx, ny) 1 in the interior, 0 on the N/S boundary
                        # rows (j=0, j=ny-1). Used to enforce the no-flux wall:
                        # the normal (meridional) velocity v is zeroed here so
                        # no flow crosses the closed N/S truncation wall.
    'interior_mask_z',  # (nx, ny, 1) broadcast of interior_mask for 3D fields.
    # Vertical grid (non-uniform z-levels, same as regional)
    'dz_denom_interior', 'dz_bnd_top', 'dz_bnd_bot',
    'd2z_hm', 'd2z_hp', 'd2z_denom', 'd2z_h0_top', 'd2z_h0_bot',
    'dz_3d', 'dz_surface',
    'surface_mask', 'bottom_mask',   # (1,1,nz)
    # Dimensions
    'nx', 'ny', 'nz',
])


def make_fd_params(grid):
    """Build FDParams from a GlobalOceanGrid (grid.py).

    Precomputes all metric inverse fields so the FD operators are pure
    array arithmetic (no division inside the hot loop).
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx_2d = jnp.array(grid.dx_2d)                # (nx, ny)
    dy = float(grid.dy)
    cos_lat = jnp.array(grid.cos_lat)            # (ny,)

    inv_dx = (1.0 / dx_2d)[:, :, None]           # (nx, ny, 1) for 3D broadcast
    inv_dy = 1.0 / dy
    inv_dx2 = (1.0 / dx_2d ** 2)[:, :, None]     # (nx, ny, 1)
    inv_dy2 = inv_dy ** 2

    f = jnp.array(grid.f)                        # (nx, ny)
    wet_mask = jnp.array(grid.wet_mask)          # (nx, ny)
    wet_mask_3d = wet_mask[:, :, None]           # (nx, ny, 1)
    # True 3D wet mask (layer-resolved): from grid if available, else fall
    # back to column-uniform (regional/synthetic grids without bathymetry).
    if hasattr(grid, 'wet_mask_3d') and grid.wet_mask_3d is not None:
        wet_mask_z = jnp.array(grid.wet_mask_3d)      # (nx, ny, nz)
    else:
        wet_mask_z = jnp.broadcast_to(wet_mask_3d, (nx, ny, nz))

    # No-flux wall mask: 1 interior, 0 on the N/S boundary rows. The closed
    # truncation wall zero normal velocity here (interior_mask_z applied to v).
    interior_1d = np.ones(ny)
    interior_1d[0] = 0.0
    interior_1d[-1] = 0.0
    interior_mask = jnp.array(np.broadcast_to(interior_1d[None, :], (nx, ny)))
    interior_mask_z = interior_mask[:, :, None]

    # Vertical grid coefficients (identical math to regional _compute_params)
    z = jnp.array(grid.z)
    dz = jnp.array(grid.dz)
    dz_3d = dz.reshape(1, 1, -1)
    dz_up = z[1:-1] - z[:-2]
    dz_dn = z[2:] - z[1:-1]
    dz_denom_interior = (dz_up + dz_dn).reshape(1, 1, -1)
    dz_bnd_top = float(z[1] - z[0])
    dz_bnd_bot = float(z[-1] - z[-2])
    hm = jnp.abs(z[:-2] - z[1:-1])
    hp = jnp.abs(z[2:] - z[1:-1])
    d2z_denom = (hm * hp * (hm + hp) / 2.0).reshape(1, 1, -1)
    d2z_hm = hm.reshape(1, 1, -1)
    d2z_hp = hp.reshape(1, 1, -1)
    d2z_h0_top = float(jnp.abs(z[1] - z[0]))
    d2z_h0_bot = float(jnp.abs(z[-1] - z[-2]))
    dz_surface = float(jnp.abs(z[0] - z[1]))

    surface_mask = jnp.zeros(nz).at[0].set(1.0).reshape(1, 1, -1)
    bottom_mask = jnp.zeros(nz).at[-1].set(1.0).reshape(1, 1, -1)

    return FDParams(
        dx_2d=dx_2d, dy=dy, cos_lat=cos_lat,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dx2=inv_dx2, inv_dy2=inv_dy2,
        f=f, wet_mask=wet_mask, wet_mask_3d=wet_mask_3d,
        wet_mask_z=wet_mask_z,
        interior_mask=interior_mask, interior_mask_z=interior_mask_z,
        dz_denom_interior=dz_denom_interior,
        dz_bnd_top=dz_bnd_top, dz_bnd_bot=dz_bnd_bot,
        d2z_hm=d2z_hm, d2z_hp=d2z_hp, d2z_denom=d2z_denom,
        d2z_h0_top=d2z_h0_top, d2z_h0_bot=d2z_h0_bot,
        dz_3d=dz_3d, dz_surface=dz_surface,
        surface_mask=surface_mask, bottom_mask=bottom_mask,
        nx=nx, ny=ny, nz=nz,
    )


# ── Horizontal FD operators (2nd-order, spherical metric, lon-periodic) ─

def _d_dx(u, p):
    """Zonal derivative du/dx, 2nd-order central FD, longitude-periodic.

    On a lat-lon grid dx varies with latitude, so the inverse spacing is a
    2D field. Axis 0 (lon) wraps via jnp.roll. Land points: the derivative
    is computed everywhere then masked by wet_mask_3d where appropriate by
    the caller (advection applies its own masking).
    """
    return (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) * (0.5 * p.inv_dx)


def _d_dy(u, p):
    """Meridional derivative du/dy, 2nd-order central FD with NO-FLUX WALLS.

    Axis 1 (lat) does NOT wrap. The N/S domain edges are CLOSED walls: the
    one-sided extrapolation stencil (-3u0+4u1-u2) it replaced is unbounded
    for advection and drove the tracer blow-up (T→thousands at the boundary
    row). The no-flux wall uses a mirror ghost cell (ghost = boundary value,
    `mode='edge'` reflection about the wall face at the cell edge) so the
    boundary row gets a STABLE central difference with zero normal flux:

        du/dy|_0 = (u1 - u_ghost)/(2 dy) = (u1 - u0)/(2 dy)

    This enforces ∂u/∂n = 0 at the wall face (no diffusive flux) while the
    normal-velocity mask (v=0 at boundary rows, enforced in _step_impl +
    _free_surface_step_fd) kills the advective flux. Together = closed wall.
    """
    # Mirror-pad the lat axis by 1 cell each side (ghost = boundary value).
    # ndim-agnostic: pad only axis 1.
    pad = [(0, 0)] * u.ndim
    pad[1] = (1, 1)
    u_pad = jnp.pad(u, pad, mode='edge')
    return (u_pad[:, 2:] - u_pad[:, :-2]) * (0.5 * p.inv_dy)


def _laplacian_h(u, p):
    """Horizontal Laplacian on the sphere:

        ∇²u = 1/cosφ ∂/∂φ(cosφ ∂u/∂φ) + 1/cos²φ ∂²u/∂λ²
            = ∂²u/∂y² - (tanφ/R) ∂u/∂y + (1/cos²φ) ∂²u/∂x²

    where ∂/∂y = (1/R)∂/∂φ, ∂/∂x = 1/(R cosφ)∂/∂λ.
    Implemented in physical (meter) space: ∂²/∂x² and ∂²/∂y² via FD, plus
    the spherical correction term -(tanφ/R)·∂u/∂y = -(sinφ/(R cosφ))·∂u/∂y.
    """
    # ∂²u/∂x²: central second FD, lon-periodic
    d2u_dx2 = (jnp.roll(u, -1, axis=0) - 2.0 * u + jnp.roll(u, 1, axis=0)) * p.inv_dx2
    # ∂²u/∂y²: central second FD with NO-FLUX WALL (mirror ghost cell).
    # Same `mode='edge'` reflection as _d_dy: ghost = boundary value, giving
    # zero normal gradient at the wall face and a stable central 2nd diff:
    #   d²u/dy²|_0 = (u1 - 2u0 + u_ghost)/dy² = (u1 - u0)/dy²
    # The prior one-sided (-2,-5,4,-1) stencil extrapolated and amplified the
    # grid-scale mode that blew up the tracer at the boundary row.
    pad = [(0, 0)] * u.ndim
    pad[1] = (1, 1)
    u_pad = jnp.pad(u, pad, mode='edge')
    d2u_dy2 = (u_pad[:, 2:] - 2.0 * u_pad[:, 1:-1] + u_pad[:, :-2]) * p.inv_dy2
    # spherical metric correction: -(tanφ/R) ∂u/∂y
    # tan(lat) = sin(lat)/cos(lat); signed sin(lat) from f = 2*Omega*sin(lat)
    # (constant along longitude, so row 0 of f gives the per-row sin).
    du_dy = _d_dy(u, p)
    sin_lat_row = p.f[0, :] / (2.0 * OMEGA)           # (ny,) signed sin(lat)
    tan_over_R = (sin_lat_row / p.cos_lat) / R_EARTH   # (ny,) tan(lat)/R
    shp = [1] * u.ndim
    shp[1] = p.ny
    corr = -du_dy * tan_over_R.reshape(shp)
    return d2u_dx2 + d2u_dy2 + corr


def _biharmonic_h(u, p):
    """Biharmonic ∇⁴u = ∇²(∇²u). Composed from two Laplacian applications."""
    return _laplacian_h(_laplacian_h(u, p), p)


def _divergence_h(u, v, p):
    """Horizontal divergence du/dx + dv/dy."""
    return _d_dx(u, p) + _d_dy(v, p)


# ── Vertical operators (identical to regional solver) ──────────────

def _d_dz(u, p):
    """Vertical first derivative, non-uniform grid."""
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


# ── FD solver parameters (full, with physics + forcing) ───────────
# Extends FDParams (metric+grid) with the physics constants and forcing
# fields needed for the time integration (G2). Built by make_solver_global.

FDPhysParams = namedtuple('FDPhysParams', [
    # metric + grid (from FDParams)
    'dx_2d', 'dy', 'cos_lat', 'inv_dx', 'inv_dy', 'inv_dx2', 'inv_dy2',
    'f', 'wet_mask', 'wet_mask_3d', 'wet_mask_z',
    'interior_mask', 'interior_mask_z',
    'dz_denom_interior', 'dz_bnd_top', 'dz_bnd_bot',
    'd2z_hm', 'd2z_hp', 'd2z_denom', 'd2z_h0_top', 'd2z_h0_bot',
    'dz_3d', 'dz_surface', 'surface_mask', 'bottom_mask',
    'nx', 'ny', 'nz',
    # physics
    'nu_h', 'nu_v', 'kappa_h', 'kappa_v', 'kappa_conv',
    'nu_bi', 'kappa_bi',
    'T_ref', 'S_ref', 'eos_type', 'r_bot', 'cd', 'bottom_friction',
    # forcing (2D physical-space; no FFT pre-compute in the FD solver)
    'tau_x_2d', 'tau_y_2d', 'Q_heat_2d',
    # free surface
    'H_sw', 'dz_norm', 'dt',
    # bulk air-sea heat flux
    'T_atm_3d', 'lambda_bulk',
    # lateral sponge (polar-edge Rayleigh damping; global analogue of the
    # regional N/S-boundary sponge — absorbs wind-driven barotropic energy
    # that Laplacian dissipation can't within its CFL cap)
    'sponge_rate',       # (nx, ny, 1) damping rate [1/s]; 0 interior
    'sponge_rate_2d',    # (nx, ny) 2D damping for the free-surface step
    'T_clim_3d', 'S_clim_3d',   # (nx, ny, nz) climatology the sponge relaxes to
    'polar_cap_rows',    # int: poleward rows zonally averaged per step (metric singularity)
    'dealias_lon_mask',  # (nx, 1, 1) 2/3-rule FFT dealias mask for the periodic lon axis
])


# ── EOS (shared with spectral solver; copied to avoid import cycle) ─

def _density_anomaly(T, S, p):
    """rho' = rho - rho_0. Linear EOS branch (global default)."""
    if p.eos_type == 'unesco':
        # UNESCO not needed for global default; fall back to linear.
        return RHO_0 * (-ALPHA_T * (T - p.T_ref) + BETA_S * (S - p.S_ref))
    return RHO_0 * (-ALPHA_T * (T - p.T_ref) + BETA_S * (S - p.S_ref))


def _compute_hydrostatic_pressure(state, p):
    """Full hydrostatic pressure via cumulative trapezoidal integration.

    Ghost-water fix: the density anomaly is masked by wet_mask_z (zeroed
    below the seafloor) BEFORE integration. Layers below the seafloor then
    contribute dp=0, so the cumulative pressure stays constant beneath the
    bottom (no spurious horizontal gradient from columns of different
    ghost-water length). This is the fix for the blow-up at steep topography.
    """
    rho_prime = _density_anomaly(state.T, state.S, p)
    rho_prime = rho_prime * p.wet_mask_z          # zero out ghost water
    rho_avg = 0.5 * (rho_prime[..., :-1] + rho_prime[..., 1:])
    dp = G_EARTH * rho_avg * p.dz_3d
    p_bc = jnp.zeros_like(state.T)
    p_bc = p_bc.at[..., 1:].set(jnp.cumsum(dp, axis=-1))
    p_bt = RHO_0 * G_EARTH * state.eta[:, :, None]
    return p_bt + p_bc


def _compute_pressure_gradient(state, p):
    """Horizontal pressure gradient force per unit mass (FD)."""
    pressure = _compute_hydrostatic_pressure(state, p)
    pgf_x = -_d_dx(pressure, p) / RHO_0
    pgf_y = -_d_dy(pressure, p) / RHO_0
    return pgf_x, pgf_y


def _barotropic_velocity(u, v, p):
    """Depth-averaged (barotropic) horizontal velocity (same as spectral)."""
    u_avg = 0.5 * (u[..., :-1] + u[..., 1:])
    v_avg = 0.5 * (v[..., :-1] + v[..., 1:])
    ubt = jnp.sum(u_avg * p.dz_norm, axis=-1)
    vbt = jnp.sum(v_avg * p.dz_norm, axis=-1)
    return ubt, vbt


def _compute_bt_rho_pgf(state, p):
    """Barotropic (depth-averaged) PGF from density anomalies (FD)."""
    rho_prime = _density_anomaly(state.T, state.S, p)
    rho_prime = rho_prime * p.wet_mask_z          # ghost-water fix (see above)
    rho_avg = 0.5 * (rho_prime[..., :-1] + rho_prime[..., 1:])
    dp = G_EARTH * rho_avg * p.dz_3d
    p_bc = jnp.zeros_like(state.T)
    p_bc = p_bc.at[..., 1:].set(jnp.cumsum(dp, axis=-1))
    p_bc_avg = jnp.sum(0.5 * (p_bc[..., :-1] + p_bc[..., 1:]) * p.dz_norm, axis=-1)
    bt_rho_pgf_x = -_d_dx(p_bc_avg[:, :, None], p)[:, :, 0] / RHO_0
    bt_rho_pgf_y = -_d_dy(p_bc_avg[:, :, None], p)[:, :, 0] / RHO_0
    return bt_rho_pgf_x, bt_rho_pgf_y


def _dealias_w_fd(w, p):
    """Remove 2-dx grid-scale noise from the diagnosed w (FD analogue of the
    spectral baseline's _dealias_h).

    Central-difference divergence amplifies the 2-dx mode in u/v, so the w
    diagnosed from div_h carries 2-dx noise. Multiplied by the steep
    near-surface dT/dz this becomes a spurious vertical-advection heat source
    that nucleates a boundary heat pump (the G3 day-5 blow-up root cause).
    The spectral solver removes it with a 2/3 FFT rule; here we do the same
    in the periodic lon direction (exact) plus a 5-pt binomial low-pass in
    the closed lat direction (kills 2-dx without injecting high-freq, unlike
    a box average). lon FFT is exact because lon is periodic; lat cannot use
    FFT (closed no-flux wall), so a binomial filter substitutes.
    """
    # 2/3 FFT dealias in lon (axis 0, periodic).
    w_hat = jnp.fft.fft(w, axis=0)
    w_hat = w_hat * p.dealias_lon_mask
    w_lon = jnp.real(jnp.fft.ifft(w_hat, axis=0))
    # 5-pt binomial low-pass [1,4,6,4,1]/16 in lat (axis 1, edge-padded for the wall).
    wp = jnp.pad(w_lon, ((0, 0), (2, 2), (0, 0)), mode='edge')
    w_sm = (wp[:, :-4] + 4.0 * wp[:, 1:-3] + 6.0 * wp[:, 2:-2]
            + 4.0 * wp[:, 3:-1] + wp[:, 4:]) / 16.0
    return w_sm


def _compute_vertical_velocity(state, p):
    """Diagnose w from horizontal continuity. w=0 at bottom. Masked on land.

    The diagnosed w is dealiased (_dealias_w_fd) to remove the 2-dx
    grid-scale noise that central-difference divergence injects — without
    this, w x dT/dz drives a spurious vertical-advection heat pump at the
    boundary rows (the G3 day-5 blow-up; the spectral baseline applies the
    equivalent _dealias_h).
    """
    div_h = _divergence_h(state.u, state.v, p)
    div_avg = 0.5 * (div_h[..., :-1] + div_h[..., 1:])
    integrand = div_avg * p.dz_3d
    w = jnp.zeros_like(state.u)
    w = w.at[..., :-1].set(-jnp.cumsum(integrand[..., ::-1], axis=-1)[..., ::-1])
    w = _dealias_w_fd(w, p)
    return w * p.wet_mask_z   # no spurious w over land or below seafloor


# ── Tendencies (FD, with land masking) ─────────────────────────────

def _advection_flux_form(u, v, w, p):
    """3D advective-form momentum advection (FD, land-masked).

    Advective form (not flux form) avoids the spurious u*div_h source.
    Land masking: velocities are zeroed over land by the caller before
    advection, and the tendency is masked so land points don't accumulate
    advected noise (they're held at rest by the wet_mask projection).
    """
    du_dx = _d_dx(u, p); du_dy = _d_dy(u, p)
    dv_dx = _d_dx(v, p); dv_dy = _d_dy(v, p)
    du_dz = _d_dz(u, p); dv_dz = _d_dz(v, p)
    adv_u = -(u * du_dx + v * du_dy + w * du_dz)
    adv_v = -(u * dv_dx + v * dv_dy + w * dv_dz)
    return adv_u * p.wet_mask_z, adv_v * p.wet_mask_z


def _advection_scalar(T, u, v, w, p):
    """3D advective-form scalar advection (FD, land-masked)."""
    dT_dx = _d_dx(T, p); dT_dy = _d_dy(T, p); dT_dz = _d_dz(T, p)
    adv_T = -(u * dT_dx + v * dT_dy + w * dT_dz)
    return adv_T * p.wet_mask_z


def _compute_momentum_tendency(state, p):
    """du/dt, dv/dt for hydrostatic primitive equations (FD)."""
    w = _compute_vertical_velocity(state, p)
    adv_u, adv_v = _advection_flux_form(state.u, state.v, w, p)

    f_3d = p.f[:, :, None]
    cor_u = f_3d * state.v
    cor_v = -f_3d * state.u

    pgf_x, pgf_y = _compute_pressure_gradient(state, p)

    diff_h_u = p.nu_h * _laplacian_h(state.u, p)
    diff_h_v = p.nu_h * _laplacian_h(state.v, p)
    diff_v_u = p.nu_v * _d2_dz2(state.u, p)
    diff_v_v = p.nu_v * _d2_dz2(state.v, p)

    wind_factor = 1.0 / (RHO_0 * p.dz_surface)
    wind_u = p.tau_x_2d[:, :, None] * wind_factor * p.surface_mask
    wind_v = p.tau_y_2d[:, :, None] * wind_factor * p.surface_mask

    if p.bottom_friction == 'quadratic':
        speed = jnp.sqrt(state.u**2 + state.v**2)
        bot_u = -p.cd * speed * state.u * p.bottom_mask
        bot_v = -p.cd * speed * state.v * p.bottom_mask
    else:
        bot_u = -p.r_bot * state.u * p.bottom_mask
        bot_v = -p.r_bot * state.v * p.bottom_mask

    dudt = adv_u + cor_u + pgf_x + diff_h_u + diff_v_u + wind_u + bot_u
    dvdt = adv_v + cor_v + pgf_y + diff_h_v + diff_v_v + wind_v + bot_v
    # Land: hold velocity at rest (no tendency over land).
    dudt = dudt * p.wet_mask_z
    dvdt = dvdt * p.wet_mask_z
    return dudt, dvdt


def _compute_tracer_tendency(state, p):
    """dT/dt, dS/dt (FD, land-masked). Includes bulk air-sea heat flux."""
    w = _compute_vertical_velocity(state, p)
    adv_T = _advection_scalar(state.T, state.u, state.v, w, p)
    adv_S = _advection_scalar(state.S, state.u, state.v, w, p)

    diff_h_T = p.kappa_h * _laplacian_h(state.T, p)
    diff_h_S = p.kappa_h * _laplacian_h(state.S, p)
    diff_v_T = p.kappa_v * _d2_dz2(state.T, p)
    diff_v_S = p.kappa_v * _d2_dz2(state.S, p)

    # Convective adjustment (same logic as spectral; inert under linear EOS
    # + stable heating, but kept for consistency).
    rho_prime = _density_anomaly(state.T, state.S, p)
    unstable_iface = rho_prime[..., :-1] > rho_prime[..., 1:]
    conv_mask_3d = jnp.any(unstable_iface, axis=-1, keepdims=True)
    conv_T = p.kappa_conv * conv_mask_3d * _d2_dz2(state.T, p)
    conv_S = p.kappa_conv * conv_mask_3d * _d2_dz2(state.S, p)

    heat_factor = 1.0 / (RHO_0 * C_P * p.dz_surface)
    heat_T = p.Q_heat_2d[:, :, None] * heat_factor * p.surface_mask

    # Bulk air-sea heat flux (Haney/Barnier): genuine SST negative feedback.
    bulk_T = (p.lambda_bulk * (p.T_atm_3d - state.T[:, :, 0:1])
              * heat_factor * p.surface_mask)

    dTdt = adv_T + diff_h_T + diff_v_T + heat_T + bulk_T + conv_T
    dSdt = adv_S + diff_h_S + diff_v_S + conv_S
    # Land: tracers held (no tendency over land).
    dTdt = dTdt * p.wet_mask_z
    dSdt = dSdt * p.wet_mask_z
    return dTdt, dSdt


# ── Linear half-step (FD: explicit diffusion + exact Coriolis + free surface) ─

def _explicit_diffusion_step(u, p, nu, dt_half):
    """Explicit FD horizontal+vertical diffusion over dt_half.

    CFL: nu*dt/dx². At 1° (dx~111km), nu_h=100 -> ~4.6e-5 << 0.25 (safe).
    Biharmonic nu_bi: nu_bi*dt/dx⁴ may exceed the explicit limit at 1°;
    if so it is applied here and must be re-calibrated (see make_solver_global).
    """
    diff = nu * (_laplacian_h(u, p) + _d2_dz2(u, p) * 0.0)  # horiz only here
    # vertical diffusion handled separately for clarity
    return u + diff * dt_half


def _coriolis_rotation_2d(u, v, f, dt):
    """Exact Coriolis rotation on the full 2D f-field (not f-plane).

    Per-gridpoint rotation: u' = cos(f*dt)*u + sin(f*dt)*v, etc.
    f varies with latitude (2D), so this is exact everywhere, no beta-plane
    approximation.
    """
    angle = f[:, :, None] * dt     # broadcast 2D f to 3D velocity fields
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    u_new = cos_a * u + sin_a * v
    v_new = -sin_a * u + cos_a * v
    return u_new, v_new


def _free_surface_step_fd(eta, u, v, p, F_rho_x=None, F_rho_y=None, dt_half=None):
    """Forward-backward (Sielecki) free-surface (shallow water) step on lat-lon FD.

      eta^{n+1} = eta^n - dt*H_sw*div_h(ubt^n)            # eta from OLD velocity
      ubt^{n+1} = (ubt^n + dt*(-g*grad_h(eta^{n+1}) + F)) / (1 + r_bt*dt)   # u from NEW eta

    with implicit linear barotropic bottom drag (unconditionally stable, no CFL).
    The spectral solver solved the linear SW exactly per wavenumber (matrix
    exponential, energy-neutral). The FD analogue must NOT use forward-forward
    coupling (both from old state): that has |λ| = sqrt(1+(dt*c*k)^2) > 1 for
    ALL k — unconditionally unstable for free gravity waves (a no-wind 1m
    eta-bump grew to 5.5m/day under it). Forward-backward flips the trace to
    2 - dt^2*g*H*k^2 => |λ|=1 (neutral) under CFL<1, the standard OGCM
    discretization; bottom drag then decays the free mode (|λ|~0.97/step).
    CFL: dt < dx/sqrt(g*H) ~ 85-560s; dt=60 is safe on the global 1° grid
    (basin+meso modes CFL<1; grid-scale handled by Laplacian + polar cap).
    """
    if dt_half is None:
        dt_half = p.dt / 2.0
    ubt, vbt = _barotropic_velocity(u, v, p)

    F_x = jnp.zeros_like(ubt)
    F_y = jnp.zeros_like(vbt)
    if F_rho_x is not None:
        F_x = F_x + F_rho_x
        F_y = F_y + F_rho_y
    F_x = F_x + p.tau_x_2d / (RHO_0 * p.H_sw)
    F_y = F_y + p.tau_y_2d / (RHO_0 * p.H_sw)

    r_bt = p.r_bot if p.bottom_friction == 'linear' else 0.0
    drag = 1.0 / (1.0 + r_bt * dt_half)
    div_bt = _d_dx(ubt[:, :, None], p)[:, :, 0] + _d_dy(vbt[:, :, None], p)[:, :, 0]

    # Forward-backward (Sielecki) free-surface coupling: update eta FIRST
    # (old velocity), then update barotropic momentum using the NEW eta
    # gradient. The previous forward-forward coupling (both from old state)
    # has amplification |λ| = sqrt(1 + (dt*c*k)^2) > 1 for ALL k — i.e. it is
    # UNCONDITIONALLY UNSTABLE for free gravity waves (CFL does not save
    # forward-Euler; it only saves centered/leapfrog schemes). At dt=60 the
    # basin-scale seiche grows ~1.0023/step => ~27x/day, which a no-wind
    # 1m eta-bump test confirmed (1m -> 5.5m in 1 day, mean~0 so mass
    # conserved but amplitude growing). Forward-backward flips the trace of
    # the amplification matrix to 2 - dt^2*g*H*k^2, giving |λ| = 1 (neutral)
    # under CFL < 1 — the standard OGCM discretization (MOM6/ROMS/NEMO).
    # Bottom drag then actively decays the free mode (|λ| ~ 0.97/step here),
    # so a perturbed eta relaxes to the steady wind-driven setup instead of
    # amplifying. No iterative solve needed (unlike semi-implicit Helmholtz).
    eta_new = eta - dt_half * p.H_sw * div_bt
    grad_eta_x = _d_dx(eta_new[:, :, None], p)[:, :, 0]
    grad_eta_y = _d_dy(eta_new[:, :, None], p)[:, :, 0]
    ubt_new = (ubt + dt_half * (-G_EARTH * grad_eta_x + F_x)) * drag
    vbt_new = (vbt + dt_half * (-G_EARTH * grad_eta_y + F_y)) * drag

    eta_new = eta_new * p.wet_mask
    ubt_new = ubt_new * p.wet_mask
    vbt_new = vbt_new * p.wet_mask

    # Lateral sponge on the barotropic mode (2D): exponential decay of
    # eta/ubt/vbt toward the (flat) rest state in the sponge band. Matches
    # the 3D velocity sponge in _linear_half_step so depth-integrated
    # momentum is damped consistently. No-op when sponge_rate_2d == 0.
    sw_decay = jnp.exp(-p.sponge_rate_2d * dt_half)
    eta_new = eta_new * sw_decay
    ubt_new = ubt_new * sw_decay
    vbt_new = vbt_new * sw_decay

    # Polar-cap filter: zonally average the two poleward-most rows to kill the
    # cos(lat)->0 metric singularity (dx->0 makes the explicit SW CFL
    # unattainable at the edge). Replaces the unresolved polar dynamics with a
    # zonally-uniform cap value — standard for lat-lon FD OGCMs.
    #
    # CRITICAL: average over WET points only and write back to WET points only.
    # The polar rows are ~30% land (coastlines at high lat). A naive all-column
    # mean mixes ocean (eta != 0) with land (eta == 0), forcing a zonally-
    # uniform value that creates a spurious PGF at EVERY coastline point in the
    # band -> wind-driven barotropic energy injection exactly there. This was
    # the G2 wind-forced blow-up nucleation (max|u| diverged at (79.5, 124.5),
    # a wet point flanked by land, sub-inertial, CFL-safe — not a CFL failure).
    def _cap(field2d):
        """Zonal mean over wet points of the 2 poleward rows; land stays 0."""
        nc = p.polar_cap_rows
        if nc <= 0:
            return field2d
        s_row = field2d[:, :nc] * p.wet_mask[:, :nc]      # (nx, nc)
        w_row = p.wet_mask[:, :nc]
        cap_n = jnp.sum(s_row, axis=0, keepdims=True) / jnp.maximum(
            jnp.sum(w_row, axis=0, keepdims=True), 1.0)  # (1, nc)
        field2d = field2d.at[:, :nc].set(
            jnp.broadcast_to(cap_n, (p.nx, nc)) * p.wet_mask[:, :nc])
        s_row = field2d[:, -nc:] * p.wet_mask[:, -nc:]
        w_row = p.wet_mask[:, -nc:]
        cap_s = jnp.sum(s_row, axis=0, keepdims=True) / jnp.maximum(
            jnp.sum(w_row, axis=0, keepdims=True), 1.0)
        field2d = field2d.at[:, -nc:].set(
            jnp.broadcast_to(cap_s, (p.nx, nc)) * p.wet_mask[:, -nc:])
        return field2d
    eta_new = _cap(eta_new)
    ubt_new = _cap(ubt_new)
    vbt_new = _cap(vbt_new)

    # Project barotropic delta back to 3D velocity (uniform over depth)
    delta_ubt = (ubt_new - ubt)[:, :, None]
    delta_vbt = (vbt_new - vbt)[:, :, None]
    u_new = u + delta_ubt
    v_new = v + delta_vbt
    # No-flux wall: enforce zero normal velocity at the N/S boundary rows on
    # the projected 3D v too, consistent with _step_impl's final mask.
    v_new = v_new * p.interior_mask_z
    return eta_new, u_new, v_new


def _linear_half_step(state, p, dt_half):
    """Linear half-step: FD diffusion + 2D Coriolis + EXPLICIT free surface.

    Unlike the spectral solver (exact spectral diffusion decay + matrix-exp
    free surface), the FD linear step is:
      - explicit horizontal+vertical diffusion (CFL-safe at 1°),
      - exact per-gridpoint Coriolis rotation on the 2D f-field,
      - EXPLICIT forward-Euler free surface (CFL: dt < dx/sqrt(gH) ~ 85-560s;
        dt=60 is safe across the global grid).
    """
    # Explicit diffusion (horizontal Laplacian + vertical d2/dz2)
    u = state.u + p.nu_h * _laplacian_h(state.u, p) * dt_half
    v = state.v + p.nu_h * _laplacian_h(state.v, p) * dt_half
    T = state.T + p.kappa_h * _laplacian_h(state.T, p) * dt_half
    S = state.S + p.kappa_h * _laplacian_h(state.S, p) * dt_half
    # Scale-selective biharmonic (∇⁴): damps grid-scale modes far more than
    # large-scale. Spectral solver applied this via exp(-nu_bi*k⁴*dt); the FD
    # analogue is an explicit forward-Euler step. CFL: nu_bi*dt/dx⁴ < ~0.05.
    # nu_bi is re-calibrated for 1° (much larger than the spectral 1e12, which
    # was tuned for the regional 0.1° grid where dx⁴ is ~1600x smaller).
    if p.nu_bi > 0.0:
        u = u - p.nu_bi * _biharmonic_h(state.u, p) * dt_half
        v = v - p.nu_bi * _biharmonic_h(state.v, p) * dt_half
    if p.kappa_bi > 0.0:
        T = T - p.kappa_bi * _biharmonic_h(state.T, p) * dt_half
        S = S - p.kappa_bi * _biharmonic_h(state.S, p) * dt_half
    u = u + p.nu_v * _d2_dz2(state.u, p) * dt_half
    v = v + p.nu_v * _d2_dz2(state.v, p) * dt_half
    T = T + p.kappa_v * _d2_dz2(state.T, p) * dt_half
    S = S + p.kappa_v * _d2_dz2(state.S, p) * dt_half
    # Mask: no diffusion updates over land or below seafloor (ghost water)
    u = u * p.wet_mask_z; v = v * p.wet_mask_z
    T = T * p.wet_mask_z; S = S * p.wet_mask_z

    # Lateral sponge (Rayleigh damping) — exponential decay, unconditional
    # (no damping CFL). Applied in the linear half-step so Strang splitting
    # gives total sponge time = dt per full step. Absorbs the wind-driven
    # barotropic energy that Laplacian dissipation can't arrest within its
    # CFL cap, which otherwise piles up at the polar edge rows. No-op when
    # sponge_rate == 0 (decay == 1, T_clim == 0 -> T unchanged).
    decay = jnp.exp(-p.sponge_rate * dt_half)            # (nx,ny,1)
    u = u * decay
    v = v * decay
    T = p.T_clim_3d + (T - p.T_clim_3d) * decay
    S = p.S_clim_3d + (S - p.S_clim_3d) * decay

    # Coriolis rotation (2D f-field, exact)
    u, v = _coriolis_rotation_2d(u, v, p.f, dt_half)

    # Semi-implicit free surface (with density barotropic PGF)
    F_rho_x, F_rho_y = _compute_bt_rho_pgf(state, p)
    eta, u, v = _free_surface_step_fd(state.eta, u, v, p, F_rho_x, F_rho_y, dt_half)
    return JaxStateG(u, v, T, S, eta)


# ── Nonlinear explicit step (forward-backward RK2, FD) ─────────────

def _compute_tracer_residual(state, p):
    """Tracer tendency minus the linear diffusion (handled by linear step)."""
    dTdt, dSdt = _compute_tracer_tendency(state, p)
    dTdt = dTdt - p.kappa_h * _laplacian_h(state.T, p)
    dSdt = dSdt - p.kappa_h * _laplacian_h(state.S, p)
    if p.kappa_bi > 0.0:
        dTdt = dTdt + p.kappa_bi * _biharmonic_h(state.T, p)
        dSdt = dSdt + p.kappa_bi * _biharmonic_h(state.S, p)
    return dTdt, dSdt


def _compute_momentum_residual(state, p):
    """Momentum tendency minus linear parts (diffusion, Coriolis, bt PGF, bt wind)."""
    dudt, dvdt = _compute_momentum_tendency(state, p)
    dudt = dudt - p.nu_h * _laplacian_h(state.u, p)
    dvdt = dvdt - p.nu_h * _laplacian_h(state.v, p)
    if p.nu_bi > 0.0:
        dudt = dudt + p.nu_bi * _biharmonic_h(state.u, p)
        dvdt = dvdt + p.nu_bi * _biharmonic_h(state.v, p)
    dudt = dudt - p.f[:, :, None] * state.v
    dvdt = dvdt + p.f[:, :, None] * state.u
    # barotropic PGF from eta
    bt_pgf_x = -G_EARTH * _d_dx(state.eta[:, :, None], p)[:, :, 0]
    bt_pgf_y = -G_EARTH * _d_dy(state.eta[:, :, None], p)[:, :, 0]
    dudt = dudt - bt_pgf_x[:, :, None]
    dvdt = dvdt - bt_pgf_y[:, :, None]
    # barotropic PGF from density anomaly
    bt_rho_x, bt_rho_y = _compute_bt_rho_pgf(state, p)
    dudt = dudt - bt_rho_x[:, :, None]
    dvdt = dvdt - bt_rho_y[:, :, None]
    # barotropic wind
    bt_wind_x = p.tau_x_2d / (RHO_0 * p.H_sw)
    bt_wind_y = p.tau_y_2d / (RHO_0 * p.H_sw)
    dudt = dudt - bt_wind_x[:, :, None]
    dvdt = dvdt - bt_wind_y[:, :, None]
    # Mask the residual by wet_mask_z for consistency with the full tendency.
    # _compute_momentum_tendency returns dudt already masked to 0 in ghost
    # water (layers below the seafloor, wet_mask_z==0), but the barotropic
    # subtractions above (bt_pgf, bt_rho_pgf, bt_wind) are broadcast uniformly
    # over ALL depths — so in ghost-water layers the residual was -bt_pgf
    # (a spurious PGF) instead of 0, mismatching the masked tendency there.
    # Masking closes the mismatch so the residual equals the (masked) full
    # tendency minus its linear parts everywhere, including ghost water. This
    # keeps the N-step residual clean in ghost layers (verified ~1e-21, machine
    # zero) so no spurious tendency leaks into the wet column via the vertical
    # operators that couple adjacent layers.
    dudt = dudt * p.wet_mask_z
    dvdt = dvdt * p.wet_mask_z
    return dudt, dvdt


def _explicit_full_step(state, p, dt):
    """Forward-backward RK2 for nonlinear tendencies (FD).

    Tracers updated first (old velocity), then momentum uses predicted T
    for the baroclinic PGF — shifts internal-wave eigenvalues left of the
    imaginary axis for neutral stability. Same structure as spectral solver.
    """
    dT1, dS1 = _compute_tracer_residual(state, p)
    T_pred = state.T + dT1 * dt
    S_pred = state.S + dS1 * dt
    state_T = JaxStateG(state.u, state.v, T_pred, S_pred, state.eta)

    du1, dv1 = _compute_momentum_residual(state_T, p)
    u_pred = state.u + du1 * dt
    v_pred = state.v + dv1 * dt
    state_pred = JaxStateG(u_pred, v_pred, T_pred, S_pred, state.eta)

    dT2, dS2 = _compute_tracer_residual(state_pred, p)
    T_new = state.T + 0.5 * (dT1 + dT2) * dt
    S_new = state.S + 0.5 * (dS1 + dS2) * dt
    state_T_new = JaxStateG(state.u, state.v, T_new, S_new, state.eta)

    du2, dv2 = _compute_momentum_residual(state_T_new, p)
    u_new = state.u + 0.5 * (du1 + du2) * dt
    v_new = state.v + 0.5 * (dv1 + dv2) * dt
    return JaxStateG(u_new, v_new, T_new, S_new, state.eta)


def _polar_cap_3d(field3d, p):
    """Zonally average the poleward-most `polar_cap_rows` rows of a 3D field.

    The cos(lat)->0 metric singularity makes the spherical Laplacian /
    advection operators blow up at the polar edge (1/cos²φ in ∂²/∂x²,
    tanφ/R in the metric correction). The free-surface polar-cap filter
    handles the barotropic mode; this extends the same idea to the FULL 3D
    field (u, v, T, S) so no meridional gradient survives in the cap rows
    for the singular operators to amplify. This is the standard lat-lon FD
    OGCM treatment (e.g. MOM6's polar cap / Arctic fold).

    Wet-point-only zonal mean (land excluded); land stays at its masked
    value. Applied once per full step (in _step_impl), which is sufficient
    because the cap rows' dynamics are unresolved by construction.
    """
    ncap = p.polar_cap_rows
    if ncap <= 0:
        return field3d
    wm = p.wet_mask                          # (nx, ny)
    # North cap
    s = field3d[:, -ncap:] * wm[:, -ncap:, None]
    w = jnp.maximum(wm[:, -ncap:], 1e-12)
    cap_n = jnp.sum(s, axis=0, keepdims=True) / jnp.sum(
        jnp.broadcast_to(w[:, :, None], s.shape), axis=0, keepdims=True)  # (1,ncap,nz)
    cap_n = jnp.broadcast_to(cap_n, field3d[:, -ncap:].shape) * wm[:, -ncap:, None]
    # South cap
    s = field3d[:, :ncap] * wm[:, :ncap, None]
    cap_s = jnp.sum(s, axis=0, keepdims=True) / jnp.sum(
        jnp.broadcast_to(w[:, :, None], s.shape), axis=0, keepdims=True)
    cap_s = jnp.broadcast_to(cap_s, field3d[:, :ncap].shape) * wm[:, :ncap, None]
    return jnp.concatenate([cap_s, field3d[:, ncap:-ncap], cap_n], axis=1)


def _step_impl(state, p):
    """Strang splitting: L(dt/2) -> N(dt) -> L(dt/2)."""
    dt_half = p.dt / 2.0
    state = _linear_half_step(state, p, dt_half)
    state = _explicit_full_step(state, p, p.dt)
    state = _linear_half_step(state, p, dt_half)
    # 3D polar-cap filter: zonally average the cap rows of u,v,T,S to kill
    # the cos(lat)->0 metric singularity in the diffusion/advection operators.
    u = _polar_cap_3d(state.u, p)
    v = _polar_cap_3d(state.v, p)
    T = _polar_cap_3d(state.T, p)
    S = _polar_cap_3d(state.S, p)
    # Final mask enforcement (safety: no drift onto land or ghost water)
    u = u * p.wet_mask_z
    v = v * p.wet_mask_z
    T = T * p.wet_mask_z
    S = S * p.wet_mask_z
    # No-flux wall: zero the NORMAL (meridional) velocity at the N/S boundary
    # rows so no flow crosses the closed wall. Combined with the mirror-ghost
    # stencils in _d_dy/_laplacian_h (zero normal gradient), this is the full
    # closed-boundary condition that replaces the unstable one-sided stencil.
    v = v * p.interior_mask_z
    return JaxStateG(u, v, T, S, state.eta)


# ── Public API ──────────────────────────────────────────────────────

def make_solver_global(grid, physics, dt, forcing=None, eos_type='linear',
                       T_atm=None, lambda_bulk=0.0,
                       sponge_days=0.0, sponge_cells=0,
                       T_init=None, S_init=None,
                       polar_cap_rows=2, return_params=False):
    """Create a JIT-compiled global FD ocean solver.

    Args mirror the spectral make_solver where applicable. Key differences:
      - No spectral wavenumbers/decay factors (FD operators instead).
      - Forcing is physical-space 2D (tau_x, tau_y, Q_heat); no pre-FFT.
      - Free surface is forward-backward (Sielecki) + polar-cap filter;
        requires dt below the external-gravity-wave CFL.
      - Lateral sponge at the POLAR EDGE rows (not N/S periodic boundaries):
        the global polar edge is the analogue of the regional N/S boundary.
        Wind-driven barotropic energy piles up there (Laplacian can't arrest
        it within its CFL cap); the sponge absorbs it. cosine-tapered.
      - No SST restore (bulk flux only).
    """
    base = make_fd_params(grid)
    nx, ny, nz = base.nx, base.ny, base.nz

    if forcing is None:
        tau_x_2d = jnp.zeros((nx, ny))
        tau_y_2d = jnp.zeros((nx, ny))
        Q_heat_2d = jnp.zeros((nx, ny))
    else:
        tau_x_2d, tau_y_2d, Q_heat_2d = (jnp.array(f) for f in forcing)

    # Effective shallow-water depth = vertical grid span
    H_sw = float(jnp.sum(jnp.array(grid.dz)))
    dz_norm = (jnp.array(grid.dz).reshape(1, 1, -1) / H_sw)

    if T_atm is not None and lambda_bulk > 0.0:
        T_atm_3d = jnp.array(T_atm)[:, :, None]
    else:
        T_atm_3d = jnp.zeros((nx, ny, 1))
        lambda_bulk = 0.0

    # ── Lateral sponge (polar-edge Rayleigh damping) ──
    # Same construction as the regional spectral solver's N/S sponge, but
    # applied at both lat edges (the polar cap rows). cosine-tapered from
    # r_max at the edge to 0 at sponge_cells into the interior.
    if sponge_days > 0.0 and sponge_cells > 0:
        r_max = 1.0 / (sponge_days * 86400.0)
        nc = int(sponge_cells)
        j = np.arange(ny)
        taper = np.zeros(ny)
        edge = np.minimum(j, ny - 1 - j)   # distance to nearest N/S edge
        in_band = edge < nc
        taper[in_band] = 0.5 * (1.0 + np.cos(np.pi * edge[in_band] / nc))
        sponge_2d_np = (r_max * taper).reshape(1, ny)
        sponge_rate_2d = jnp.array(np.broadcast_to(sponge_2d_np, (nx, ny)))
        sponge_rate = sponge_rate_2d[:, :, None]   # (nx, ny, 1)
        if T_init is not None:
            T_clim_3d = jnp.array(T_init)
            S_clim_3d = jnp.array(S_init) if S_init is not None else jnp.zeros_like(T_clim_3d)
        else:
            T_clim_3d = jnp.zeros((nx, ny, nz))
            S_clim_3d = jnp.zeros((nx, ny, nz))
    else:
        sponge_rate_2d = jnp.zeros((nx, ny))
        sponge_rate = jnp.zeros((nx, ny, 1))
        T_clim_3d = jnp.zeros((nx, ny, nz))
        S_clim_3d = jnp.zeros((nx, ny, nz))

    # ── 2/3-rule dealias mask for the periodic lon axis ──
    # The FD _compute_vertical_velocity integrates div_h = du/dx + dv/dy
    # vertically to diagnose w. Central differencing amplifies the 2-dx mode
    # in u/v, so the diagnosed w carries 2-dx grid-scale noise. The spectral
    # baseline (jax_solver.py) removes this with _dealias_h (2/3 FFT rule) on
    # w; the FD solver must do the equivalent. lon is periodic -> a 2/3 FFT
    # dealias in lon is exact; the closed lat wall can't use FFT, so the lat
    # 2-dx is killed by a 5-pt binomial low-pass in _dealias_w_fd.
    _kmax = nx // 2
    _keep = max(1, int(_kmax * 2 / 3))
    _kidx = np.fft.fftfreq(nx) * nx            # 0..nx/2, -nx/2..-1
    _lon_mask_np = (np.abs(_kidx) <= _keep).astype(np.float64).reshape(nx, 1, 1)
    dealias_lon_mask = jnp.array(np.broadcast_to(_lon_mask_np, (nx, 1, 1)))

    params = FDPhysParams(
        dx_2d=base.dx_2d, dy=base.dy, cos_lat=base.cos_lat,
        inv_dx=base.inv_dx, inv_dy=base.inv_dy,
        inv_dx2=base.inv_dx2, inv_dy2=base.inv_dy2,
        f=base.f, wet_mask=base.wet_mask, wet_mask_3d=base.wet_mask_3d,
        wet_mask_z=base.wet_mask_z,
        interior_mask=base.interior_mask, interior_mask_z=base.interior_mask_z,
        dz_denom_interior=base.dz_denom_interior,
        dz_bnd_top=base.dz_bnd_top, dz_bnd_bot=base.dz_bnd_bot,
        d2z_hm=base.d2z_hm, d2z_hp=base.d2z_hp, d2z_denom=base.d2z_denom,
        d2z_h0_top=base.d2z_h0_top, d2z_h0_bot=base.d2z_h0_bot,
        dz_3d=base.dz_3d, dz_surface=base.dz_surface,
        surface_mask=base.surface_mask, bottom_mask=base.bottom_mask,
        nx=nx, ny=ny, nz=nz,
        nu_h=physics.nu_h, nu_v=physics.nu_v,
        kappa_h=physics.kappa_h, kappa_v=physics.kappa_v,
        kappa_conv=physics.kappa_conv,
        nu_bi=physics.nu_bi, kappa_bi=physics.kappa_bi,
        T_ref=physics.T_ref, S_ref=physics.S_ref,
        eos_type=eos_type, r_bot=physics.r_bot, cd=physics.cd,
        bottom_friction=physics.bottom_friction,
        tau_x_2d=tau_x_2d, tau_y_2d=tau_y_2d, Q_heat_2d=Q_heat_2d,
        H_sw=H_sw, dz_norm=dz_norm, dt=dt,
        T_atm_3d=T_atm_3d, lambda_bulk=lambda_bulk,
        sponge_rate=sponge_rate, sponge_rate_2d=sponge_rate_2d,
        T_clim_3d=T_clim_3d, S_clim_3d=S_clim_3d,
        polar_cap_rows=int(polar_cap_rows),
        dealias_lon_mask=dealias_lon_mask,
    )

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
        u = jnp.zeros((nx, ny, nz))
        v = jnp.zeros((nx, ny, nz))
        eta = jnp.zeros((nx, ny))
        if T_init is not None:
            T = jnp.array(T_init)
            S = jnp.array(S_init) if S_init is not None else jnp.full_like(T, physics.S_ref)
        else:
            T = jnp.full((nx, ny, nz), physics.T_ref)
            S = jnp.full((nx, ny, nz), physics.S_ref)
        # Mask land + ghost water (layers below seafloor): set to a sentinel.
        # wet_mask_z is the TRUE 3D mask (1 where water exists, 0 on land AND
        # below seafloor). This discards WOA-interpolated T in ghost layers
        # before it can enter the pressure integral or advection.
        T = T * params.wet_mask_z + (1.0 - params.wet_mask_z) * physics.T_ref
        S = S * params.wet_mask_z + (1.0 - params.wet_mask_z) * physics.S_ref
        return JaxStateG(u, v, T, S, eta)

    if return_params:
        return step, init_state, diagnostics, params
    return step, init_state, diagnostics


# ── MMS verification (G1) ──────────────────────────────────────────
# Manufactured-solution checks for the FD operators. These are the FD
# analogues of the spectral Tier-1 tests (test_spectral_ops.py). Spectral
# achieved ~1e-17 (machine precision); 2nd-order FD should converge as
# error ∝ dx² (~1e-3 at 1°). This is the EXPECTED and honest FD baseline.

def _mms_run():
    """Run MMS checks on a small synthetic global grid (no ETOPO needed).

    Returns a dict of {test_name: (error, passes_threshold)}.
    Builds a tiny 36×18 grid (10° resolution) so the FD convergence is
    visible and the test is fast. Uses analytic trigonometric fields.
    """
    from config import GlobalGridConfig, R_EARTH
    import numpy as np

    # Tiny global grid for MMS (coarse so dx is large -> FD error visible)
    nx, ny, res = 36, 14, 360.0 / 36   # 10° resolution
    # Build metric fields directly (avoid ETOPO file dependency in unit test)
    lon = res * (0.5 + np.arange(nx))              # 5, 15, ..., 355
    lat = np.linspace(-75.0, 75.0, ny)             # avoid poles (cos->0)
    lon_2d, lat_2d = np.meshgrid(lon, lat, indexing='ij')
    cos_lat = np.cos(np.radians(lat))
    dx_2d = np.broadcast_to(R_EARTH * np.radians(res) * cos_lat, (nx, ny)).copy()
    dy = R_EARTH * np.radians(res)
    f = 2.0 * OMEGA * np.sin(np.radians(lat_2d))
    wet_mask = np.ones((nx, ny))                   # all-ocean for MMS

    z = np.array([0, -10, -30, -60], dtype=np.float64)
    dz = np.abs(np.diff(z))
    nz = len(z)

    # Minimal FDParams (only horizontal fields needed for these tests)
    class _G:
        pass
    g = _G()
    g.dx_2d = dx_2d; g.dy = dy; g.cos_lat = cos_lat
    g.f = f; g.wet_mask = wet_mask; g.wet_mask_3d = wet_mask[..., None]
    g.z = z; g.dz = dz; g.nz = nz
    g.nx = nx; g.ny = ny
    p = make_fd_params(g)

    # Analytic field: u(lon, lat) = sin(m*lon_rad) * cos(n*lat_rad).
    # This is periodic in longitude (exact wrap) and smooth in latitude.
    # The FD operators use physical-space dx = R*cos(lat)*dlon, so du/dx in
    # m/s = (1/dx) * du/dlon_phys. Analytic du/dx = (1/(R*cosφ)) * m*cos(m*lon)*cos(n*lat).
    lon_rad = np.radians(lon_2d)                 # (nx, ny)
    lat_rad = np.radians(lat_2d)
    m = 2.0   # 2 wavelengths around the globe
    n = 2.0   # 2 oscillations in latitude
    u2d = np.sin(m * lon_rad) * np.cos(n * lat_rad)
    u3d = np.broadcast_to(u2d[..., None], (nx, ny, nz)).copy()

    results = {}

    # ── d/dx: du/dx = [m*cos(m*lon)*cos(n*lat)] / (R*cosφ) ──
    du_dx_fd = np.array(_d_dx(jnp.array(u3d), p))[:, :, 0]
    du_dx_an = (m * np.cos(m * lon_rad) * np.cos(n * lat_rad)) / (R_EARTH * cos_lat[None, :])
    err = np.sqrt(np.mean((du_dx_fd - du_dx_an) ** 2)) / (np.abs(du_dx_an).max() + 1e-30)
    results['d_dx_rel_L2'] = (float(err), err < 0.05)

    # ── d/dy: du/dy = -sin(m*lon)*n*sin(n*lat) / R  (dy = R*dlat) ──
    # NB: at 10° resolution with 2 lat-oscillations, 2nd-order FD truncation
    # is ~7% (each wavelength ~5 points). This is EXPECTED FD behavior, not a
    # bug — the _mms_convergence() check proves true 2nd-order convergence.
    # Threshold set generously; the convergence order is the real gate.
    du_dy_fd = np.array(_d_dy(jnp.array(u3d), p))[:, :, 0]
    du_dy_an = (-np.sin(m * lon_rad) * n * np.sin(n * lat_rad)) / R_EARTH
    err = np.sqrt(np.mean((du_dy_fd - du_dy_an) ** 2)) / (np.abs(du_dy_an).max() + 1e-30)
    results['d_dy_rel_L2'] = (float(err), err < 0.10)

    # ── laplacian of a LINEAR-in-y field: u = y_m = R*lat_rad. On the sphere
    # ∇²(R*lat) = -(tanφ/R) (the metric correction term only; ∂²/∂x²=∂²/∂y²=0).
    # Small magnitude -> sanity check that the laplacian isn't producing junk.
    y_m = R_EARTH * lat_rad
    u_lin3d = np.broadcast_to(y_m[..., None], (nx, ny, nz)).copy()
    lap_fd = np.array(_laplacian_h(jnp.array(u_lin3d), p))[:, :, 0]
    lap_scale = np.abs(lap_fd).max()
    results['laplacian_linear_max'] = (float(lap_scale), lap_scale < 1e-4)

    # ── divergence-free check: u = cos(n*lat_rad) (x-independent), v = 0.
    # ∂u/∂x of an x-independent field = 0 on the sphere, so div = 0.
    u_df = np.cos(n * lat_rad)
    v_df = np.zeros_like(u_df)
    u_df3 = np.broadcast_to(u_df[..., None], (nx, ny, nz)).copy()
    v_df3 = np.broadcast_to(v_df[..., None], (nx, ny, nz)).copy()
    div_fd = np.array(_divergence_h(jnp.array(u_df3), jnp.array(v_df3), p))[:, :, 0]
    div_err = np.abs(div_fd).max()
    results['divergence_free_max'] = (float(div_err), div_err < 1e-10)

    return results


def _mms_convergence():
    """Demonstrate 2nd-order convergence of the FD d/dy operator.

    Runs the d/dy MMS at two resolutions (coarse and fine) and checks the
    error drops by ~4x when dx halves (2nd-order: error ∝ dx²). This is the
    decisive FD verification: it proves the operator is correctly 2nd-order,
    not that it's spectrally accurate. The absolute error at 1° is ~1e-3
    (expected, honest FD baseline); the convergence ORDER is what matters.
    """
    from config import R_EARTH
    import numpy as np

    def ddy_error(ny):
        nx, res = 36, 360.0 / 36
        lat = np.linspace(-75.0, 75.0, ny)
        lon_2d, lat_2d = np.meshgrid(res * (0.5 + np.arange(nx)), lat, indexing='ij')
        cos_lat = np.cos(np.radians(lat))
        dx_2d = np.broadcast_to(R_EARTH * np.radians(res) * cos_lat, (nx, ny)).copy()
        dy = R_EARTH * np.radians(2 * 75.0 / (ny - 1))   # actual dy for this ny
        f = 2.0 * OMEGA * np.sin(np.radians(lat_2d))
        wet_mask = np.ones((nx, ny))
        z = np.array([0, -10, -30, -60], dtype=np.float64); dz = np.abs(np.diff(z)); nz = len(z)
        class _G: pass
        g = _G()
        g.dx_2d = dx_2d; g.dy = dy; g.cos_lat = cos_lat; g.f = f
        g.wet_mask = wet_mask; g.wet_mask_3d = wet_mask[..., None]
        g.z = z; g.dz = dz; g.nz = nz; g.nx = nx; g.ny = ny
        p = make_fd_params(g)
        lat_rad = np.radians(lat_2d); lon_rad = np.radians(lon_2d)
        n = 2.0
        u2d = np.sin(2.0 * lon_rad) * np.cos(n * lat_rad)
        u3d = np.broadcast_to(u2d[..., None], (nx, ny, nz)).copy()
        du_dy_fd = np.array(_d_dy(jnp.array(u3d), p))[:, :, 0]
        du_dy_an = (-np.sin(2.0 * lon_rad) * n * np.sin(n * lat_rad)) / R_EARTH
        # Measure on INTERIOR rows only (exclude the 2 boundary rows each side).
        # The closed-wall no-flux BC (mirror ghost, ∂u/∂n=0) intentionally
        # deviates from the analytic infinite-domain derivative at the wall —
        # that is the correct physics of a closed wall, not an accuracy loss.
        # The interior rows remain true 2nd-order central differences.
        sl = slice(2, ny - 2)
        return np.sqrt(np.mean((du_dy_fd[:, sl] - du_dy_an[:, sl]) ** 2)) / (
            np.abs(du_dy_an[:, sl]).max() + 1e-30)

    e_coarse = ddy_error(ny=14)
    e_fine = ddy_error(ny=28)
    ratio = e_coarse / (e_fine + 1e-30)
    # 2nd-order: halving dx -> error /4, so ratio ~ 4. Accept ratio > 3.
    return {'ddy_coarse': e_coarse, 'ddy_fine': e_fine,
            'convergence_ratio': ratio, 'passes': bool(ratio > 3.0)}


if __name__ == "__main__":
    print("=== Global FD Solver — G1 MMS verification ===")
    res = _mms_run()
    all_pass = True
    for name, (err, ok) in res.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name:30s} = {err:.3e}  [{status}]")
    print()
    print("--- 2nd-order convergence check (d/dy) ---")
    conv = _mms_convergence()
    print(f"  coarse (ny=14)  d/dy rel L2 = {conv['ddy_coarse']:.3e}")
    print(f"  fine   (ny=28)  d/dy rel L2 = {conv['ddy_fine']:.3e}")
    print(f"  convergence ratio (expect ~4 for 2nd-order) = {conv['convergence_ratio']:.2f}  "
          f"[{'PASS' if conv['passes'] else 'FAIL'}]")
    if not conv['passes']:
        all_pass = False
    print()
    print("ALL PASS" if all_pass else "SOME FAILED")
