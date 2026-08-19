"""
Spectral Operators Library — FFT-based spatial derivatives for ocean equations.

Core principle: In Fourier space, differential operators become diagonal matrices.
This means:
  - d/dx      -> IFFT(i*k * FFT(u))         O(N log N), infinite order accuracy
  - laplacian -> IFFT(-k^2 * FFT(u))         O(N log N)
  - Poisson   -> IFFT(f_hat / (-k^2))        O(N log N), non-iterative!
  - exp(L*dt) -> IFFT(exp(L_diag*dt) * FFT(u))  element-wise, unconditionally stable

Framework: JAX (with numpy fallback for testing).
"""
import numpy as np
from numpy import pi

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jnp = np
    HAS_JAX = False


# ── Wavenumber generation ───────────────────────────────────────────

def wavenumbers(n, dx):
    """
    Generate Fourier wavenumbers for a periodic domain of n points with spacing dx.

    Returns k in radians per unit length: k = 2*pi * fftfreq(n, dx)
    Shape: (n,) for 1D, broadcastable for 2D/3D.
    """
    return 2.0 * pi * np.fft.fftfreq(n, d=dx)


def wavenumber_grid(nx, ny, dx, dy):
    """
    2D wavenumber grids for spectral operations on a (nx, ny) periodic domain.

    Returns (kx, ky, k2) where:
      kx, ky: 2D arrays of shape (nx, ny)
      k2:     kx^2 + ky^2  (for Laplacian / Poisson)

    Array convention: axis 0 = x (zonal), axis 1 = y (meridional).
    This matches numpy.fft.fft2 / jnp.fft.fft2 default axes.
    """
    kx_1d = wavenumbers(nx, dx)
    ky_1d = wavenumbers(ny, dy)
    kx, ky = np.meshgrid(kx_1d, ky_1d, indexing='ij')
    k2 = kx**2 + ky**2
    return kx, ky, k2


# ── First derivative ────────────────────────────────────────────────

def d_dx(u, dx):
    """
    Spectral first derivative in x: du/dx = IFFT(i*kx * FFT(u))

    Args:
        u: 2D array (nx, ny) or 3D (nx, ny, nz)
        dx: grid spacing in x [m]

    Returns: du/dx, same shape as u
    """
    nx = u.shape[0]
    kx = wavenumbers(nx, dx)
    # Reshape kx for broadcasting: (nx, 1, 1, ...) depending on ndim
    shape = [nx] + [1] * (u.ndim - 1)
    kx = kx.reshape(shape)

    u_hat = np.fft.fft(u, axis=0)
    du_hat = 1j * kx * u_hat
    return np.real(np.fft.ifft(du_hat, axis=0))


def d_dy(u, dy):
    """
    Spectral first derivative in y: du/dy = IFFT(i*ky * FFT(u))

    Args:
        u: 2D array (nx, ny) or 3D (nx, ny, nz)
        dy: grid spacing in y [m]

    Returns: du/dy, same shape as u
    """
    ny = u.shape[1]
    ky = wavenumbers(ny, dy)
    shape = [1, ny] + [1] * (u.ndim - 2)
    ky = ky.reshape(shape)

    u_hat = np.fft.fft(u, axis=1)
    du_hat = 1j * ky * u_hat
    return np.real(np.fft.ifft(du_hat, axis=1))


# ── Second derivative / Laplacian ───────────────────────────────────

def d2_dx2(u, dx):
    """Spectral second derivative in x: d2u/dx2 = IFFT(-kx^2 * FFT(u))"""
    nx = u.shape[0]
    kx = wavenumbers(nx, dx)
    shape = [nx] + [1] * (u.ndim - 1)
    kx2 = (kx**2).reshape(shape)

    u_hat = np.fft.fft(u, axis=0)
    d2u_hat = -kx2 * u_hat
    return np.real(np.fft.ifft(d2u_hat, axis=0))


def d2_dy2(u, dy):
    """Spectral second derivative in y: d2u/dy2 = IFFT(-ky^2 * FFT(u))"""
    ny = u.shape[1]
    ky = wavenumbers(ny, dy)
    shape = [1, ny] + [1] * (u.ndim - 2)
    ky2 = (ky**2).reshape(shape)

    u_hat = np.fft.fft(u, axis=1)
    d2u_hat = -ky2 * u_hat
    return np.real(np.fft.ifft(d2u_hat, axis=1))


def laplacian_h(u, dx, dy):
    """
    Horizontal Laplacian: nabla_h^2 u = IFFT(-(kx^2 + ky^2) * FFT(u))

    Args:
        u: 2D array (nx, ny) or 3D (nx, ny, nz)
        dx, dy: grid spacings [m]

    Returns: nabla_h^2 u, same shape as u
    """
    nx, ny = u.shape[0], u.shape[1]
    kx, ky, k2 = wavenumber_grid(nx, ny, dx, dy)
    # Reshape k2 for broadcasting if 3D
    if u.ndim == 3:
        k2 = k2[:, :, np.newaxis]

    u_hat = np.fft.fft2(u, axes=(0, 1))
    lap_hat = -k2 * u_hat
    return np.real(np.fft.ifft2(lap_hat, axes=(0, 1)))


# ── Divergence ──────────────────────────────────────────────────────

def divergence_h(u, v, dx, dy):
    """
    Horizontal divergence: div_h = du/dx + dv/dy

    Args:
        u, v: 2D arrays (nx, ny) or 3D (nx, ny, nz)
        dx, dy: grid spacings [m]

    Returns: div_h, same shape as u
    """
    return d_dx(u, dx) + d_dy(v, dy)


# ── Pressure Poisson solver (frequency-domain closed form) ──────────

def solve_poisson(rhs, dx, dy):
    """
    Solve nabla^2 p = rhs on a periodic domain.
    Spectral closed form: p_hat = rhs_hat / (-k^2)

    This is the killer feature: O(N log N), non-iterative.
    Traditional ocean models spend 50%+ of time on iterative Poisson solvers.

    Args:
        rhs: 2D array (nx, ny) — right-hand side of nabla^2 p = rhs
        dx, dy: grid spacings [m]

    Returns: p, same shape as rhs
    """
    nx, ny = rhs.shape[0], rhs.shape[1]
    kx, ky, k2 = wavenumber_grid(nx, ny, dx, dy)

    rhs_hat = np.fft.fft2(rhs)
    # Avoid division by zero at k=0 (mean mode): set pressure reference to 0
    k2_safe = np.where(k2 == 0, 1.0, k2)
    p_hat = rhs_hat / (-k2_safe)
    # Zero the mean mode (pressure is defined up to a constant)
    p_hat[0, 0] = 0.0

    return np.real(np.fft.ifft2(p_hat))


def pressure_projection(u, v, dx, dy, rho0=1025.0):
    """
    Projection method: enforce incompressibility (div_h = 0) by solving
    for a potential and removing its gradient.

    Mathematical projection (no rho0 needed):
      1. div = du/dx + dv/dy
      2. nabla^2 phi = div        =>  phi = IFFT(div_hat / (-k^2))
      3. u = u* - d(phi)/dx,  v = v* - d(phi)/dy

    The returned 'p' is the projection potential phi. Physical pressure
    can be recovered as p_phys = rho0 * phi / dt by the caller.

    For spectral method with periodic BC, this projects onto the
    divergence-free subspace exactly (to machine precision).

    Args:
        u, v: 2D velocity arrays (nx, ny)
        dx, dy: grid spacings [m]
        rho0: reference density [kg/m^3] (unused, kept for API compat)

    Returns: (u_divfree, v_divfree, phi)
    """
    div = divergence_h(u, v, dx, dy)
    phi = solve_poisson(div, dx, dy)

    u_divfree = u - d_dx(phi, dx)
    v_divfree = v - d_dy(phi, dy)

    return u_divfree, v_divfree, phi


# ── Matrix exponential (linear step) ────────────────────────────────

def linear_step_diffusion(u, nu, dx, dy, dt):
    """
    Exact solution of du/dt = nu * laplacian(u) via matrix exponential.

    In spectral space, diffusion is diagonal:
      du_hat/dt = -nu * k^2 * u_hat
      u_hat(t+dt) = exp(-nu * k^2 * dt) * u_hat(t)

    This is UNCONDITIONALLY STABLE — no diffusion CFL constraint.
    Traditional explicit schemes need dt < dx^2/(4*nu).

    Args:
        u: 2D array (nx, ny) or 3D (nx, ny, nz)
        nu: diffusivity [m^2/s]
        dx, dy: grid spacings [m]
        dt: time step [s]

    Returns: u(t+dt), same shape as u
    """
    nx, ny = u.shape[0], u.shape[1]
    kx, ky, k2 = wavenumber_grid(nx, ny, dx, dy)
    if u.ndim == 3:
        k2 = k2[:, :, np.newaxis]

    # Exponential decay factor in frequency domain
    decay = np.exp(-nu * k2 * dt)

    u_hat = np.fft.fft2(u, axes=(0, 1))
    u_hat_new = decay * u_hat
    return np.real(np.fft.ifft2(u_hat_new, axes=(0, 1)))


def linear_step_coriolis(u, v, f0, dt):
    """
    Exact solution of the Coriolis rotation:
      du/dt = -f0 * v
      dv/dt =  f0 * u

    This is a pure rotation in (u, v) space:
      [u(t+dt)]         [cos(fd)  -sin(fd)] [u(t)]
      [v(t+dt)] =       [sin(fd)   cos(fd)] [v(t)]

    Unconditionally stable — no inertial oscillation CFL.

    Args:
        u, v: velocity arrays, any shape (2D or 3D)
        f0: Coriolis parameter [1/s]
        dt: time step [s]

    Returns: (u_new, v_new)
    """
    angle = f0 * dt
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    u_new =  cos_a * u - sin_a * v
    v_new =  sin_a * u + cos_a * v
    return u_new, v_new


# ── Dealiasing (2/3 rule) ────────────────────────────────────────────

def dealias_2_3(u_hat, axis=None):
    """
    Apply the 2/3 dealiasing rule: zero out the upper 1/3 of wavenumbers
    to prevent aliasing errors from nonlinear products.

    For an array of N modes, keep modes |k| <= N/3 (i.e., zero out |k| > N/3).

    Args:
        u_hat: Fourier coefficients (from fft)
        axis: axis along which to dealias. If None, dealias all spatial axes.

    Returns: dealiased u_hat
    """
    if axis is not None:
        n = u_hat.shape[axis]
        cutoff = n // 3
        # Create a mask: 1 for kept modes, 0 for truncated
        mask = np.zeros(n, dtype=u_hat.dtype)
        mask[:cutoff] = 1.0
        mask[-cutoff:] = 1.0  # negative frequencies
        shape = [1] * u_hat.ndim
        shape[axis] = n
        mask = mask.reshape(shape)
        return u_hat * mask
    else:
        result = u_hat.copy()
        for ax in range(u_hat.ndim):
            result = dealias_2_3(result, axis=ax)
        return result


# ── Pseudospectral advection ────────────────────────────────────────

def advection_flux_form(u, v, dx, dy, dealias=True):
    """
    Compute horizontal advection in flux form (conservative):
      -[d(uu)/dx + d(uv)/dy]  for u-momentum
      -[d(uv)/dx + d(vv)/dy]  for v-momentum

    Flux form ensures discrete conservation: fluxes across cell boundaries
    cancel exactly, so total momentum changes only from boundary fluxes.

    Args:
        u, v: 2D velocity arrays (nx, ny)
        dx, dy: grid spacings [m]
        dealias: apply 2/3 rule to nonlinear products

    Returns: (adv_u, adv_v) - advection tendencies for u and v
    """
    # Compute products in physical space
    fluxes = [u * u, u * v, v * v]

    if dealias:
        # Transform to spectral, dealias, transform back (in-place per flux)
        for i in range(len(fluxes)):
            f_hat = np.fft.fft2(fluxes[i])
            f_hat = dealias_2_3(f_hat)
            fluxes[i] = np.real(np.fft.ifft2(f_hat))

    uu, uv, vv = fluxes

    # Derivatives in spectral space
    adv_u = -(d_dx(uu, dx) + d_dy(uv, dy))
    adv_v = -(d_dx(uv, dx) + d_dy(vv, dy))
    return adv_u, adv_v


def advection_scalar(T, u, v, dx, dy, dealias=True):
    """
    Compute scalar advection in flux form (conservative):
      -[d(uT)/dx + d(vT)/dy]

    Used for temperature and salinity equations.

    Args:
        T: scalar field (nx, ny) or (nx, ny, nz)
        u, v: velocity fields, same shape as T
        dx, dy: grid spacings [m]
        dealias: apply 2/3 rule

    Returns: advection tendency for T
    """
    fluxes = [u * T, v * T]

    if dealias:
        for i in range(len(fluxes)):
            f_hat = np.fft.fft2(fluxes[i], axes=(0, 1))
            f_hat = dealias_2_3(f_hat, axis=0)
            f_hat = dealias_2_3(f_hat, axis=1)
            fluxes[i] = np.real(np.fft.ifft2(f_hat, axes=(0, 1)))

    uT, vT = fluxes
    return -(d_dx(uT, dx) + d_dy(vT, dy))


# ── Vertical derivatives (non-uniform z-grid, tridiagonal) ──────────

def d_dz(u, z_levels):
    """
    Vertical first derivative on non-uniform z-grid.
    Uses centered differences for interior, one-sided at boundaries.

    Args:
        u: array with vertical axis as last axis, shape (..., nz)
        z_levels: 1D array of z coordinates [m], negative downward

    Returns: du/dz, same shape as u
    """
    nz = u.shape[-1]
    dz = np.diff(z_levels)  # negative (z decreases downward)
    du = np.zeros_like(u)

    # Interior: centered difference with non-uniform spacing
    for k in range(1, nz - 1):
        dz_up = z_levels[k] - z_levels[k - 1]   # usually negative
        dz_dn = z_levels[k + 1] - z_levels[k]   # usually negative
        du[..., k] = (u[..., k + 1] - u[..., k - 1]) / (dz_up + dz_dn)

    # Boundary: one-sided
    du[..., 0] = (u[..., 1] - u[..., 0]) / (z_levels[1] - z_levels[0])
    du[..., -1] = (u[..., -1] - u[..., -2]) / (z_levels[-1] - z_levels[-2])

    return du


def d2_dz2(u, z_levels):
    """
    Vertical second derivative on non-uniform z-grid.
    Uses second-order accurate formula for non-uniform spacing.

    Args:
        u: array with vertical axis as last axis, shape (..., nz)
        z_levels: 1D array of z coordinates [m]

    Returns: d2u/dz2, same shape as u
    """
    nz = u.shape[-1]
    du = np.zeros_like(u)

    for k in range(1, nz - 1):
        h1 = z_levels[k] - z_levels[k - 1]  # spacing to level below (going up)
        h2 = z_levels[k + 1] - z_levels[k]  # spacing to level above (going up)
        # Actually z is negative downward, so h1, h2 < 0
        # Standard non-uniform second derivative:
        # d2u/dz2 = 2*(u[k-1] - u[k]*((h1+h2)/h2) + u[k+1]*(h1/h2)) / (h1*(h1+h2))
        # But let's use absolute spacing for clarity
        hm = abs(z_levels[k - 1] - z_levels[k])      # distance to k-1
        hp = abs(z_levels[k + 1] - z_levels[k])      # distance to k+1

        du[..., k] = (u[..., k + 1] * hm + u[..., k - 1] * hp
                      - u[..., k] * (hm + hp)) / (hm * hp * (hm + hp) / 2)

    # Boundaries: simple second-order one-sided
    h0 = abs(z_levels[1] - z_levels[0])
    h1 = abs(z_levels[2] - z_levels[0])
    du[..., 0] = (u[..., 2] - 2 * u[..., 1] + u[..., 0]) / (h0 * h0)

    h0 = abs(z_levels[-1] - z_levels[-2])
    h1 = abs(z_levels[-2] - z_levels[-3])
    du[..., -1] = (u[..., -3] - 2 * u[..., -2] + u[..., -1]) / (h0 * h0)

    return du


# ── Utility: spectral truncation (low-pass filter) ──────────────────

def spectral_truncate(u, keep_fraction, dx=None, dy=None):
    """
    Low-pass filter: keep only the lowest `keep_fraction` of wavenumbers.

    Useful for:
      - Removing numerical noise
      - Testing resolution sensitivity
      - Initializing smooth fields

    Args:
        u: 2D array (nx, ny)
        keep_fraction: fraction of modes to keep (0 < keep_fraction <= 1)

    Returns: filtered u
    """
    nx, ny = u.shape
    u_hat = np.fft.fft2(u)

    nx_keep = max(1, int(nx * keep_fraction))
    ny_keep = max(1, int(ny * keep_fraction))

    mask = np.zeros_like(u_hat)
    mask[:nx_keep, :ny_keep] = 1.0
    mask[-nx_keep:, :ny_keep] = 1.0
    mask[:nx_keep, -ny_keep:] = 1.0
    mask[-nx_keep:, -ny_keep:] = 1.0

    return np.real(np.fft.ifft2(u_hat * mask))
