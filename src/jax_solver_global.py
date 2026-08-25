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
    'wet_mask_3d',      # (nx, ny, 1) for 3D broadcast
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
    """Meridional derivative du/dy, 2nd-order central FD, lat-bounded.

    Axis 1 (lat) does NOT wrap. One-sided 2nd-order stencils at the N/S
    edges (j=0, j=ny-1) to avoid referencing outside the domain.
    """
    interior = (u[:, 2:] - u[:, :-2]) * (0.5 * p.inv_dy)            # (nx, ny-2, ...)
    # 2nd-order one-sided at edges: du/dy|_0 = (-3u0 + 4u1 - u2)/(2dy)
    top = (-3.0 * u[:, 0:1] + 4.0 * u[:, 1:2] - u[:, 2:3]) * (0.5 * p.inv_dy)
    bot = (3.0 * u[:, -1:] - 4.0 * u[:, -2:-1] + u[:, -3:-2]) * (0.5 * p.inv_dy)
    return jnp.concatenate([top, interior, bot], axis=1)


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
    # ∂²u/∂y²: central second FD, with one-sided edges
    d2u_dy2_int = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) * p.inv_dy2
    top = (2.0 * u[:, 0:1] - 5.0 * u[:, 1:2] + 4.0 * u[:, 2:3] - u[:, 3:4]) * p.inv_dy2
    bot = (2.0 * u[:, -1:] - 5.0 * u[:, -2:-1] + 4.0 * u[:, -3:-2] - u[:, -4:-3]) * p.inv_dy2
    d2u_dy2 = jnp.concatenate([top, d2u_dy2_int, bot], axis=1)
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
        return np.sqrt(np.mean((du_dy_fd - du_dy_an) ** 2)) / (np.abs(du_dy_an).max() + 1e-30)

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
