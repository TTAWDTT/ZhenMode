"""Canonical mms definitions; legacy operations unchanged."""
from config import OMEGA, R_EARTH
from ocean_solver.fd.backend import jnp, np
from ocean_solver.fd.geometry import make_fd_params
from ocean_solver.fd.horizontal import _d_dx, _d_dy, _divergence_h, _laplacian_h


def _mms_grid(nx, ny, res, lat_max=75.0):
    """Metric-only grid + FDParams for the MMS operators (no ETOPO needed).

    `dy` must come from the SAME lat span that defines the rows: deriving it
    from the lon resolution instead left the d/dy check 15.4% off, which the
    RMS/max normalization hid inside a generous threshold.
    """
    lat = np.linspace(-lat_max, lat_max, ny)       # avoid poles (cos->0)
    lon = res * (0.5 + np.arange(nx))
    lon_2d, lat_2d = np.meshgrid(lon, lat, indexing='ij')
    cos_lat = np.cos(np.radians(lat))
    z = np.array([0, -10, -30, -60], dtype=np.float64)

    class _G:
        pass

    g = _G()
    g.dx_2d = np.broadcast_to(R_EARTH * np.radians(res) * cos_lat, (nx, ny)).copy()
    g.dy = R_EARTH * np.radians(2.0 * lat_max / (ny - 1))
    g.cos_lat = cos_lat
    g.f = 2.0 * OMEGA * np.sin(np.radians(lat_2d))
    g.wet_mask = np.ones((nx, ny))                 # all-ocean for MMS
    g.wet_mask_3d = g.wet_mask[..., None]
    g.z = z
    g.dz = np.abs(np.diff(z))
    g.nz = len(z)
    g.nx, g.ny = nx, ny
    return make_fd_params(g), lon_2d, lat_2d, cos_lat

def _mms_run():
    """Run MMS checks on a small synthetic global grid (no ETOPO needed).

    Returns a dict of {test_name: (error, passes_threshold)}.
    Uses a 10° grid so the FD truncation is visible, and analytic
    trigonometric fields.
    """
    nx, ny, res = 36, 14, 360.0 / 36   # 10° resolution
    p, lon_2d, lat_2d, cos_lat = _mms_grid(nx, ny, res)
    nz = p.nz

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
    # Measured on INTERIOR rows: the wall rows use the mirror-ghost no-flux BC,
    # which deliberately differs from the infinite-domain analytic derivative
    # (see _mms_convergence). At 10° with 2 lat-oscillations each wavelength is
    # ~6.5 points, so the 2nd-order truncation is ~1.5% here.
    du_dy_fd = np.array(_d_dy(jnp.array(u3d), p))[:, :, 0]
    du_dy_an = (-np.sin(m * lon_rad) * n * np.sin(n * lat_rad)) / R_EARTH
    sl = slice(2, ny - 2)
    err = (np.sqrt(np.mean((du_dy_fd[:, sl] - du_dy_an[:, sl]) ** 2))
           / (np.abs(du_dy_an[:, sl]).max() + 1e-30))
    results['d_dy_rel_L2'] = (float(err), err < 0.03)

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
    def ddy_error(ny):
        nx, res = 36, 360.0 / 36
        p, lon_2d, lat_2d, _ = _mms_grid(nx, ny, res)
        lat_rad = np.radians(lat_2d)
        lon_rad = np.radians(lon_2d)
        n = 2.0
        u2d = np.sin(2.0 * lon_rad) * np.cos(n * lat_rad)
        u3d = np.broadcast_to(u2d[..., None], (nx, ny, p.nz)).copy()
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
