"""
Surface Forcing Module — wind stress and heat flux fields.

Generates 2D forcing fields for the subtropical gyre domain:
  - Wind stress: meridional profile with easterlies (south) and
    westerlies (north), matching the Stommel/Munk gyre paradigm.
  - Heat flux: meridional gradient (warming south, cooling north).

Supports both idealized analytic profiles and user-provided arrays.

All fields are (nx, ny) shaped, matching the solver axis convention
(axis 0 = zonal/lon, axis 1 = meridional/lat).
"""
import numpy as np

from config import RHO_0
from dataclasses import dataclass
from typing import Optional


@dataclass
class Forcing:
    """Spatially-varying surface forcing fields.

    Holds 2D (nx, ny) arrays for wind stress and heat flux. When passed to
    compute_momentum_tendency / compute_tracer_tendency, these override the
    scalar defaults on PhysicsConfig. All fields default to None; any field
    left None falls back to the corresponding physics scalar.
    """
    tau_x: Optional[np.ndarray] = None   # (nx, ny) zonal wind stress [N/m^2]
    tau_y: Optional[np.ndarray] = None   # (nx, ny) meridional wind stress [N/m^2]
    Q_heat: Optional[np.ndarray] = None  # (nx, ny) surface heat flux [W/m^2]



def taper_weight_1d(ny, taper_cells):
    """Raised-cosine y-edge taper weight (multiplicative mask).

    Returns a ``(ny,)`` weight that is 1 in the interior and ramps smoothly
    to 0 over ``taper_cells`` cells at each meridional edge.  Used to zero
    forcing at the periodic FFT seam in both 1D profiles and 2D fields.
    """
    taper_cells = int(taper_cells)
    weight = np.ones(int(ny), dtype=np.float64)
    if taper_cells <= 0:
        return weight
    taper_cells = min(taper_cells, int(ny) // 2)
    taper = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, taper_cells)))
    weight[:taper_cells] = taper
    weight[-taper_cells:] = taper[::-1]
    return weight


def _taper_y(profile, ny, taper_cells):
    """Force a 1D y-profile to zero smoothly at both meridional edges.

    The solver differentiates horizontally with periodic (FFT) spectral
    derivatives in both x and y.  A wind/heat profile that is non-zero at
    the southern (y=0) or northern (y=ny-1) edge creates a step
    discontinuity at the periodic meridional seam, which injects spurious
    grid-scale forcing energy at those boundaries (observed as a persistent
    NW-corner heat pump).  This applies a raised-cosine taper over
    ``taper_cells`` grid cells at each edge so the profile reaches zero at
    both ends and is continuous across the periodic seam.

    Args:
        profile: (ny,) 1D meridional profile before tapering.
        ny: number of meridional grid cells.
        taper_cells: number of edge cells over which the profile is
            smoothly ramped to zero at each boundary.

    Returns:
        (ny,) tapered profile.
    """
    return np.asarray(profile, dtype=np.float64) * taper_weight_1d(ny, taper_cells)


def taper_2d_y(field, ny, taper_cells):
    """Apply the meridional y-edge taper to a 2D ``(nx, ny)`` forcing field.

    The 1D taper weight is broadcast along axis 1 so the whole field reaches
    zero at both y-edges, keeping it continuous across the periodic seam.
    """
    field = np.asarray(field, dtype=np.float64)
    return field * taper_weight_1d(ny, taper_cells)[None, :]

def wind_stress_gyre(grid, tau0=0.1):
    """Subtropical gyre wind stress (classic Stommel profile).

    tau_x(y) = -tau0 * cos(pi * y' / Ly)

    where y' is distance from southern boundary. This gives:
      - Easterlies (tau_x < 0) in the southern half (trade winds)
      - Westerlies (tau_x > 0) in the northern half
      - Zero stress at the center

    Args:
        grid: OceanGrid
        tau0: peak wind stress magnitude [N/m^2], default 0.1

    Returns:
        tau_x, tau_y: (nx, ny) wind stress fields [N/m^2]
    """
    ny = grid.ny
    taper_cells = getattr(grid, "forcing_taper_cells", 8)
    # Normalized meridional coordinate: 0 at south, 1 at north
    y_frac = np.arange(ny, dtype=np.float64) / (ny - 1)

    tau_x_profile = -tau0 * np.cos(np.pi * y_frac)  # (ny,)
    # Taper to zero at both y-edges so the profile is continuous across the
    # periodic meridional seam (avoids a spectral step-discontinuity artifact).
    tau_x_profile = _taper_y(tau_x_profile, ny, taper_cells)
    tau_x = np.broadcast_to(tau_x_profile[None, :], (grid.nx, ny)).copy()
    tau_y = np.zeros((grid.nx, ny))
    return tau_x, tau_y


def heat_flux_meridional(grid, Q0=50.0):
    """Meridional heat flux gradient.

    Q(y) = -Q0 * (2*y' - 1)

    where y' is normalized meridional coordinate (0=south, 1=north).
    This gives:
      - Warming (Q > 0) in the south
      - Cooling (Q < 0) in the north
      - Zero at the center

    Args:
        grid: OceanGrid
        Q0: peak heat flux magnitude [W/m^2], default 50

    Returns:
        Q_heat: (nx, ny) heat flux field [W/m^2]
    """
    ny = grid.ny
    taper_cells = getattr(grid, "forcing_taper_cells", 8)
    y_frac = np.arange(ny, dtype=np.float64) / (ny - 1)

    q_profile = -Q0 * (2.0 * y_frac - 1.0)  # (ny,)
    q_profile = _taper_y(q_profile, ny, taper_cells)
    return np.broadcast_to(q_profile[None, :], (grid.nx, ny)).copy()


def wind_stress_seasonal(grid, tau0=0.1, season_frac=0.0):
    """Seasonally modulated gyre wind stress.

    Adds a sinusoidal seasonal modulation to the gyre pattern:
    tau_x(y, t) = -tau0 * cos(pi*y') * (1 + amp * cos(2*pi*season_frac))

    Args:
        grid: OceanGrid
        tau0: peak wind stress [N/m^2]
        season_frac: seasonal phase (0=winter solstice, 0.5=summer)

    Returns:
        tau_x, tau_y: (nx, ny) wind stress fields [N/m^2]
    """
    ny = grid.ny
    taper_cells = getattr(grid, "forcing_taper_cells", 8)
    y_frac = np.arange(ny, dtype=np.float64) / (ny - 1)

    # Seasonal amplitude: stronger winds in winter (season_frac=0)
    seasonal_amp = 0.3 * np.cos(2.0 * np.pi * season_frac)
    tau_x_profile = -tau0 * np.cos(np.pi * y_frac) * (1.0 + seasonal_amp)
    tau_x_profile = _taper_y(tau_x_profile, ny, taper_cells)
    tau_x = np.broadcast_to(tau_x_profile[None, :], (grid.nx, ny)).copy()
    tau_y = np.zeros((grid.nx, ny))
    return tau_x, tau_y


def ekman_pumping(grid, tau_x, tau_y):
    """Compute Ekman pumping velocity from wind stress curl.

    w_ek = curl(tau) / (rho_0 * f)

    where curl(tau) = d(tau_y)/dx - d(tau_x)/dy

    This drives the interior Sverdrup transport in a stratified ocean.

    Args:
        grid: OceanGrid
        tau_x, tau_y: (nx, ny) wind stress fields [N/m^2]

    Returns:
        w_ek: (nx, ny) Ekman pumping velocity [m/s]
    """
    # Central finite differences (wind stress profiles are non-periodic;
    # spectral derivatives would introduce Gibbs-like boundary artifacts)
    # Axis 0 = x (zonal), Axis 1 = y (meridional)
    dtau_y_dx = np.zeros_like(tau_y)
    dtau_x_dy = np.zeros_like(tau_x)
    dtau_y_dx[1:-1, :] = (tau_y[2:, :] - tau_y[:-2, :]) / (2.0 * grid.dx)
    dtau_x_dy[:, 1:-1] = (tau_x[:, 2:] - tau_x[:, :-2]) / (2.0 * grid.dy)
    curl_tau = dtau_y_dx - dtau_x_dy

    # Avoid division by zero at equator (f=0), though our domain is mid-lat
    f = grid.f
    f_safe = np.where(np.abs(f) > 1e-10, f, 1e-10)
    w_ek = curl_tau / (RHO_0 * f_safe)
    return w_ek


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import DEFAULT_CONFIG
    from grid import make_grid

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)

    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)
    w_ek = ekman_pumping(grid, tau_x, tau_y)

    print("=== Surface Forcing ===")
    print(f"Grid: {grid.nx}x{grid.ny}")
    print(f"Lat range: {grid.lat[0]:.1f} - {grid.lat[-1]:.1f} N")
    print()
    print(f"Wind stress tau_x:")
    print(f"  range: {tau_x.min():.4f} to {tau_x.max():.4f} N/m^2")
    print(f"  south (trade): {tau_x[:, 0].mean():.4f} N/m^2 (easterly)")
    print(f"  center:         {tau_x[:, grid.ny//2].mean():.4f} N/m^2")
    print(f"  north (wester): {tau_x[:, -1].mean():.4f} N/m^2 (westerly)")
    print()
    print(f"Heat flux Q:")
    print(f"  range: {Q_heat.min():.1f} to {Q_heat.max():.1f} W/m^2")
    print(f"  south: {Q_heat[:, 0].mean():.1f} W/m^2 (warming)")
    print(f"  north: {Q_heat[:, -1].mean():.1f} W/m^2 (cooling)")
    print()
    print(f"Ekman pumping w_ek:")
    print(f"  range: {w_ek.min():.2e} to {w_ek.max():.2e} m/s")
    print(f"  max |w_ek|: {np.abs(w_ek).max():.2e} m/s")
    print(f"  (typical OGCM values: 10-50 m/year = 3e-7 to 1.5e-6 m/s)")
