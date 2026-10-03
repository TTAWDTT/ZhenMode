"""
Ocean Solver — configuration.

Physical constants, the global lat-lon grid spec (GlobalGridConfig), the
physics parameterizations (PhysicsConfig) and the shared data paths
(Config / DEFAULT_CONFIG).
"""

import os
from dataclasses import dataclass, field

import numpy as np

from ocean_solver.provenance.locations import source_root

# ── Physical constants ──────────────────────────────────────────────
G_EARTH     = 9.81          # m/s²
RHO_0       = 1025.0        # kg/m³  reference seawater density
ALPHA_T     = 2.0e-4        # 1/°C   thermal expansion coefficient
BETA_S      = 7.6e-4        # 1/psu  haline contraction coefficient
C_P         = 3992.0        # J/(kg·°C) specific heat of seawater
R_EARTH     = 6371.0e3      # m      Earth radius
OMEGA       = 7.2921e-5     # rad/s  Earth angular velocity


@dataclass(frozen=True)
class GlobalGridConfig:
    """Global lat-lon grid specification (finite-difference solver).

    A true global grid: lon spans 0..360 (periodic), lat spans the globe with
    spherical metric factors (dx = R*cos(lat)*dlon varies with latitude).
    Built by grid.make_global_grid and consumed by jax_solver_global.
    """
    # ── Horizontal ──
    nx: int = 360                      # zonal grid points (lon, periodic)
    ny: int = 170                      # meridional grid points (lat, ±85°)
    resolution: float = 1.0            # degrees per grid cell
    lat_max: float = 85.0              # poleward lat limit (polar cap below)

    # ── Vertical ──
    z_levels: tuple = (
        0, -5, -15, -30, -50, -75, -100,
        -150, -200, -300, -500, -1000, -2000, -4000,
    )
    nz: int = 14

    @property
    def dlon(self) -> float:
        return self.resolution

    @property
    def dlat(self) -> float:
        return self.resolution

    @property
    def lon(self) -> np.ndarray:
        """Longitude centers [degrees E], 0..360-dlon, periodic."""
        return self.resolution * (0.5 + np.arange(self.nx))

    @property
    def lat(self) -> np.ndarray:
        """Latitude centers [degrees N], symmetric ±, excluding polar cap."""
        return np.linspace(-self.lat_max, self.lat_max, self.ny)

    @property
    def lat_2d(self) -> np.ndarray:
        """2D latitude field (nx, ny) for metric computation."""
        return np.broadcast_to(self.lat[None, :], (self.nx, self.ny))

    @property
    def cos_lat(self) -> np.ndarray:
        """cos(lat) per meridional row (ny,) — spherical metric factor."""
        return np.cos(np.radians(self.lat))

    @property
    def dx_2d(self) -> np.ndarray:
        """Zonal grid spacing [m], varies with latitude: R*cos(lat)*dlon."""
        return R_EARTH * np.radians(self.dlon) * self.cos_lat  # (ny,)

    @property
    def dy(self) -> float:
        """Meridional grid spacing [m] (constant on a lat-lon grid)."""
        return R_EARTH * np.radians(self.dlat)

@dataclass(frozen=True)
class PhysicsConfig:
    """Physics parameterization — hydrostatic primitive equations."""
    # ── Turbulence closure ──
    nu_h: float = 100.0        # m²/s  horizontal eddy viscosity (background)
    nu_v: float = 1.0e-4       # m²/s  vertical eddy viscosity
    kappa_h: float = 100.0     # m²/s  horizontal diffusivity (T, S)
    kappa_v: float = 1.0e-5    # m²/s  vertical diffusivity (T, S)
    # Convective adjustment: enhanced vertical diffusivity applied to
    # statically-unstable columns (heavier over lighter). Standard OGCM
    # remedy for the columnar warm-water accumulation that surface-only
    # restoring cannot remove. Chosen below the explicit RK2 vertical
    # CFL limit for the thin (5 m) top layer at dt=300 s (~0.083 m^2/s).
    kappa_conv: float = 0.05     # m²/s  convective vertical diffusivity

    # ── Scale-selective (biharmonic) viscosity/diffusivity ──
    # Damp grid-scale baroclinic eddy modes (∇⁴, ∝ k⁴) far more than
    # the large-scale flow, killing the eddy-instability blowup while
    # leaving basin-scale physics intact. 0 = disabled (Laplacian only).
    nu_bi: float = 1.0e12      # m⁴/s  biharmonic horizontal viscosity (calibrated 2026-08-21)
    kappa_bi: float = 1.0e12   # m⁴/s  biharmonic horizontal diffusivity

    # ── Gent-McWilliams eddy closure ──
    # Represents unresolved baroclinic eddies as an advective bolus transport
    # that flattens isopycnal slopes, releasing baroclinic available potential
    # energy. Required at coarse (1°) resolution where the baroclinic Rossby
    # radius (~30-50km) is sub-grid; near-inactive at eddy-resolving resolution.
    # 0 = disabled (default; the closure is enabled only on coarse global runs).
    kappa_gm: float = 0.0       # m²/s  GM eddy diffusivity (bolus transport)
    gm_slope_max: float = 0.01  # dimensionless isopycnal-slope limiter
    # ── Redi isopycnal mixing (dissipative counterpart to GM) ──
    # Diffuses tracers ALONG sloped isopycnals. In Griffies skew-flux residual
    # form only the slope-driven terms are applied (the horizontal-gradient
    # part is absorbed into the background kappa_h*lap handled by the linear
    # step). The vertical term -κ_redi|S|²∂zC provides the diapycnal-style
    # APE sink that pure (advective) GM bolus lacks — it is what arrests the
    # w* steepening feedback. Typically κ_redi = κ_gm. 0 = disabled (default).
    kappa_redi: float = 0.0     # m²/s  Redi isopycnal diffusivity; 0 = off

    # ── Equation of state (linear only; see _density_anomaly) ──
    T_ref: float = 15.0        # °C    reference temperature
    S_ref: float = 35.0        # psu   reference salinity

    # ── Surface forcing (defaults, can be overridden at runtime) ──
    tau_x: float = 0.0         # N/m²  zonal wind stress
    tau_y: float = 0.0         # N/m²  meridional wind stress
    Q_heat: float = 0.0        # W/m²  surface heat flux

    # ── Bottom friction ──
    cd: float = 2.5e-3         # quadratic drag coefficient
    r_bot: float = 1.0e-3      # linear bottom friction coefficient
    bottom_friction: str = 'linear'  # 'linear' or 'quadratic'


_REPO_ROOT = str(source_root(__file__).parent)

# Conventional location for the ETOPO2022 0.1 deg global relief file
# (3600x1800). A pre-extracted "<file>.npz" twin (z/lon/lat) is accepted on
# nodes without netCDF4/HDF support; see grid._read_etopo_global.
ETOPO_FILENAME = "ETOPO_2022_v1_r3600x1800_surface.nc"
BATHYMETRY_ENV_VAR = "OCEAN_SOLVER_BATHYMETRY"


def _default_bathymetry_file() -> str:
    """Locate the ETOPO2022 bathymetry file without machine-specific paths.

    Order: the ``OCEAN_SOLVER_BATHYMETRY`` environment variable, then
    ``<repo>/data/``, then the repository root. The returned path is not
    guaranteed to exist — grid._read_etopo_global raises a clear error naming
    this env var when the file is missing (the test suite uses that to skip).
    """
    env = os.environ.get(BATHYMETRY_ENV_VAR)
    if env:
        return env
    for root in (_REPO_ROOT, os.path.join(_REPO_ROOT, "data")):
        candidate = os.path.join(root, ETOPO_FILENAME)
        if os.path.exists(candidate) or os.path.exists(candidate + ".npz"):
            return candidate
    return os.path.join(_REPO_ROOT, "data", ETOPO_FILENAME)


@dataclass(frozen=True)
class Config:
    """Shared defaults: physics + data paths.

    The horizontal grid is per run (GlobalGridConfig, driven by --resolution /
    --ny / --lat-max in run_long_integration_global), so it is not held here.
    """
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    # ── Data paths ──
    # Resolution order: $OCEAN_SOLVER_BATHYMETRY, then <repo>/data/, then
    # <repo>/. Never hard-code a machine-specific absolute path here.
    bathymetry_file: str = field(default_factory=_default_bathymetry_file)


# Default singleton
DEFAULT_CONFIG = Config()




if __name__ == "__main__":
    cfg = DEFAULT_CONFIG
    g = GlobalGridConfig()
    print("=== Ocean Solver configuration ===")
    print("Framework:  JAX")
    print("Equations:  hydrostatic primitive equations (global FD)")
    print(f"Bathymetry: {cfg.bathymetry_file}")
    print()
    print("--- Global grid defaults ---")
    print(f"Horizontal: {g.nx} x {g.ny} (lon x lat) at {g.resolution} deg")
    print(f"Vertical:   {g.nz} levels")
    print(f"Z levels:   {g.z_levels}")
    print(f"Latitude:   [-{g.lat_max}, {g.lat_max}] deg (polar cap excluded)")
    print()
    p_ = cfg.physics
    print("--- Physics ---")
    print(f"nu_h = {p_.nu_h} m^2/s,  nu_v = {p_.nu_v} m^2/s")
    print(f"kappa_h = {p_.kappa_h} m^2/s,  kappa_v = {p_.kappa_v} m^2/s,  "
          f"kappa_conv = {p_.kappa_conv} m^2/s")
    print(f"kappa_gm = {p_.kappa_gm} m^2/s,  kappa_redi = {p_.kappa_redi} m^2/s")
    print("EOS: rho = rho0[1 - alpha*(T-T0) + beta*(S-S0)]")
    print(f"  T0 = {p_.T_ref} C,  S0 = {p_.S_ref} psu")
