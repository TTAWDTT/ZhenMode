"""
Ocean Solver — Configuration
Hydrostatic primitive equations, spectral method, JAX.

All confirmed design decisions in one place.
"""
from dataclasses import dataclass, field
import numpy as np


# ── Physical constants ──────────────────────────────────────────────
G_EARTH     = 9.81          # m/s²
RHO_0       = 1025.0        # kg/m³  reference seawater density
ALPHA_T     = 2.0e-4        # 1/°C   thermal expansion coefficient
BETA_S      = 7.6e-4        # 1/psu  haline contraction coefficient
C_P         = 3992.0        # J/(kg·°C) specific heat of seawater
R_EARTH     = 6371.0e3      # m      Earth radius
OMEGA       = 7.2921e-5     # rad/s  Earth angular velocity


@dataclass(frozen=True)
class GridConfig:
    """Horizontal & vertical grid specification."""
    # ── Horizontal ──
    nx: int = 128                      # zonal grid points
    ny: int = 128                      # meridional grid points
    # Domain center (North Pacific subtropical region, mostly open ocean)
    lat_center: float = 35.0           # °N
    lon_center: float = 150.0          # °E
    # Span computed from nx × ETOPO resolution (0.1°)
    etopo_resolution: float = 0.1      # degrees per grid cell
    # Physical dx/dy computed at lat_center (f-plane / beta-plane approximation)
    # dx = R_EARTH * cos(lat_center) * dlon
    # dy = R_EARTH * dlat

    # ── Vertical (z-level, upper-ocean focused) ──
    z_levels: tuple = (
        0, -5, -15, -30, -50, -75, -100,
        -150, -200, -300, -500, -1000, -2000, -4000,
    )
    nz: int = 14                       # number of vertical levels

    @property
    def dlon(self) -> float:
        return self.etopo_resolution

    @property
    def dlat(self) -> float:
        return self.etopo_resolution

    @property
    def dx(self) -> float:
        """Zonal grid spacing [m] at domain center."""
        return R_EARTH * np.cos(np.radians(self.lat_center)) * np.radians(self.dlon)

    @property
    def dy(self) -> float:
        """Meridional grid spacing [m]."""
        return R_EARTH * np.radians(self.dlat)

    @property
    def lon_span(self) -> float:
        """Total zonal span [degrees]."""
        return self.nx * self.dlon

    @property
    def lat_span(self) -> float:
        """Total meridional span [degrees]."""
        return self.ny * self.dlat

    @property
    def lon_bounds(self) -> tuple:
        half = self.lon_span / 2
        return (self.lon_center - half, self.lon_center + half)

    @property
    def lat_bounds(self) -> tuple:
        half = self.lat_span / 2
        return (self.lat_center - half, self.lat_center + half)

    @property
    def f0(self) -> float:
        """Coriolis parameter at domain center [s⁻¹] (f-plane)."""
        return 2 * OMEGA * np.sin(np.radians(self.lat_center))

    @property
    def beta(self) -> float:
        """Meridional gradient of Coriolis [m⁻¹s⁻¹] (beta-plane)."""
        return 2 * OMEGA * np.cos(np.radians(self.lat_center)) / R_EARTH


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

    # ── Smagorinsky subgrid closure ──
    smag_cs: float = 0.0       # Smagorinsky constant (0 = disabled)

    # ── Equation of state ──
    T_ref: float = 15.0        # °C    reference temperature
    S_ref: float = 35.0        # psu   reference salinity
    eos_type: str = 'linear'   # 'linear' or 'unesco'

    # ── Surface forcing (defaults, can be overridden at runtime) ──
    tau_x: float = 0.0         # N/m²  zonal wind stress
    tau_y: float = 0.0         # N/m²  meridional wind stress
    Q_heat: float = 0.0        # W/m²  surface heat flux

    # ── Bottom friction ──
    cd: float = 2.5e-3         # quadratic drag coefficient
    r_bot: float = 1.0e-3      # linear bottom friction coefficient
    bottom_friction: str = 'linear'  # 'linear' or 'quadratic'


@dataclass(frozen=True)
class TimeConfig:
    """Time integration — IMEX (matrix exponential + explicit)."""
    dt: float = 600.0          # s      time step (CFL ~ 1000s for dx~9km, u~1m/s)
    dt_output: float = 3600.0  # s      output interval
    t_total: float = 86400.0   # s      total integration time (1 day default)

    # IMEX parameters
    n_substeps_lin: int = 1    # linear sub-steps per nonlinear step


@dataclass(frozen=True)
class Config:
    """Master configuration — single source of truth."""
    grid: GridConfig = field(default_factory=GridConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    time: TimeConfig = field(default_factory=TimeConfig)

    # ── Data paths ──
    bathymetry_file: str = r"C:\Users\zhen.luo\Desktop\ETOPO_2022_v1_r3600x1800_surface.nc"

    # ── Framework ──
    framework: str = "jax"     # "jax" or "numpy" (for testing)

    # ── Equation level ──
    equation_level: str = "hydrostatic"  # hydrostatic primitive equations

    # ── Dealiasing ──
    dealias: bool = True       # 2/3 rule for pseudospectral products


# Default singleton
DEFAULT_CONFIG = Config()


if __name__ == "__main__":
    cfg = DEFAULT_CONFIG
    g = cfg.grid
    print("=== Ocean Solver Configuration ===")
    print(f"Equation level: {cfg.equation_level}")
    print(f"Framework: {cfg.framework}")
    print(f"Bathymetry: {cfg.bathymetry_file}")
    print()
    print("--- Grid ---")
    print(f"Horizontal: {g.nx} x {g.ny} (lon x lat)")
    print(f"Vertical:   {g.nz} levels")
    print(f"Z levels:   {g.z_levels}")
    print(f"Domain:     lon [{g.lon_bounds[0]:.1f}, {g.lon_bounds[1]:.1f}E]")
    print(f"            lat [{g.lat_bounds[0]:.1f}, {g.lat_bounds[1]:.1f}N]")
    print(f"dx = {g.dx:.1f} m,  dy = {g.dy:.1f} m")
    print(f"f0 = {g.f0:.5e} /s")
    print(f"beta  = {g.beta:.5e} /(m*s)")
    print()
    p = cfg.physics
    print("--- Physics ---")
    print(f"nu_h = {p.nu_h} m^2/s,  nu_v = {p.nu_v} m^2/s")
    print(f"kappa_h = {p.kappa_h} m^2/s,  kappa_v = {p.kappa_v} m^2/s,  kappa_conv = {p.kappa_conv} m^2/s")
    print(f"EOS: rho = rho0[1 - alpha*(T-T0) + beta*(S-S0)]")
    print(f"  T0 = {p.T_ref} C,  S0 = {p.S_ref} psu")
    print()
    t = cfg.time
    print("--- Time ---")
    print(f"dt = {t.dt} s,  t_total = {t.t_total} s ({t.t_total/3600:.0f} h)")
