"""
Model State Container — packs all prognostic variables for the hydrostatic primitive equations.

Prognostic variables (time-integrated):
  u, v  — horizontal velocity components        (nx, ny, nz)
  T     — potential temperature                  (nx, ny, nz)
  S     — salinity                               (nx, ny, nz)
  eta   — sea surface height                     (nx, ny)

Diagnostic variables (computed from prognostics):
  w     — vertical velocity                      (nx, ny, nz)
  rho   — density via equation of state          (nx, ny, nz)
  p     — hydrostatic pressure                   (nx, ny, nz)

Axis convention (consistent with spectral_ops.py):
  axis 0 = x (zonal/lon), axis 1 = y (meridional/lat), axis 2 = z (vertical)
  Vertical axis is always LAST so d_dz / d2_dz2 work directly.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ModelState:
    """Prognostic state for the hydrostatic primitive equations."""

    # ── Prognostic variables ──
    u: np.ndarray     # (nx, ny, nz) zonal velocity [m/s]
    v: np.ndarray     # (nx, ny, nz) meridional velocity [m/s]
    T: np.ndarray     # (nx, ny, nz) temperature [C]
    S: np.ndarray     # (nx, ny, nz) salinity [psu]
    eta: np.ndarray   # (nx, ny)    sea surface height [m]

    # ── Diagnostic variables (filled by physics modules, not time-integrated) ──
    w: Optional[np.ndarray] = field(default=None)       # (nx, ny, nz) vertical velocity [m/s]
    rho: Optional[np.ndarray] = field(default=None)     # (nx, ny, nz) density [kg/m^3]
    p: Optional[np.ndarray] = field(default=None)       # (nx, ny, nz) pressure [Pa]

    @property
    def shape(self):
        """Horizontal shape (nx, ny)."""
        return self.u.shape[:2]

    @property
    def nz(self):
        """Number of vertical levels."""
        return self.u.shape[2]

    def copy(self):
        """Deep copy of the state (arrays copied, diagnostics reset)."""
        return ModelState(
            u=self.u.copy(),
            v=self.v.copy(),
            T=self.T.copy(),
            S=self.S.copy(),
            eta=self.eta.copy(),
            w=self.w.copy() if self.w is not None else None,
            rho=self.rho.copy() if self.rho is not None else None,
            p=self.p.copy() if self.p is not None else None,
        )

    def diagnostics_filled(self):
        """Check if diagnostic fields (w, rho, p) have been computed."""
        return self.w is not None and self.rho is not None and self.p is not None


def initialize_state(grid, physics):
    """
    Create a ModelState with uniform reference values (rest state).

    Args:
        grid: OceanGrid instance
        physics: PhysicsConfig instance

    Returns: ModelState with u=v=0, T=T_ref, S=S_ref, eta=0
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    return ModelState(
        u=np.zeros((nx, ny, nz)),
        v=np.zeros((nx, ny, nz)),
        T=np.full((nx, ny, nz), physics.T_ref),
        S=np.full((nx, ny, nz), physics.S_ref),
        eta=np.zeros((nx, ny)),
    )
