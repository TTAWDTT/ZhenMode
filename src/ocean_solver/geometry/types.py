"""Pure GlobalOceanGrid contract without bathymetry I/O."""
from dataclasses import dataclass

import numpy as np


@dataclass
class GlobalOceanGrid:
    """Global lat-lon grid for the FD solver.

    Convention:
      - 2D arrays: shape (nx, ny), axis 0 = zonal (lon), axis 1 = meridional (lat)
      - lon is periodic (axis 0 wraps); lat is bounded (±lat_max, no wrap)
      - Vertical: z negative downward, z=0 at surface
      - y is a real closed boundary, not a periodic seam, and land cells are
        REAL land (wet_mask = 0) rather than sponge/fringe nodes. Forcing
        profiles must therefore be built from ocean-only statistics and must
        NOT be y-tapered to the domain mean: that is a periodic-seam
        artifact, and tapering T_atm to ~14 C at 59.5 N/S injects +0.3..0.9 K/d
        of spurious polar warming and flattens the model meridional SST
        gradient by ~30% vs WOA.
    """
    # Horizontal coordinates (1D)
    lon: np.ndarray       # (nx,) longitude [degrees E], periodic
    lat: np.ndarray       # (ny,) latitude [degrees N], ±lat_max

    # Grid spacing — dx varies with latitude (spherical metric)
    dx_2d: np.ndarray     # (nx, ny) zonal spacing [m] = R*cos(lat)*dlon
    dy: float             # (ny,) meridional spacing [m] (constant)
    cos_lat: np.ndarray   # (ny,) cos(lat) metric factor

    # Coriolis
    f: np.ndarray         # (nx, ny) Coriolis parameter [1/s] = 2*Omega*sin(lat)

    # Vertical grid
    z: np.ndarray         # (nz,) level depths [m, negative downward]
    dz: np.ndarray        # (nz-1,) layer thicknesses [m, positive]
    nz: int

    # Bathymetry
    depth: np.ndarray     # (nx, ny) ocean depth [m, positive; 0 on land]
    wet_mask: np.ndarray  # (nx, ny) float, 1.0 = ocean (wet), 0.0 = land (dry)
    ocean_mask: np.ndarray  # (nx, ny) bool, True = ocean (alias of wet_mask>0)
    land_mask: np.ndarray   # (nx, ny) bool, True = land
    # Vertical wet mask: True where layer k is above the seafloor AND the
    # column is ocean. This is the FIX for the ghost-water-column bug —
    # without it, _compute_hydrostatic_pressure integrates density over
    # layers below the seafloor (WOA-interpolated T that has no physical
    # water), producing huge spurious PGF at steep topography.
    # Convention: layer k (at z[k]) is wet iff |z[k]| <= depth AND ocean.
    wet_mask_3d: np.ndarray  # (nx, ny, nz) float, 1.0 = wet (water present)

    # Dimensions
    nx: int
    ny: int
