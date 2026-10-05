"""Density-threshold mixed-layer diagnostic used by initialization and scoring."""

from __future__ import annotations

import numpy as np

RHO_0 = 1025.0
T_REF = 15.0
ALPHA_T = 2.0e-4
BETA_S = 7.6e-4
S_REF = 35.0


def seawater_density(T: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Linear-equation-of-state density used by the current solver."""
    return RHO_0 * (1.0 - ALPHA_T * (np.asarray(T, dtype=float) - T_REF)
                    + BETA_S * (np.asarray(S, dtype=float) - S_REF))



def mixed_layer_depth(T: np.ndarray, S: np.ndarray, z: np.ndarray,
                      ocean: np.ndarray | None = None,
                      ref_depth: float = 10.0,
                      density_delta: float = 0.03) -> np.ndarray:
    """Column mixed-layer depth in metres positive down.

    Uses the common density-threshold definition and the current solver's
    linear equation of state.  This is the smallest defensible MLD diagnostic:
    independent of a particular mixing parameterization and applicable to WOA,
    a saved 3D snapshot, or a model restart.
    """
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)
    depth = -np.asarray(z, dtype=float)
    rho = seawater_density(T, S)
    nx, ny, _ = rho.shape
    wet = np.ones((nx, ny), dtype=bool) if ocean is None \
        else np.asarray(ocean, dtype=bool)
    ref_k = int(np.argmin(np.abs(depth - ref_depth)))
    mld = np.full((nx, ny), np.nan, dtype=float)
    for i in range(nx):
        for j in range(ny):
            if not wet[i, j]:
                continue
            valid = np.isfinite(rho[i, j]) & np.isfinite(S[i, j])
            if not valid.any():
                continue
            delta = rho[i, j] - rho[i, j, ref_k]
            below = np.where(valid & (depth >= ref_depth)
                             & (delta >= density_delta))[0]
            if not below.size:
                wet_k = np.where(valid)[0]
                mld[i, j] = float(depth[wet_k[-1]]) if wet_k.size else np.nan
            else:
                mld[i, j] = float(depth[int(below[0])])
    return mld
