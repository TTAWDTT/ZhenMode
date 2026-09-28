"""Run-level budget diagnostics.

This module is deliberately separate from the solver core: it should be usable
from drivers and analysis scripts without pulling the JAX graph into the test.
It works on snapshot arrays, not on time derivatives, so it can be added to the
runner without changing the dynamical core.

The goal is not yet a full flux budget; it is a cheap, well-defined "audit
trail" that answers: is total heat/salt content drifting, is volume fixed, and
is that behaviour consistent across tracer-transport options?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import C_P, RHO_0


@dataclass(frozen=True)
class BudgetDiagnostics:
    """Global tracer/volume diagnostics for one snapshot."""

    mean_T: float
    mean_S: float
    mean_T_top: float
    mean_S_top: float
    heat_content_J: float
    salt_content_kg: float
    total_volume_m3: float
    mean_depth_m: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mean_T": self.mean_T,
            "mean_S": self.mean_S,
            "mean_T_top": self.mean_T_top,
            "mean_S_top": self.mean_S_top,
            "heat_content_J": self.heat_content_J,
            "salt_content_kg": self.salt_content_kg,
            "total_volume_m3": self.total_volume_m3,
            "mean_depth_m": self.mean_depth_m,
        }


def node_thickness(z):
    """Positive thickness associated with each z-level node.

    The model fields live on z-level nodes, not finite-volume layer centres.
    This mirrors the solver's ``dz_node`` convention:
    ``dz_node[0] = |z1-z0|`` and ``dz_node[-1] = |zN-1-zN-2|``.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z must be a 1-D array with at least two levels")
    dz = np.abs(np.diff(z))
    return np.concatenate(([dz[0]], 0.5 * (dz[:-1] + dz[1:]), [dz[-1]]))


def compute_budget_diagnostics(state, grid, rho0=RHO_0, cp=C_P) -> BudgetDiagnostics:
    """Compute global heat/salt/volume diagnostics from a solver snapshot.

    All calculations use the true wet mask and the node-based vertical metric.
    This is intentionally independent of JAX: the caller can pass device or
    host arrays; the result is always a plain ``BudgetDiagnostics``.
    """
    T = np.asarray(state.T, dtype=np.float64)
    S = np.asarray(state.S, dtype=np.float64)
    area = np.asarray(grid.dx_2d, dtype=np.float64) * float(grid.dy)
    wet3 = np.asarray(grid.wet_mask_3d, dtype=np.float64)
    dz_node = node_thickness(grid.z)

    if T.shape != wet3.shape or S.shape != wet3.shape:
        raise ValueError(
            f"state fields {T.shape} must match grid.wet_mask_3d {wet3.shape}")
    if wet3.ndim != 3:
        raise ValueError("wet_mask_3d must be 3-D")

    volume = (area[:, :, None] * dz_node[None, None, :]) * wet3
    total_volume = float(volume.sum())
    if total_volume <= 0.0:
        raise ValueError("total wet volume is zero; check the grid/mask")

    mean_T = float((T * volume).sum() / total_volume)
    mean_S = float((S * volume).sum() / total_volume)

    surface_area = area * np.asarray(grid.wet_mask, dtype=np.float64)
    surface_area_total = float(surface_area.sum())
    T_top = T[:, :, 0]
    S_top = S[:, :, 0]
    mean_T_top = float((T_top * surface_area).sum() / surface_area_total)
    mean_S_top = float((S_top * surface_area).sum() / surface_area_total)

    heat_content = float((rho0 * cp * T * volume).sum())
    salt_content = float((rho0 * (S / 1000.0) * volume).sum())
    mean_depth = total_volume / float(area.sum())

    return BudgetDiagnostics(
        mean_T=mean_T,
        mean_S=mean_S,
        mean_T_top=mean_T_top,
        mean_S_top=mean_S_top,
        heat_content_J=heat_content,
        salt_content_kg=salt_content,
        total_volume_m3=total_volume,
        mean_depth_m=mean_depth,
    )


def diagnostics_to_arrays(rows: list[BudgetDiagnostics]) -> dict[str, np.ndarray]:
    """Convert a list of diagnostics into flat arrays for npz output."""
    return {
        key: np.array([getattr(row, key) for row in rows], dtype=np.float64)
        for key in BudgetDiagnostics.__dataclass_fields__
    }
