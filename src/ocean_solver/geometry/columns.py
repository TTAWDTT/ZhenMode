"""Pure nodal_control_thickness contract without bathymetry I/O."""
import numpy as np


def nodal_control_thickness(z):
    """Static nodal dual widths: surface at zero, bottom at the last node.

    Interior faces lie halfway between point samples. Masking a prefix of
    these cells represents a staircase bottom at the last retained cell edge;
    it does not reproduce the raw bathymetry or water below the deepest node.
    A first sample below zero represents the surface-to-first-face interval.
    """
    depths = -np.asarray(z, dtype=np.float64)
    if (depths.ndim != 1 or depths.size < 2 or not np.all(np.isfinite(depths))
            or depths[0] < 0. or not np.all(np.diff(depths) > 0.)):
        raise ValueError("z must contain at least two finite, decreasing, nonpositive nodes")
    edges = np.concatenate(([0.], 0.5 * (depths[:-1] + depths[1:]), [depths[-1]]))
    return np.diff(edges)
