"""Input contracts shared by the public solver and production CLI."""

from numbers import Integral, Real

import numpy as np


def finite_number(name, value, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0.:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.:
        raise ValueError(f"{name} must be nonnegative")
    return value


def integer_count(name, value, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_grid(grid):
    for name in ("nx", "ny", "nz"):
        integer_count(name, getattr(grid, name), minimum=2)
    finite_number("dy", grid.dy, positive=True)
    widths = np.asarray(grid.dx_2d)
    depths = np.asarray(grid.z)
    spacing = np.asarray(grid.dz)
    if widths.shape != (grid.nx, grid.ny) or not np.all(np.isfinite(widths) & (widths > 0.)):
        raise ValueError("dx_2d must have grid shape and positive finite spacing")
    if (depths.shape != (grid.nz,) or not np.all(np.isfinite(depths))
            or not np.all(np.diff(depths) < 0.)):
        raise ValueError("z must contain finite, strictly descending nodes")
    if (spacing.shape != (grid.nz - 1,) or not np.all(np.isfinite(spacing) & (spacing > 0.))
            or not np.allclose(spacing, -np.diff(depths), rtol=1e-12, atol=0.)):
        raise ValueError("dz must match the positive distances between z nodes")
