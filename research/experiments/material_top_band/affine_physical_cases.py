"""Exact manufactured means; separate physical field is the oracle authority."""
import numpy as np

from . import inventory_pressure as inventory
from .affine_physical_oracle import Specification
from .real_geometry import ColumnStocks


def manufactured_case(*, eta=(-.2, -.2), density_gradient_x=.15,
                      density_intercept=1., density_slope=-.4, bottom=-3.,
                      external_pressure=(80., 92.), nonlinear_TS=False,
                      u0=.03, strain=.008, velocity_y=-.02):
    spec = Specification(tuple(eta), tuple(external_pressure), density_gradient_x,
                         density_intercept=density_intercept, density_slope=density_slope,
                         bottom=bottom, velocity_y=velocity_y)
    widths = [np.arange(1., 15.), np.arange(14., 0., -1.)**1.3]
    z = np.empty((2, 15))
    stocks = np.empty((2, 14, 4))
    for side in range(2):
        fractions = widths[side] / widths[side].sum()
        z[side] = np.r_[eta[side], eta[side] - np.cumsum(fractions) * (eta[side] - spec.bottom)]
        z[side, -1] = spec.bottom
        for k in range(14):
            lower, upper = z[side, k + 1], z[side, k]
            height, center = upper - lower, .5 * (upper + lower)
            anomaly = spec.density_intercept + density_gradient_x * side * spec.distance + spec.density_slope * center
            temperature = 15. + .1 * (lower**2 + lower * upper + upper**2) / 3. if nonlinear_TS else 15.
            salinity = 35. + (anomaly / 1025. + 2e-4 * (temperature - 15.)) / 7.6e-4
            stocks[side, k] = height * np.array([temperature, salinity, 1025. * (u0 + strain * side * spec.distance), velocity_y * 1025.])
    h = -np.diff(z, axis=-1)
    state = ColumnStocks(np.full(2, 14, dtype=int), np.ones((2, 14), dtype=bool),
                         np.asarray(eta, dtype=float), np.full(2, spec.bottom), z[:, 3].copy(), z, h, stocks)
    return inventory.reconstruct(state, external_pressure_Pa=external_pressure), spec
