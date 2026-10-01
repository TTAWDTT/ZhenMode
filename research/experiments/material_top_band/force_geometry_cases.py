"""Manufactured stock/footprint inputs, never an archived observational profile."""
import numpy as np

from . import inventory_pressure as p
from .real_geometry import ColumnStocks


def columns(eta=(0., 1., 2.), *, slots=14, bottom=-1., periodic=True,
            temperature=15., salinity=35., velocity=(.03, -.02), shear=False,
            external_pressure=0.):
    eta = np.asarray(eta, dtype=float)
    if eta.ndim != 1 or slots < 1:
        raise ValueError('one-dimensional columns and positive slot count required')
    bottoms = np.broadcast_to(bottom, eta.shape).copy()
    count = np.full(eta.shape, slots, dtype=int)
    z = np.empty(eta.shape + (slots + 1,))
    h = np.empty(eta.shape + (slots,))
    stocks = np.empty(h.shape + (4,))
    moving = min(slots, 3)
    fractions = (2. * np.arange(moving) + 1.) / moving**2
    bands = bottoms.copy() if slots == moving else .25 * bottoms
    for i in range(len(eta)):
        z[i, 0] = eta[i]
        z[i, 1:moving + 1] = eta[i] - np.cumsum(fractions) * (eta[i] - bands[i])
        z[i, moving:] = np.linspace(bands[i], bottoms[i], slots - moving + 1)
        h[i] = -np.diff(z[i])
        for k in range(slots):
            lo, hi = z[i, k + 1], z[i, k]
            T = temperature(lo, hi) if callable(temperature) else temperature
            S = salinity(lo, hi) if callable(salinity) else salinity
            u, v = velocity
            if shear:
                u += .01 * i + .005 * k
                v += -.004 * i + .002 * k
            stocks[i, k] = h[i, k] * np.array([T, S, 1025. * u, 1025. * v])
    state = ColumnStocks(count, np.ones_like(h, dtype=bool), eta, bottoms, bands, z, h, stocks)
    boxes = np.array([[i, i + 1., 0., 2.] for i in range(len(eta))])
    footprint = p.RectangularFootprint(boxes, np.full(eta.shape, 2.),
                                       (float(len(eta)), 0.) if periodic else (0., 0.))
    requests = [p.FaceRequest((i,), (i + 1,), 2., (1., 0.)) for i in range(len(eta) - 1)]
    walls = [p.OuterWall((i,), 1., (0., sign)) for i in range(len(eta)) for sign in [-1., 1.]]
    if periodic:
        requests.append(p.FaceRequest((len(eta) - 1,), (0,), 2., (1., 0.), (float(len(eta)), 0.)))
    else:
        walls.extend([p.OuterWall((0,), 2., (-1., 0.)), p.OuterWall((len(eta) - 1,), 2., (1., 0.))])
    return p.reconstruct(state, external_pressure_Pa=external_pressure), footprint, tuple(requests), tuple(walls)
