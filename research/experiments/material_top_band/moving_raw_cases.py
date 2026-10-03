"""Frozen manufactured physical inputs for the new raw-mean authority."""
from dataclasses import dataclass, replace

import numpy as np

from . import inventory_pressure as inventory
from .affine_physical_cases import manufactured_case


@dataclass(frozen=True)
class MeanFlowSpecification:
    distance: float = 2.
    length: float = 1.5
    eta: float = -.2
    bottom: float = -3.
    U: float = .03
    alpha: float = .08
    V: float = -.02
    density_intercept: float = 1.
    density_slope: float = -.4
    external: tuple = (80., 200.)
    interior_interfaces: tuple = ()


def manufactured_mean_case(*, U=.03, alpha=.08, V=-.02, external=(80., 200.), density_intercept=1., density_slope=-.4):
    profile, _ = manufactured_case(density_gradient_x=0., u0=0., strain=0., velocity_y=V,
                                   external_pressure=external, density_intercept=density_intercept, density_slope=density_slope)
    stocks = profile.state.stocks.copy()
    stocks[..., 2] = profile.eos.rho0 * profile.state.h * np.array([U + alpha * .5, U + alpha * 1.5])[:, None]
    profile = inventory.reconstruct(replace(profile.state, stocks=stocks), external_pressure_Pa=external)
    spec = MeanFlowSpecification(U=U, alpha=alpha, V=V, external=tuple(external), density_intercept=density_intercept,
                                 density_slope=density_slope, interior_interfaces=tuple(tuple(z) for z in profile.state.interfaces[:, 1:]))
    return profile, spec
