"""Shared nonuniform Cartesian channel fixture for transport operator tests."""

from dataclasses import replace

import numpy as np

from tests.support.grid import all_wet_grid
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.solver.factory import make_solver_global


def channel(nx=16, **changes):
    grid = all_wet_grid(nx=nx, ny=8, nz=4)
    grid = replace(grid, z=np.array([0., -5., -20., -100.]), dz=np.array([5., 15., 80.]),
                   depth=np.full((nx, 8), 100.), dx_2d=np.full((nx, 8), 100000 / nx),
                   dy=1000., cos_lat=np.ones(8), f=np.zeros((nx, 8)))
    physics = PhysicsConfig(**dict.fromkeys((
        'nu_h', 'nu_v', 'nu_bi', 'kappa_h', 'kappa_v', 'kappa_bi',
        'kappa_conv', 'kappa_gm', 'kappa_redi', 'r_bot', 'cd'), 0.))
    options = dict(return_params=True, column_geometry='nodal_dual_v1',
                   conservative_kv=True, localize_conv=True, polar_cap_rows=0,
                   polar_cap_taper=0, external_mode_scheme='symmetric',
                   meridional_boundary_scheme='closed_faces',
                   pressure_continuity_scheme='centered_fourth',
                   tracer_transport_scheme='product_fourth') | changes
    return grid, make_solver_global(grid, physics, 10., **options)
