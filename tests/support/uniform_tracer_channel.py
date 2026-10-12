"""Fully wet physical-node fixture for consistent mass/arithmetic owners."""

from dataclasses import replace

import numpy as np

from tests.support.grid import all_wet_grid
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.solver.factory import make_solver_global


def column_case(**changes):
    n = 8
    grid = all_wet_grid(nx=n, ny=n, nz=16)
    z = np.linspace(0., -100., 16)
    grid = replace(grid, z=z, dz=-np.diff(z), depth=np.full((n, n), 100.),
                   dx_2d=np.full((n, n), 2000.), dy=2000., cos_lat=np.ones(n),
                   f=np.full((n, n), 1e-4))
    physics = PhysicsConfig(**dict.fromkeys((
        'nu_h', 'nu_v', 'nu_bi', 'kappa_h', 'kappa_v', 'kappa_bi',
        'kappa_conv', 'kappa_gm', 'kappa_redi', 'r_bot', 'cd'), 0.))
    options = dict(return_params=True, column_geometry='nodal_dual_v1',
                   conservative_kv=True, localize_conv=True, polar_cap_rows=0, polar_cap_taper=0,
                   external_mode_scheme='symmetric', meridional_boundary_scheme='closed_faces',
                   pressure_continuity_scheme='centered_fourth', density_rotation_scheme='joint_shear',
                   hydrostatic_surface_scheme='linear_density', hydrostatic_integral_scheme='cubic_uniform_v1',
                   column_mean_scheme='cubic_consistent_v1', vertical_tracer_scheme='muscl_band_v3') | changes
    return grid, make_solver_global(grid, physics, 12.5, **options)
