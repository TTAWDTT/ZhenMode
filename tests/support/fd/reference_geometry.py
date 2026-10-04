"""Independent M1 gates for an opt-in repair of the original FD core."""

from dataclasses import replace

import numpy as np

from tests.support.grid import all_wet_grid
from zhenmode.model.config import PhysicsConfig
from zhenmode.model.solver.factory import make_solver_global

WIDTHS = np.array([2.5, 10., 22.5, 15.])

def _fixture(stairs=False, physics=None, wet_nodes=4, **options):
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    wet = np.ones((8, 8, 4))
    depth = np.full((8, 8), 50.)
    if wet_nodes < 4:
        wet[..., wet_nodes:] = 0.
        depth.fill({1: 1., 2: 10., 3: 30.}[wet_nodes])
    if stairs:
        wet[1:3, 3:5] = 0.
        depth[1:3, 3:5] = 0.
        wet[4:6, 2:5, 2:] = 0.
        depth[4:6, 2:5] = 10.
        wet[6, 3, 1:] = 0.
        depth[6, 3] = 1.
    grid = replace(grid, z=np.array([0., -5., -20., -50.]), dz=np.array([5., 15., 30.]),
                   depth=depth, wet_mask_3d=wet, wet_mask=wet[..., 0],
                   ocean_mask=wet[..., 0].astype(bool), land_mask=~wet[..., 0].astype(bool),
                   f=np.zeros((8, 8)))
    if physics is None:
        physics = replace(PhysicsConfig(), nu_h=0., nu_v=0., nu_bi=0., kappa_h=0.,
                          kappa_v=0., kappa_bi=0., kappa_conv=0., kappa_gm=0.,
                          kappa_redi=0., r_bot=0.)
    settings = dict(mode_split=True, dt_bt=5., dtype="float64", return_params=True,
                    column_geometry="nodal_dual_v1", conservative_kv=True, localize_conv=True,
                    polar_cap_rows=0, polar_cap_taper=0)
    settings.update(options)
    solver = make_solver_global(grid, physics, 10., **settings)
    return grid, solver

def _numpy_column_divergence(velocity_x, velocity_y, grid):
    wet = grid.wet_mask_3d
    flux_x = np.sum(0.5 * (velocity_x + np.roll(velocity_x, -1, axis=0))
                    * wet * np.roll(wet, -1, axis=0) * WIDTHS, axis=-1)
    flux_y = np.sum(0.5 * (velocity_y + np.roll(velocity_y, -1, axis=1))
                    * wet * np.roll(wet, -1, axis=1) * WIDTHS, axis=-1)
    flux_y[:, -1] = 0.
    cosine_face = 0.5 * (grid.cos_lat + np.roll(grid.cos_lat, -1))
    flux_y *= cosine_face
    incoming_y = np.roll(flux_y, 1, axis=1)
    incoming_y[:, 0] = 0.
    return ((flux_x - np.roll(flux_x, 1, axis=0)) / grid.dx_2d
            + (flux_y - incoming_y) / (grid.dy * grid.cos_lat))
