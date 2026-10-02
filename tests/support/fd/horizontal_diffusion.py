"""Conservation, dissipation and accuracy of the default scalar diffusion path."""

from dataclasses import replace

import numpy as np

from config import OMEGA, R_EARTH, PhysicsConfig
from jax_solver_global import (
    make_solver_global,
)
from tests.support.grid import all_wet_grid


def _parameters(latitude_limit=30., land=False, ny=32, nx=24):
    grid = all_wet_grid(nx=nx, ny=ny, nz=4)
    latitude = np.linspace(-latitude_limit, latitude_limit, ny)
    cosine = np.cos(np.radians(latitude))
    wet = np.ones((nx, ny, 4))
    if land:
        wet[4:9, ny // 2:ny // 2 + 5, :] = 0.
        wet[12:17, ny // 4:ny // 4 + 5, 2:] = 0.
    grid = replace(
        grid, lat=latitude, cos_lat=cosine,
        dx_2d=np.broadcast_to(R_EARTH * np.radians(360. / nx) * cosine, (nx, ny)).copy(),
        dy=R_EARTH * np.radians(latitude[1] - latitude[0]),
        f=np.broadcast_to(2. * OMEGA * np.sin(np.radians(latitude)), (nx, ny)).copy(),
        z=np.array([0., -5., -20., -50.]), dz=np.array([5., 15., 30.]),
        wet_mask_3d=wet, wet_mask=wet[:, :, 0], ocean_mask=wet[:, :, 0].astype(bool),
        land_mask=1. - wet[:, :, 0])
    physics = replace(PhysicsConfig(), nu_h=0., nu_v=0., nu_bi=0., kappa_h=100.,
                      kappa_v=0., kappa_bi=0., kappa_conv=0., kappa_gm=0., kappa_redi=0.)
    forcing = tuple(np.zeros((nx, ny)) for _ in range(3))
    _, _, _, params, _ = make_solver_global(
        grid, physics, 60., forcing=forcing, lambda_bulk=0., mode_split=True,
        polar_cap_rows=0, polar_cap_taper=0, dtype="float64", return_params=True)
    volume = grid.dx_2d[:, :, None] * grid.dy * np.asarray(params.dz_node) * wet
    return grid, params, volume
