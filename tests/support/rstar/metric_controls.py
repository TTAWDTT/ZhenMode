"""Independent moving-coordinate controls before any material factory cutover."""

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from ocean_solver.config.definitions import ALPHA_T, G_EARTH, R_EARTH, RHO_0
from ocean_solver.geometry.fd import make_fd_params
from ocean_solver.numerics.horizontal import _gradient_conservative_3d
from ocean_solver.state.types import JaxStateG
from research.experiments.material_rstar_coordinates.kernel import (
    coordinate_pressure_gradient,
    hydrostatic_pressure,
    rstar_geometry,
)
from tests.support.grid import all_wet_grid


def _fixture(stairs=True, nx=8, ny=5, depths=None):
    if depths is None:
        depths = np.array([0., 5., 15., 50., 500., 1000., 2000.])
    grid = all_wet_grid(nx=nx, ny=ny, nz=len(depths))
    wet = np.ones((nx, ny, len(depths)))
    if stairs:
        wet[1:3, :, 5:] = 0.
        wet[4:6, :, 6:] = 0.
        wet[0, 2] = 0.
    ocean = wet[..., 0]
    grid = replace(grid, z=-depths, dz=np.diff(depths), wet_mask_3d=wet,
                   wet_mask=ocean, ocean_mask=ocean.astype(bool), land_mask=1. - ocean)
    base = make_fd_params(grid, column_geometry="nodal_dual_v1")
    params = SimpleNamespace(**base._asdict(), T_ref=15., S_ref=35., monotone_adv=True, fct_adv=False)
    return grid, params, depths

def _geometry(params, depths, eta):
    return rstar_geometry(jnp.asarray(eta), jnp.asarray(depths), params.dz_node, params.wet_mask_z)

def _state(rho, eta, params):
    zeros = jnp.zeros_like(rho)
    return JaxStateG(zeros, zeros, params.T_ref - rho / (RHO_0 * ALPHA_T),
                     jnp.full_like(rho, params.S_ref), eta, jnp.zeros_like(eta))

def _pressure_coordinate_errors():
    errors = []
    for count in (24, 48, 96):
        depths = np.linspace(0., 4000., count + 1)
        grid, params, _ = _fixture(stairs=False, nx=count, ny=count // 2, depths=depths)
        longitude = np.arange(count) * (360. / count)
        latitude = -30. + (np.arange(count // 2) + .5) * (60. / (count // 2))
        cosine = np.cos(np.deg2rad(latitude))
        spacing = R_EARTH * np.deg2rad(60. / (count // 2))
        grid = replace(grid, lon=longitude, lat=latitude, dy=spacing, cos_lat=cosine,
                       dx_2d=np.broadcast_to(R_EARTH * np.deg2rad(360. / count) * cosine, (count, count // 2)).copy())
        params = SimpleNamespace(**make_fd_params(grid, column_geometry="nodal_dual_v1")._asdict())
        eta = -2.6 + .4 * np.sin(np.deg2rad(longitude))[:, None] * cosine[None, :]
        geometry = _geometry(params, depths, eta)
        rho = 1.5 + .002 * geometry.node_depth + 1e-7 * geometry.node_depth ** 2
        pressure = hydrostatic_pressure(rho, jnp.asarray(eta), geometry, params)
        actual = coordinate_pressure_gradient(pressure, rho, geometry, params)
        primitive_surface = -1.5 * eta + .001 * eta ** 2 - (1e-7 / 3.) * eta ** 3
        potential = G_EARTH * (eta - primitive_surface / RHO_0)
        expected = tuple(-value for value in _gradient_conservative_3d(jnp.asarray(potential[..., None]), params))
        area = np.asarray(params.dx_2d)[:, 1:-1] * params.dy
        error = sum(np.sum((np.asarray(current)[:, 1:-1] - np.asarray(reference)[:, 1:-1]) ** 2 * area[..., None])
                    for current, reference in zip(actual, expected, strict=True))
        errors.append(float(np.sqrt(error / (area.sum() * (count + 1)))))
    return errors
