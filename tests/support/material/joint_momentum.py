"""Independent viscosity matrices, modal energy, time, AD and restart gates."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from config import OMEGA, R_EARTH, PhysicsConfig
from jax_solver_global import make_solver_global
from tests.support.material.reference_geometry import _fixture

POLICY = dict(subcycle_scheme="actual_geometry_v2", momentum_diffusion_scheme="joint_heun_v1")

def _controlled_factory(*, stairs=False, metric=False, duration=600., physics=None):
    grid, _ = _fixture(stairs=stairs)
    if metric:
        latitude = np.arange(-7., 8., 2.)
        cosine = np.cos(np.deg2rad(latitude))
        spacing = R_EARTH * np.deg2rad(2.)
        grid = replace(grid, lat=latitude, cos_lat=cosine, dy=spacing,
                       dx_2d=np.broadcast_to(spacing * cosine, (8, 8)).copy(),
                       f=np.broadcast_to(2. * OMEGA * np.sin(np.deg2rad(latitude)), (8, 8)).copy())
    else:
        grid = replace(grid, dx_2d=np.full((8, 8), 1000.), dy=1000., cos_lat=np.ones(8), f=np.zeros((8, 8)))
    if physics is None:
        physics = replace(PhysicsConfig(), nu_h=800., nu_v=.01625, nu_bi=7.5e7,
                          kappa_h=0., kappa_v=0., kappa_bi=0., kappa_conv=0., kappa_gm=0., kappa_redi=0., r_bot=0.)
    _, initialize, _, params, _ = make_solver_global(
        grid, physics, duration, dt_bt=duration / 24., mode_split=True, dtype="float64", return_params=True,
        nu_nsub="cfl", column_geometry="nodal_dual_v1", conservative_kv=True, localize_conv=True,
        match_barotropic_transport=True, process_time_scheme="symmetric_fast_v3", monotone_adv=True,
        polar_cap_rows=0, polar_cap_taper=0, use_scan=True)
    state = initialize()._replace(T=jnp.full((8, 8, 4), 15.), S=jnp.full((8, 8, 4), 35.))
    return grid, params, state

def _numpy_operators(params):
    shape = params.wet_mask_z.shape
    size = int(np.prod(shape))
    wet = np.asarray(params.wet_mask_z)
    horizontal, vertical = np.zeros((size, size)), np.zeros((size, size))
    widths = np.asarray(params.dz_node).ravel()
    distances = np.asarray(params.dz_iface).ravel()
    for longitude, latitude, depth in np.ndindex(shape):
        row = np.ravel_multi_index((longitude, latitude, depth), shape)
        for next_longitude in ((longitude - 1) % shape[0], (longitude + 1) % shape[0]):
            column = np.ravel_multi_index((next_longitude, latitude, depth), shape)
            rate = wet[longitude, latitude, depth] * wet[next_longitude, latitude, depth] / float(params.dx_2d[longitude, latitude]) ** 2
            horizontal[row, column] += rate
            horizontal[row, row] -= rate
        for next_latitude in (max(latitude - 1, 0), min(latitude + 1, shape[1] - 1)):
            column = np.ravel_multi_index((longitude, next_latitude, depth), shape)
            cosine_face = .5 * (float(params.cos_lat[latitude]) + float(params.cos_lat[next_latitude]))
            rate = (wet[longitude, latitude, depth] * wet[longitude, next_latitude, depth]
                    * cosine_face / float(params.cos_lat[latitude]) / float(params.dy) ** 2)
            horizontal[row, column] += rate
            horizontal[row, row] -= rate
        for next_depth in (depth - 1, depth + 1):
            if 0 <= next_depth < shape[2] and wet[longitude, latitude, depth] and wet[longitude, latitude, next_depth]:
                column = np.ravel_multi_index((longitude, latitude, next_depth), shape)
                rate = 1. / (widths[depth] * distances[min(depth, next_depth)])
                vertical[row, column] += rate
                vertical[row, row] -= rate
    operator = (params.nu_h * horizontal + params.nu_v * vertical - params.nu_bi * horizontal @ horizontal) * wet.ravel()[:, None]
    return horizontal, vertical, operator
