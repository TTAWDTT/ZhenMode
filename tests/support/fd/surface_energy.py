"""Discrete surface-energy and local-diffusion budget regressions."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from config import PhysicsConfig
from jax_solver_global import (
    JaxStateG,
    make_solver_global,
)
from tests.support.grid import all_wet_grid


def _setup(depth=None, heat=100., ice=False, coastal_mask=None):
    grid = all_wet_grid(nx=8, ny=8, nz=4)
    grid = replace(grid, z=np.array([0., -5., -20., -50.]), dz=np.array([5., 15., 30.]))
    physics = replace(PhysicsConfig(), nu_h=0., nu_bi=0., nu_v=0., kappa_h=0.,
                      kappa_v=0., kappa_conv=0., kappa_gm=0., kappa_redi=0.)
    forcing = (np.zeros((8, 8)), np.zeros((8, 8)), np.full((8, 8), heat))
    step, _, _, params, _ = make_solver_global(
        grid, physics, 3600., forcing=forcing, lambda_bulk=0.,
        mixed_layer_depth_m=depth, dynamic_ice=ice, polar_cap_rows=0, polar_cap_taper=0,
        coastal_kappa_h_mask=coastal_mask, coastal_kappa_h=100., return_params=True)
    shape = (8, 8, 4)
    state = JaxStateG(jnp.zeros(shape), jnp.zeros(shape), jnp.full(shape, -1.8),
                      jnp.full(shape, 35.), jnp.zeros((8, 8)), jnp.zeros((8, 8)))
    return grid, params, state, step
