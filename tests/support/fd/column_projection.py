"""Projection must constrain native wet-face continuity, not a surface-mask proxy."""


import jax.numpy as jnp
import numpy as np


def _velocities(params, seed=2718):
    random = np.random.default_rng(seed)
    shape = params.wet_mask_z.shape
    return tuple(jnp.asarray(random.normal(size=shape)) * params.wet_mask_z for _ in range(2))
