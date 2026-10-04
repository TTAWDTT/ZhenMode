"""Actual substep means and top-face transports, not snapshot-rate proxies."""


import jax.numpy as jnp
import numpy as np

from zhenmode.model.state.types import JaxStateG


def _state(params):
    random = np.random.default_rng(2718)
    shape = params.wet_mask_z.shape
    return JaxStateG(jnp.asarray(random.normal(size=shape) * 0.03) * params.wet_mask_z,
                     jnp.asarray(random.normal(size=shape) * 0.03) * params.wet_mask_z,
                     jnp.asarray(random.uniform(5., 25., shape)),
                     jnp.asarray(random.uniform(33., 37., shape)),
                     jnp.zeros(shape[:2]), jnp.zeros(shape[:2]))
