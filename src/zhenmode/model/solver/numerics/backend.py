"""Historical JAX precision setup shared by FD execution modules."""

import os

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


jax.config.update('jax_enable_x64', True)

__all__ = ('jax', 'jnp', 'np')
