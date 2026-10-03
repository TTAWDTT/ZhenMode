"""Historical JAX precision setup shared by FD execution modules."""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402
import jax.scipy.sparse.linalg  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ('jax', 'jnp', 'np')
