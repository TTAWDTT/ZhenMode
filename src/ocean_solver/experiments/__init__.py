"""Strict, reproducible experiment assembly for the existing production method.

The catalog and resolver are usable without JAX. Numerical execution imports the
production runtime only inside its isolated worker process.
"""

from .resolve import expand_experiment, expand_sweep
from .schema import ConfigurationError, load_document

__all__ = ["ConfigurationError", "expand_experiment", "expand_sweep", "load_document"]
