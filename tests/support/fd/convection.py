"""Controlled FD columns for hard-gated convection checks."""
import jax.numpy as jnp

from tests.support.fd.reference_geometry import _fixture


def _material_fixture(**options):
    settings = dict(process_time_scheme="symmetric_fast_v3", match_barotropic_transport=True,
                    monotone_adv=True, use_scan=True)
    settings.update(options)
    return _fixture(**settings)


def _state(initialize, eta=0., temperature=15., salinity=35.):
    state = initialize()
    return state._replace(T=jnp.full_like(state.T, temperature), S=jnp.full_like(state.S, salinity),
                          eta=jnp.full_like(state.eta, eta))
