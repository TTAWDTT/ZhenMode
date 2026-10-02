"""Independent moving-node stock gates; these are not industrial qualification."""



import jax.numpy as jnp
import numpy as np

from tests.support.material.reference_geometry import _fixture


def _material_fixture(**options):
    settings = dict(process_time_scheme="symmetric_fast_v3", match_barotropic_transport=True,
                    monotone_adv=True, use_scan=True)
    settings.update(options)
    return _fixture(**settings)

def _state(initialize, eta=0., temperature=15., salinity=35.):
    state = initialize()
    return state._replace(T=jnp.full_like(state.T, temperature), S=jnp.full_like(state.S, salinity),
                          eta=jnp.full_like(state.eta, eta))

def _numpy_divergence(east, north, params):
    incoming_north = np.roll(north, 1, axis=1)
    incoming_north[:, 0] = 0.
    return ((east - np.roll(east, 1, axis=0)) / np.asarray(params.dx_2d)[..., None]
            + (north - incoming_north) / (float(params.dy) * np.asarray(params.cos_lat)[None, :, None]))

def _numpy_transport_rhs(concentration, faces, params):
    wet = np.asarray(params.wet_mask_z)
    east, north = faces
    vertical = np.flip(np.cumsum(np.flip(_numpy_divergence(east, north, params), axis=-1), axis=-1), axis=-1)
    internal = vertical[..., 1:] * np.where(vertical[..., 1:] >= 0., concentration[..., :-1], concentration[..., 1:])
    internal *= wet[..., :-1] * wet[..., 1:]
    zero = np.zeros_like(concentration[..., :1])
    out_z = np.concatenate((internal, zero), axis=-1) - np.concatenate((zero, internal), axis=-1)
    tracer_east = east * np.where(east >= 0., concentration, np.roll(concentration, -1, axis=0))
    tracer_north = north * np.where(north >= 0., concentration, np.roll(concentration, -1, axis=1))
    return -_numpy_divergence(tracer_east, tracer_north, params) - out_z
