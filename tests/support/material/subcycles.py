"""Independent actual-volume schedules, source clocks, AD and restart gates."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from ocean_solver.config.definitions import ALPHA_T, BETA_S, C_P, RHO_0, PhysicsConfig
from tests.support.material.top import (
    _material_fixture,
    _numpy_divergence,
    _state,
)

POLICY = "actual_geometry_v2"

def _numpy_plan(state, params, faces, maximum):
    wet = np.asarray(params.wet_mask_z) > 0.
    east, north = faces
    divergence = _numpy_divergence(east, north, params)
    accumulated = np.flip(np.cumsum(np.flip(divergence, axis=-1), axis=-1), axis=-1)
    vertical = np.concatenate((np.zeros_like(accumulated[..., :1]), accumulated[..., 1:],
                               np.zeros_like(accumulated[..., :1])), axis=-1)
    south = np.roll(north, 1, axis=1)
    south[:, 0] = 0.
    outflow = ((np.maximum(east, 0.) + np.maximum(-np.roll(east, 1, axis=0), 0.))
               / np.asarray(params.dx_2d)[..., None]
               + (np.maximum(north, 0.) + np.maximum(-south, 0.))
               / (float(params.dy) * np.asarray(params.cos_lat)[None, :, None])
               + np.maximum(vertical[..., 1:], 0.) + np.maximum(-vertical[..., :-1], 0.))
    conductance = params.kappa_conv * wet[..., :-1] * wet[..., 1:] / np.asarray(params.dz_iface)
    zero = np.zeros_like(conductance[..., :1])
    convection = np.concatenate((zero, conductance), axis=-1) + np.concatenate((conductance, zero), axis=-1)
    feedback = np.zeros_like(outflow)
    feedback[..., 0] = np.maximum((params.lambda_bulk + np.asarray(params.coastal_bulk_lambda_2d)) / (RHO_0 * C_P)
                                  + np.asarray(params.dz_node).ravel()[0] * np.asarray(params.coastal_restore_coef_2d),
                                  np.asarray(params.dz_node).ravel()[0] * params.restore_coef_S)
    end_eta = np.asarray(state.eta) - params.dt * np.sum(divergence, axis=-1)
    thickness = np.broadcast_to(np.asarray(params.dz_node), wet.shape).copy()
    thickness[..., 0] += np.minimum(np.asarray(state.eta), end_eta)
    required = max(params.adv_nsub, params.conv_nsub,
                   int(np.ceil(np.max(np.where(wet, params.dt * (outflow + convection + feedback)
                                               / np.where(thickness > 0., thickness, 1.), 0.)) / .5)))
    return required, bool(required <= maximum and np.all(thickness[wet] > 0.))

def _numpy_convection(concentration, temperature, salinity, params):
    wet = np.asarray(params.wet_mask_z) > 0.
    density = -ALPHA_T * (temperature - params.T_ref) + BETA_S * (salinity - params.S_ref)
    gate = (density[..., :-1] > density[..., 1:]) & wet[..., :-1] & wet[..., 1:]
    flux = -params.kappa_conv * np.diff(concentration, axis=-1) / np.asarray(params.dz_iface) * gate
    zero = np.zeros_like(flux[..., :1])
    return np.concatenate((zero, flux), axis=-1) - np.concatenate((flux, zero), axis=-1)

def _flowing_case():
    physics = replace(PhysicsConfig(), nu_h=0., nu_v=0., nu_bi=0., kappa_h=100., kappa_v=1e-5,
                      kappa_bi=0., kappa_conv=.01, kappa_gm=0., kappa_redi=0., r_bot=0.)
    grid, (_, initialize, _, params, _) = _material_fixture(stairs=True, physics=physics,
                                                           lambda_bulk=20., T_atm=np.full((8, 8), 5.))
    generator = np.random.default_rng(299112)
    state = _state(initialize, temperature=10., salinity=34.7)
    state = state._replace(T=state.T - .2 * jnp.arange(state.T.shape[-1])[None, None, :],
                           eta=(jnp.full_like(state.eta, -2.4825)
                                + jnp.asarray(generator.normal(size=state.eta.shape) * 1e-4)) * params.wet_mask,
                           u=jnp.asarray(generator.normal(size=state.u.shape) * .05) * params.wet_mask_z,
                           v=jnp.asarray(generator.normal(size=state.v.shape) * .05) * params.wet_mask_z * params.interior_mask_z)
    return grid, params, state
