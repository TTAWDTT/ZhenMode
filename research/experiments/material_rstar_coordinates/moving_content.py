"""Prescribed-transport r-star content controls, not a coupled ocean solver."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax_solver_global import _advection_scalar, _face_transport_divergence
from research.experiments.material_rstar_coordinates.kernel import (
    relative_vertical_transport,
    rstar_geometry,
)
from research.experiments.material_rstar_coordinates.sparse_diffusion import (
    accumulate_edges,
    edge_coefficients,
    limited_negative_exchange,
    neighbor_bounds,
)


class MovingContentResult(NamedTuple):
    content: jax.Array
    surface: jax.Array
    valid: jax.Array
    fraction: jax.Array
    minimum_limiter: jax.Array
    exchange: jax.Array
    applied_source: jax.Array
    mass_residual: jax.Array
    bottom_residual: jax.Array


def donor_outgoing_volume(east, north, relative, params):
    """Original spherical face contract, before multiplication by cell area."""
    west = jnp.roll(east, 1, axis=0)
    south = jnp.roll(north, 1, axis=1).at[:, 0].set(0.)
    horizontal = ((jnp.maximum(east, 0.) + jnp.maximum(-west, 0.)) * params.inv_dx[..., :1]
                  + (jnp.maximum(north, 0.) + jnp.maximum(-south, 0.)) * params.inv_dy / params.cos_lat[None, :, None])
    connected = params.wet_mask_z[..., :-1] * params.wet_mask_z[..., 1:]
    internal = relative[..., 1:-1] * connected
    upper = jnp.concatenate((jnp.zeros_like(internal[..., :1]), internal), axis=-1)
    lower = jnp.concatenate((internal, jnp.zeros_like(internal[..., :1])), axis=-1)
    return horizontal + jnp.maximum(lower, 0.) + jnp.maximum(-upper, 0.)


def moving_euler(content, surface, faces, source_rate, duration, params, depths, graph, kappa_h, kappa_v):
    """Joint donor/diffusion/source stage with actual endpoint mass and rollback."""
    if content.dtype != jnp.float64 or content.shape != graph.shape or surface.dtype != jnp.float64:
        raise ValueError("moving content requires the frozen float64 grid")
    if not params.monotone_adv or params.fct_adv:
        raise ValueError("moving content controls require original donor transport")
    geometry = rstar_geometry(surface, depths, params.dz_node, params.wet_mask_z)
    divergence = _face_transport_divergence(*faces, params)
    surface_rate = -jnp.sum(divergence, axis=-1)
    endpoint = surface + duration * surface_rate
    next_geometry = rstar_geometry(endpoint, depths, params.dz_node, params.wet_mask_z)
    relative = relative_vertical_transport(divergence, geometry)
    area = graph.area.reshape(graph.shape)
    mass = area * geometry.thickness
    next_mass = area * next_geometry.thickness
    denominator = jnp.where(graph.wet.reshape(graph.shape) & (mass > 0.), mass, 1.)
    concentration = jnp.where(graph.wet.reshape(graph.shape), content / denominator, 0.)
    zeros = jnp.zeros_like(content)
    advection = area * params.dz_node * _advection_scalar(concentration, zeros, zeros, relative, params, face_transport=faces)
    volume_rate = area * params.dz_node * _advection_scalar(jnp.ones_like(content), zeros, zeros, relative, params, face_transport=faces)
    mass_residual = next_mass - mass - duration * volume_rate
    coefficients, coefficient_valid = edge_coefficients(graph, geometry, kappa_h, kappa_v)
    positive = jnp.maximum(coefficients, 0.)
    difference = concentration.ravel()[graph.pair_right] - concentration.ravel()[graph.pair_left]
    low_exchange = duration * advection.ravel() + accumulate_edges(graph, duration * positive * difference)
    applied_source = duration * jnp.broadcast_to(source_rate, graph.shape)
    low_content = content.ravel() + low_exchange + applied_source.ravel()
    lower, upper = neighbor_bounds(graph, concentration.ravel(), jnp.ones_like(coefficients))
    lower_content = next_mass.ravel() * lower + applied_source.ravel()
    upper_content = next_mass.ravel() * upper + applied_source.ravel()
    correction, minimum = limited_negative_exchange(graph, duration * jnp.minimum(coefficients, 0.) * difference,
                                                    low_content, lower_content, upper_content)
    exchange = (low_exchange + correction).reshape(graph.shape)
    attempted = content + exchange + applied_source
    next_denominator = jnp.where(next_mass > 0., next_mass, 1.)
    after = attempted / next_denominator
    source_increment = applied_source / next_denominator
    bound_floor = 64. * jnp.finfo(content.dtype).eps * (1. + jnp.abs(concentration) + jnp.abs(after) + jnp.abs(source_increment))
    outgoing = area * donor_outgoing_volume(*faces, relative, params)
    outgoing += accumulate_edges(graph, positive, "unsigned").reshape(graph.shape)
    fraction = jnp.max(jnp.where(graph.wet.reshape(graph.shape), duration * outgoing / denominator, 0.))
    wet = params.wet_mask_z > 0.
    east_gate = wet & jnp.roll(wet, -1, axis=0)
    north_gate = (wet & jnp.roll(wet, -1, axis=1)).at[:, -1].set(False)
    faces_valid = (jnp.all(jnp.isfinite(faces[0])) & jnp.all(jnp.isfinite(faces[1]))
                   & jnp.all(jnp.where(east_gate, True, faces[0] == 0.))
                   & jnp.all(jnp.where(north_gate, True, faces[1] == 0.)))
    source_valid = jnp.all(jnp.isfinite(applied_source)) & jnp.all(jnp.where(wet, True, applied_source == 0.))
    continuity_floor = 64. * jnp.finfo(content.dtype).eps * (jnp.abs(mass) + jnp.abs(next_mass) + jnp.abs(duration * volume_rate))
    bottom_floor = 64. * jnp.finfo(content.dtype).eps * jnp.sum(jnp.abs(divergence), axis=-1)
    valid = (geometry.valid & next_geometry.valid & coefficient_valid & faces_valid & source_valid
             & jnp.isfinite(duration) & (duration > 0.) & (fraction <= .5)
             & jnp.all(jnp.abs(mass_residual) <= continuity_floor)
             & jnp.all(jnp.abs(relative[..., -1]) <= bottom_floor)
             & jnp.all(jnp.where(wet, jnp.isfinite(content) & jnp.isfinite(attempted)
                                 & (after >= lower.reshape(graph.shape) + source_increment - bound_floor)
                                 & (after <= upper.reshape(graph.shape) + source_increment + bound_floor), True)))
    return MovingContentResult(jnp.where(valid, attempted, content), jnp.where(valid, endpoint, surface), valid,
                                fraction, minimum, exchange, applied_source, mass_residual, relative[..., -1])


def moving_heun(content, surface, faces, source_rate, duration, params, depths, graph, kappa_h, kappa_v):
    """SSP content and mass combination; both Euler geometries must be valid."""
    first = moving_euler(content, surface, faces, source_rate, duration, params, depths, graph, kappa_h, kappa_v)
    second = moving_euler(first.content, first.surface, faces, source_rate, duration, params, depths, graph, kappa_h, kappa_v)
    exchange = .5 * (first.exchange + second.exchange)
    source = .5 * (first.applied_source + second.applied_source)
    attempted = content + exchange + source
    endpoint = surface + .5 * ((first.surface - surface) + (second.surface - first.surface))
    valid = first.valid & second.valid & jnp.all(jnp.where(params.wet_mask_z > 0., jnp.isfinite(attempted), True))
    return MovingContentResult(jnp.where(valid, attempted, content), jnp.where(valid, endpoint, surface), valid,
                                jnp.maximum(first.fraction, second.fraction), jnp.minimum(first.minimum_limiter, second.minimum_limiter),
                                exchange, source, .5 * (first.mass_residual + second.mass_residual),
                                jnp.maximum(jnp.abs(first.bottom_residual), jnp.abs(second.bottom_residual)))
