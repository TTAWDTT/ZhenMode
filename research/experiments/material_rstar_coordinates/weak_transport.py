"""Physical-overlap nodal weak transport; instantaneous unbounded experiment."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from config import RHO_0
from research.experiments.material_rstar_coordinates.nodal_mass import solve_nodal_mass
from research.experiments.material_rstar_coordinates.pressure_work import (
    PotentialBasis,
    make_potential_basis,
    potential_conjugates,
)


class HatTrace(NamedTuple):
    weights: jax.Array
    derivative: jax.Array


class WeakParameters(NamedTuple):
    wet_mask_z: jax.Array
    wet_mask: jax.Array
    dz_node: jax.Array
    dx_2d: jax.Array
    dy: float
    cos_lat: jax.Array


def consistent_potential_basis(depths, params, reference_geometry, mass):
    lumped = make_potential_basis(depths, params)
    moment = lumped.mean_depth * params.dz_node * params.wet_mask_z
    conjugate = solve_nodal_mass(moment, reference_geometry, jnp.ones(params.wet_mask.shape), mass)
    return PotentialBasis(conjugate, lumped.column_depth)


def _hat_trace(nodes, wet, positions):
    count = jnp.sum(wet > 0., axis=-1)
    safe_nodes = jnp.where(wet > 0., nodes, jnp.inf)
    shape = positions.shape
    flattened_nodes = safe_nodes.reshape((-1, safe_nodes.shape[-1]))
    flattened_positions = positions.reshape((-1, int(np.prod(shape[2:]))))
    selected = jax.vmap(lambda column, locations: jnp.searchsorted(column, locations, side="right"))(flattened_nodes, flattened_positions)
    selected = selected.reshape(shape) - 1
    lower = jnp.clip(selected, 0, jnp.maximum(count - 2, 0)[..., None, None])
    upper = jnp.minimum(lower + 1, jnp.maximum(count - 1, 0)[..., None, None])
    broadcast_nodes = jnp.broadcast_to(nodes[..., None, None, :], shape + (nodes.shape[-1],))
    low = jnp.take_along_axis(broadcast_nodes, lower[..., None], axis=-1)[..., 0]
    high = jnp.take_along_axis(broadcast_nodes, upper[..., None], axis=-1)[..., 0]
    spacing = jnp.where(high > low, high - low, 1.)
    fraction = (positions - low) / spacing
    lower_hat = jax.nn.one_hot(lower, nodes.shape[-1], dtype=nodes.dtype)
    upper_hat = jax.nn.one_hot(upper, nodes.shape[-1], dtype=nodes.dtype)
    weights = lower_hat * (1. - fraction[..., None]) + upper_hat * fraction[..., None]
    derivative = (upper_hat - lower_hat) / spacing[..., None]
    last = jnp.maximum(count - 1, 0)
    bed_node = jnp.take_along_axis(nodes, last[..., None], axis=-1)[..., 0]
    tail = positions >= bed_node[..., None, None]
    bottom_hat = jax.nn.one_hot(last, nodes.shape[-1], dtype=nodes.dtype)[..., None, None, :]
    weights = jnp.where(tail[..., None], bottom_hat, weights)
    derivative = jnp.where(tail[..., None], 0., derivative)
    active = (count > 0)[..., None, None, None]
    return HatTrace(jnp.where(active, weights, 0.), jnp.where(active, derivative, 0.))


def _evaluate(trace, field, wet):
    safe = jnp.where(wet > 0., field, 0.)
    return jnp.sum(trace.weights * safe[..., None, None, :], axis=-1)


def _face_rates(density, velocity, geometry, surface, params, axis):
    def neighbor(values):
        return jnp.roll(values, -1, axis=axis)

    wet = params.wet_mask_z
    other_wet = neighbor(wet)
    nodes, other_nodes = geometry.node_depth, neighbor(geometry.node_depth)
    bed = geometry.interface_depth[..., -1]
    other_bed = neighbor(bed)
    top, other_top = -surface, -neighbor(surface)
    minimum_top = jnp.minimum(top, other_top)
    maximum_bed = jnp.maximum(bed, other_bed)
    face_top, face_bed = jnp.maximum(top, other_top), jnp.minimum(bed, other_bed)
    active = params.wet_mask * neighbor(params.wet_mask)
    if axis == 1:
        active = active.at[:, -1].set(0.)
    candidates = jnp.concatenate((jnp.where(wet > 0., nodes, bed[..., None]),
                                  jnp.where(other_wet > 0., other_nodes, other_bed[..., None]),
                                  top[..., None], other_top[..., None], bed[..., None], other_bed[..., None]), axis=-1)
    knots = jnp.sort(jnp.clip(candidates, minimum_top[..., None], maximum_bed[..., None]), axis=-1)
    left, right = knots[..., :-1], knots[..., 1:]
    width = right - left
    gauss, gauss_weight = np.polynomial.legendre.leggauss(3)
    locations = .5 * (left[..., None] + right[..., None] + width[..., None] * jnp.asarray(gauss))
    measures = .5 * width[..., None] * jnp.asarray(gauss_weight)
    left_trace, right_trace = _hat_trace(nodes, wet, locations), _hat_trace(other_nodes, other_wet, locations)
    endpoints = jnp.stack((left, right), axis=-1)
    left_end = _hat_trace(nodes, wet, endpoints)
    right_end = _hat_trace(other_nodes, other_wet, endpoints)
    measure = params.dy if axis == 0 else params.dx_2d / params.cos_lat[None, :] * .5 * (params.cos_lat + jnp.roll(params.cos_lat, -1))[None, :]
    aperture = ((.5 * (left + right) >= face_top[..., None])
                & (.5 * (left + right) <= face_bed[..., None])) * active[..., None]
    endpoint_flux = .5 * (_evaluate(left_end, velocity, wet) + _evaluate(right_end, neighbor(velocity), other_wet))
    endpoint_flux *= jnp.asarray(measure)[..., None, None] if jnp.ndim(measure) else measure
    endpoint_flux *= aperture[..., None]
    segment_volume = .5 * width * jnp.sum(endpoint_flux, axis=-1)
    prefix = jnp.concatenate((jnp.zeros_like(segment_volume[..., :1]), jnp.cumsum(segment_volume, axis=-1)[..., :-1]), axis=-1)
    displacement = locations - left[..., None]
    slope = (endpoint_flux[..., 1] - endpoint_flux[..., 0]) / jnp.where(width > 0., width, 1.)
    flux = endpoint_flux[..., 0, None] + slope[..., None] * displacement
    partial = prefix[..., None] + endpoint_flux[..., 0, None] * displacement + .5 * slope[..., None] * displacement ** 2
    total = jnp.sum(segment_volume, axis=-1)
    left_density = _evaluate(left_trace, density, wet)
    right_density = _evaluate(right_trace, neighbor(density), other_wet)
    centered = .5 * (left_density + right_density)
    horizontal = measures * flux * centered
    left_height, right_height = bed - top, other_bed - other_top
    left_primitive = partial - total[..., None, None] * (locations - top[..., None, None]) / jnp.where(left_height > 0., left_height, 1.)[..., None, None]
    right_primitive = partial - total[..., None, None] * (locations - other_top[..., None, None]) / jnp.where(right_height > 0., right_height, 1.)[..., None, None]
    inside_left = (locations >= top[..., None, None]) & (locations <= bed[..., None, None])
    inside_right = (locations >= other_top[..., None, None]) & (locations <= other_bed[..., None, None])
    left_terms = (-horizontal[..., None] * left_trace.weights
                  - (measures * left_density * left_primitive * inside_left)[..., None] * left_trace.derivative)
    right_terms = (horizontal[..., None] * right_trace.weights
                   + (measures * right_density * right_primitive * inside_right)[..., None] * right_trace.derivative)
    return jnp.sum(left_terms, axis=(-3, -2)), jnp.sum(right_terms, axis=(-3, -2)), total


def weak_content_rate(density, velocities, surface, geometry, params):
    safe_surface = jnp.where(params.wet_mask > 0., surface, 0.)
    content_rate = jnp.zeros_like(density)
    volume_rate = jnp.zeros_like(surface)
    for axis, velocity in enumerate(velocities):
        left, right, total = _face_rates(density, velocity, geometry, safe_surface, params, axis)
        content_rate += left + jnp.roll(right, 1, axis=axis)
        volume_rate += -total + jnp.roll(total, 1, axis=axis)
    return content_rate, volume_rate / (params.dx_2d * params.dy)


def weak_pressure_force(density, content, surface, geometry, basis, params):
    def transport(velocities):
        return weak_content_rate(density, velocities, surface, geometry, params)

    zeros = jnp.zeros_like(density)
    conjugates = potential_conjugates(content, surface, geometry, basis, params)
    adjoint = jax.linear_transpose(transport, (zeros, zeros))(conjugates)[0]
    mass = (params.dx_2d * params.dy)[..., None] * geometry.thickness
    denominator = RHO_0 * jnp.where(params.wet_mask_z > 0., mass, 1.)
    return tuple(jnp.where(params.wet_mask_z > 0., -force / denominator, 0.) for force in adjoint)
