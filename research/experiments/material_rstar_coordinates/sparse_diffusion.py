"""Isolated sparse physical diffusion graph; not a material or production API."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class GatherGroup(NamedTuple):
    owners: jax.Array
    entries: jax.Array


class IncidenceGroup(NamedTuple):
    owners: jax.Array
    edges: jax.Array
    signs: jax.Array
    neighbors: jax.Array


class DiffusionGraph(NamedTuple):
    shape: tuple
    wet: jax.Array
    area: jax.Array
    left: jax.Array
    right: jax.Array
    axis: jax.Array
    supports: jax.Array
    reference_derivative: jax.Array
    difference: jax.Array
    inverse_horizontal_spacing: jax.Array
    face_measure: jax.Array
    term_faces: jax.Array
    term_left: jax.Array
    term_right: jax.Array
    pair_left: jax.Array
    pair_right: jax.Array
    coefficient_groups: tuple
    incidence_groups: tuple


class BoundedDiffusionResult(NamedTuple):
    content: jax.Array
    attempted_content: jax.Array
    valid: jax.Array
    fraction: jax.Array
    minimum_limiter: jax.Array
    exchange: jax.Array


def _gather_groups(owners, entries):
    identifiers, starts, counts = np.unique(owners, return_index=True, return_counts=True)
    ordering = np.argsort(owners, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts[:-1]))) if len(counts) else starts
    groups = []
    for count in np.unique(counts):
        selected = counts == count
        positions = starts[selected, None] + np.arange(count)
        groups.append(GatherGroup(jnp.asarray(identifiers[selected], dtype=jnp.int32),
                                  jnp.asarray(entries[ordering[positions]], dtype=jnp.int32)))
    return tuple(groups)


def make_diffusion_graph(params, stencil):
    """Freeze open-face topology; bounded-degree storage, no dense wet matrix."""
    wet = np.asarray(params.wet_mask_z) > 0.
    if wet.ndim != 3 or wet.shape[0] < 2 or wet.shape[1] < 2 or wet.size > np.iinfo(np.int32).max:
        raise ValueError("sparse graph requires a representable three-dimensional wet grid")
    levels = wet.shape[-1]
    node_ids = np.arange(wet.size, dtype=np.int32).reshape(wet.shape)
    indices = np.asarray(stencil.indices).reshape(wet.size, 3)
    derivative = np.asarray(stencil.coefficients).reshape(wet.size, 3)
    area = np.broadcast_to((np.asarray(params.dx_2d) * float(params.dy))[..., None], wet.shape).ravel().copy()
    inverse_x = np.broadcast_to(np.asarray(params.inv_dx), wet.shape).ravel()
    cosine = np.asarray(params.cos_lat)
    face_lists = [[] for _ in range(8)]
    for axis in range(3):
        next_ids = np.roll(node_ids, -1, axis=axis)
        opened = wet & np.roll(wet, -1, axis=axis)
        if axis != 0:
            selection = [slice(None)] * 3
            selection[axis] = -1
            opened[tuple(selection)] = False
        left, right = node_ids[opened], next_ids[opened]
        if axis < 2:
            supports = np.concatenate(((left // levels)[:, None] * levels + indices[left],
                                       (right // levels)[:, None] * levels + indices[right]), axis=1)
            reference = np.concatenate((derivative[left], derivative[right]), axis=1)
            difference = ((np.arange(6)[None, :] == np.argmax(supports == right[:, None], axis=1)[:, None]).astype(float)
                          - (np.arange(6)[None, :] == np.argmax(supports == left[:, None], axis=1)[:, None]).astype(float))
            inverse = inverse_x[left] if axis == 0 else np.full(len(left), float(params.inv_dy))
        else:
            supports = np.stack((left, right, left, left, left, left), axis=1)
            reference = np.zeros((len(left), 6))
            difference = np.broadcast_to(np.array([-1., 1., 0., 0., 0., 0.]), reference.shape).copy()
            inverse = np.ones(len(left))
        measure = area[left].copy()
        if axis == 1:
            latitude = (left // levels) % wet.shape[1]
            measure *= .5 * (cosine[latitude] + cosine[latitude + 1]) / cosine[latitude]
        values = (left, right, np.full(len(left), axis), supports, reference, difference, inverse, measure)
        for collection, value in zip(face_lists, values, strict=True):
            collection.append(value)
    left, right, axes, supports, reference, difference, inverse, measure = tuple(np.concatenate(values, axis=0) for values in face_lists)
    slots_left, slots_right = np.triu_indices(6, 1)
    potential = (reference != 0.) | (difference != 0.)
    opened = (potential[:, slots_left] & potential[:, slots_right]
              & (supports[:, slots_left] != supports[:, slots_right]))
    term_faces = np.broadcast_to(np.arange(len(left))[:, None], opened.shape)[opened]
    term_left = np.broadcast_to(slots_left, opened.shape)[opened]
    term_right = np.broadcast_to(slots_right, opened.shape)[opened]
    first = supports[term_faces, term_left]
    second = supports[term_faces, term_right]
    keys = np.minimum(first, second).astype(np.int64) * wet.size + np.maximum(first, second)
    unique, membership = np.unique(keys, return_inverse=True)
    pair_left, pair_right = unique // wet.size, unique % wet.size
    coefficient_groups = _gather_groups(membership, np.arange(len(keys)))
    edge_ids = np.arange(len(unique))
    incidence_owners = np.concatenate((pair_left, pair_right))
    incidence_edges = np.concatenate((edge_ids, edge_ids))
    incidence_signs = np.concatenate((np.ones(len(unique)), -np.ones(len(unique))))
    incidence_neighbors = np.concatenate((pair_right, pair_left))
    incident = _gather_groups(incidence_owners, np.arange(len(incidence_owners)))
    groups = tuple(IncidenceGroup(group.owners, jnp.asarray(incidence_edges[np.asarray(group.entries)], dtype=jnp.int32),
                                  jnp.asarray(incidence_signs[np.asarray(group.entries)]),
                                  jnp.asarray(incidence_neighbors[np.asarray(group.entries)], dtype=jnp.int32)) for group in incident)
    return DiffusionGraph(wet.shape, jnp.asarray(wet.ravel()), jnp.asarray(area),
                           jnp.asarray(left), jnp.asarray(right), jnp.asarray(axes), jnp.asarray(supports),
                           jnp.asarray(reference), jnp.asarray(difference), jnp.asarray(inverse), jnp.asarray(measure),
                           jnp.asarray(term_faces, dtype=jnp.int32), jnp.asarray(term_left, dtype=jnp.int32),
                           jnp.asarray(term_right, dtype=jnp.int32), jnp.asarray(pair_left, dtype=jnp.int32),
                           jnp.asarray(pair_right, dtype=jnp.int32), coefficient_groups, groups)


def edge_coefficients(graph, geometry, kappa_h, kappa_v):
    """Coalesce complete physical off-diagonal coefficients before limiting."""
    depth = geometry.node_depth.ravel()
    thickness = geometry.thickness.ravel()
    scale = geometry.scale.ravel()
    gap = depth[graph.right] - depth[graph.left]
    vertical = graph.axis == 2
    safe_gap = jnp.where(vertical & (gap > 0.), gap, 1.)
    inverse = jnp.where(vertical, 1. / safe_gap, graph.inverse_horizontal_spacing)
    derivatives = graph.reference_derivative / scale[graph.supports // graph.shape[-1]]
    coefficients = (graph.difference - jnp.where(vertical, 0., .5 * gap)[:, None] * derivatives) * inverse[:, None]
    horizontal = jnp.broadcast_to(jnp.asarray(kappa_h), graph.shape).ravel()
    vertical_coefficient = jnp.broadcast_to(jnp.asarray(kappa_v), graph.shape).ravel()
    selected = jnp.where(vertical[:, None], jnp.stack((vertical_coefficient[graph.left], vertical_coefficient[graph.right]), axis=1),
                         jnp.stack((horizontal[graph.left], horizontal[graph.right]), axis=1))
    width = jnp.where(vertical, gap, .5 * (thickness[graph.left] + thickness[graph.right]))
    weights = graph.face_measure * width * jnp.mean(selected, axis=1)
    raw = (-weights[graph.term_faces] * coefficients[graph.term_faces, graph.term_left]
           * coefficients[graph.term_faces, graph.term_right])
    coalesced = jnp.zeros(graph.pair_left.shape, dtype=thickness.dtype)
    for group in graph.coefficient_groups:
        coalesced = coalesced.at[group.owners].set(jnp.sum(raw[group.entries], axis=1))
    coefficient_valid = jnp.all(jnp.where(graph.wet, jnp.isfinite(horizontal) & (horizontal >= 0.)
                                          & jnp.isfinite(vertical_coefficient) & (vertical_coefficient >= 0.), True))
    valid = (geometry.valid & coefficient_valid & jnp.all(jnp.isfinite(selected)) & jnp.all(selected >= 0.)
             & jnp.all(jnp.isfinite(coalesced)) & jnp.all(jnp.where(vertical, gap > 0., True)))
    return coalesced, valid


def accumulate_edges(graph, values, mode="signed"):
    """Fixed-index gather reductions followed by disjoint node writes."""
    result = jnp.zeros(graph.wet.shape, dtype=values.dtype)
    for group in graph.incidence_groups:
        amounts = values[group.edges]
        if mode != "unsigned":
            amounts = amounts * group.signs
        if mode == "positive":
            amounts = jnp.maximum(amounts, 0.)
        elif mode == "negative":
            amounts = jnp.minimum(amounts, 0.)
        result = result.at[group.owners].set(jnp.sum(amounts, axis=1))
    return result


def neighbor_bounds(graph, concentration, coefficients):
    lower, upper = concentration, concentration
    for group in graph.incidence_groups:
        adjacent = jnp.where(coefficients[group.edges] != 0., concentration[group.neighbors], concentration[group.owners, None])
        lower = lower.at[group.owners].set(jnp.minimum(concentration[group.owners], jnp.min(adjacent, axis=1)))
        upper = upper.at[group.owners].set(jnp.maximum(concentration[group.owners], jnp.max(adjacent, axis=1)))
    return lower, upper


def graph_volume_rhs(concentration, coefficients, graph):
    safe = jnp.where(graph.wet, concentration.ravel(), 0.)
    flux = coefficients * (safe[graph.pair_right] - safe[graph.pair_left])
    return accumulate_edges(graph, flux).reshape(graph.shape)


def limited_negative_exchange(graph, negative_flux, low_content, lower_content, upper_content):
    """Conservative paired correction using explicitly supplied content bounds."""
    positive_amount = accumulate_edges(graph, negative_flux, "positive")
    negative_amount = accumulate_edges(graph, negative_flux, "negative")
    allowance_positive = jnp.maximum(upper_content - low_content, 0.)
    allowance_negative = jnp.maximum(low_content - lower_content, 0.)
    ratio_positive = jnp.minimum(1., allowance_positive / jnp.where(positive_amount > 0., positive_amount, 1.))
    ratio_negative = jnp.minimum(1., allowance_negative / jnp.where(negative_amount < 0., -negative_amount, 1.))
    limiter = jnp.where(negative_flux >= 0., jnp.minimum(ratio_positive[graph.pair_left], ratio_negative[graph.pair_right]),
                        jnp.minimum(ratio_negative[graph.pair_left], ratio_positive[graph.pair_right]))
    limiter = jnp.where(negative_flux == 0., 1., limiter)
    minimum = jnp.min(limiter) if graph.pair_left.shape[0] else jnp.asarray(1., dtype=low_content.dtype)
    return accumulate_edges(graph, negative_flux * limiter), minimum


def bounded_euler(content, geometry, coefficients, coefficient_valid, duration, graph):
    """One fixed-mass conservative limited exchange with checked rollback."""
    if content.dtype != jnp.float64 or content.shape != graph.shape:
        raise ValueError("bounded sparse content requires the frozen float64 grid")
    mass = graph.area * geometry.thickness.ravel()
    denominator = jnp.where(graph.wet & (mass > 0.), mass, 1.)
    original = content.ravel()
    concentration = jnp.where(graph.wet, original / denominator, 0.)
    difference = concentration[graph.pair_right] - concentration[graph.pair_left]
    positive_coefficient = jnp.maximum(coefficients, 0.)
    low_flux = duration * positive_coefficient * difference
    low_change = accumulate_edges(graph, low_flux)
    low_content = original + low_change
    negative_flux = duration * jnp.minimum(coefficients, 0.) * difference
    lower, upper = neighbor_bounds(graph, concentration, coefficients)
    correction, minimum = limited_negative_exchange(graph, negative_flux, low_content, mass * lower, mass * upper)
    exchange = low_change + correction
    attempted = original + exchange
    attempted_concentration = attempted / denominator
    outflow = accumulate_edges(graph, positive_coefficient, "unsigned")
    fraction = jnp.max(jnp.where(graph.wet, duration * outflow / denominator, 0.))
    floor = 64. * jnp.finfo(original.dtype).eps * (1. + jnp.abs(concentration) + jnp.abs(attempted_concentration))
    valid = (coefficient_valid & geometry.valid & jnp.isfinite(duration) & (duration > 0.) & (fraction <= .5)
             & jnp.all(jnp.where(graph.wet, jnp.isfinite(original) & (mass > 0.), True))
             & jnp.all(jnp.where(graph.wet, jnp.isfinite(attempted)
                                 & (attempted_concentration >= lower - floor) & (attempted_concentration <= upper + floor), True)))
    selected = jnp.where(valid, attempted, original).reshape(graph.shape)
    return BoundedDiffusionResult(selected, attempted.reshape(graph.shape), valid, fraction, minimum, exchange.reshape(graph.shape))


def bounded_heun(content, geometry, coefficients, coefficient_valid, duration, graph):
    """Fixed-geometry SSP Heun, retaining either-stage refusal and all dry bytes."""
    first = bounded_euler(content, geometry, coefficients, coefficient_valid, duration, graph)
    second = bounded_euler(first.content, geometry, coefficients, coefficient_valid, duration, graph)
    attempted = content + .5 * (first.exchange + second.exchange)
    valid = first.valid & second.valid & jnp.all(jnp.where(graph.wet.reshape(graph.shape), jnp.isfinite(attempted), True))
    return BoundedDiffusionResult(jnp.where(valid, attempted, content), attempted, valid,
                                  jnp.maximum(first.fraction, second.fraction),
                                  jnp.minimum(first.minimum_limiter, second.minimum_limiter), .5 * (first.exchange + second.exchange))
