"""Physical weak-form sparse coefficient graph with bounded local overlap blocks."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from research.experiments.material_rstar_coordinates.sparse_diffusion import (
    IncidenceGroup,
    _gather_groups,
)
from research.experiments.material_rstar_coordinates.weak_transport import (
    physical_face_quadrature,
)


class WeakGraph(NamedTuple):
    wet: jax.Array
    pair_left: jax.Array
    pair_right: jax.Array
    vertical_entries: jax.Array
    east_entries: jax.Array
    north_entries: jax.Array
    incidence_groups: tuple


class WeakCoefficients(NamedTuple):
    diagonal: jax.Array
    forward: jax.Array
    reverse: jax.Array
    surface_rate: jax.Array


def make_weak_graph(params):
    mask = np.asarray(params.wet_mask_z)
    if (mask.ndim != 3 or mask.shape[0] < 3 or mask.shape[1] < 2 or mask.shape[-1] < 2
            or mask.size > np.iinfo(np.int32).max or not np.any(mask) or not np.all((mask == 0.) | (mask == 1.))
            or np.any(np.diff(mask, axis=-1) > 0.)):
        raise ValueError("weak graph requires representable binary contiguous wet nx>=3,ny>=2,nz>=2")
    dx, cosine, dy = np.asarray(params.dx_2d), np.asarray(params.cos_lat), float(params.dy)
    if (dx.shape != mask.shape[:2] or cosine.shape != (mask.shape[1],)
            or not np.all(np.isfinite(dx) & (dx > 0.)) or not np.all(np.isfinite(cosine) & (cosine > 0.))
            or not np.isfinite(dy) or dy <= 0. or not np.array_equal(mask[..., 0], np.asarray(params.wet_mask))):
        raise ValueError("weak graph requires finite positive matching original metrics and wet columns")
    wet = mask > 0.
    identifiers = np.arange(mask.size, dtype=np.int32).reshape(mask.shape)
    vertical = wet[..., :-1] & wet[..., 1:]
    entries = [np.flatnonzero(vertical)]
    left, right = [identifiers[..., :-1][vertical]], [identifiers[..., 1:][vertical]]
    for axis in (0, 1):
        other = np.roll(identifiers, -1, axis=axis)
        opened = wet[..., :, None] & np.roll(wet, -1, axis=axis)[..., None, :]
        if axis == 1:
            opened[:, -1] = False
        entries.append(np.flatnonzero(opened))
        left.append(np.broadcast_to(identifiers[..., :, None], opened.shape)[opened])
        right.append(np.broadcast_to(other[..., None, :], opened.shape)[opened])
    pair_left, pair_right = np.concatenate(left), np.concatenate(right)
    edge_ids = np.arange(len(pair_left), dtype=np.int32)
    owners = np.concatenate((pair_left, pair_right))
    edges = np.concatenate((edge_ids, edge_ids))
    neighbors = np.concatenate((pair_right, pair_left))
    signs = np.concatenate((np.ones(len(edge_ids)), -np.ones(len(edge_ids))))
    gathered = _gather_groups(owners, np.arange(len(owners)))
    groups = tuple(IncidenceGroup(group.owners, jnp.asarray(edges[np.asarray(group.entries)]),
                                  jnp.asarray(signs[np.asarray(group.entries)]),
                                  jnp.asarray(neighbors[np.asarray(group.entries)])) for group in gathered)
    return WeakGraph(jnp.asarray(wet.ravel()), jnp.asarray(pair_left), jnp.asarray(pair_right),
                     *(jnp.asarray(value, dtype=jnp.int32) for value in entries), groups)


def _band(trace, horizontal, vertical):
    weights, derivative = trace
    def integrate(first, second):
        return jnp.sum(horizontal[..., None] * first * second, axis=(-3, -2))
    def transport(first, second):
        return jnp.sum(vertical[..., None] * first * second, axis=(-3, -2))
    diagonal = integrate(weights, weights) + transport(derivative, weights)
    upper = integrate(weights[..., :-1], weights[..., 1:]) + transport(derivative[..., :-1], weights[..., 1:])
    lower = integrate(weights[..., 1:], weights[..., :-1]) + transport(derivative[..., 1:], weights[..., :-1])
    return diagonal, upper, lower


def _face_blocks(velocity, surface, geometry, params, axis):
    face = physical_face_quadrature(velocity, geometry, surface, params, axis)
    horizontal = .5 * face.measures * face.flux
    left = _band(face.left_trace, -horizontal, -face.measures * face.left_primitive * face.inside_left)
    right = _band(face.right_trace, horizontal, face.measures * face.right_primitive * face.inside_right)
    coupling = -jnp.einsum("...pq,...pqi,...pqj->...ij", horizontal, face.left_trace.weights, face.right_trace.weights)
    return left, right, coupling, face.total


def _streamed_face_blocks(velocity, surface, geometry, params, axis):
    columns_per_block = 16
    longitude_count = velocity.shape[0]
    count = (longitude_count + columns_per_block - 1) // columns_per_block

    def integrate_block(carry, start):
        indices = (start + jnp.arange(columns_per_block + 1)) % longitude_count
        local_geometry = type(geometry)(*(value[indices] if jnp.ndim(value) > 0 else value for value in geometry))
        local_params = params._replace(wet_mask_z=params.wet_mask_z[indices], wet_mask=params.wet_mask[indices],
                                        dz_node=jnp.broadcast_to(params.dz_node, params.wet_mask_z.shape)[indices], dx_2d=params.dx_2d[indices])
        left, right, coupling, total = _face_blocks(velocity[indices], surface[indices], local_geometry, local_params, axis)
        return carry, (tuple(value[:columns_per_block] for value in left),
                       tuple(value[:columns_per_block] for value in right), coupling[:columns_per_block], total[:columns_per_block])

    _, outputs = jax.lax.scan(integrate_block, None, jnp.arange(count) * columns_per_block)
    return jax.tree.map(lambda value: value.reshape((-1,) + value.shape[2:])[:longitude_count], outputs)


def weak_coefficients(velocities, surface, geometry, params, graph):
    safe_surface = jnp.where(params.wet_mask > 0., surface, 0.)
    diagonal = jnp.zeros_like(geometry.thickness)
    upper, lower = jnp.zeros_like(diagonal[..., :-1]), jnp.zeros_like(diagonal[..., :-1])
    volume = jnp.zeros_like(surface)
    cross = []
    for axis, velocity in enumerate(velocities):
        left, right, coupling, total = _streamed_face_blocks(velocity, safe_surface, geometry, params, axis)
        diagonal += left[0] + jnp.roll(right[0], 1, axis=axis)
        upper += left[1] + jnp.roll(right[1], 1, axis=axis)
        lower += left[2] + jnp.roll(right[2], 1, axis=axis)
        cross.append(coupling.ravel()[graph.east_entries if axis == 0 else graph.north_entries])
        volume += -total + jnp.roll(total, 1, axis=axis)
    forward = jnp.concatenate((upper.ravel()[graph.vertical_entries], *cross))
    reverse = jnp.concatenate((lower.ravel()[graph.vertical_entries], *(-value for value in cross)))
    return WeakCoefficients(diagonal.ravel(), forward, reverse, volume / (params.dx_2d * params.dy))


def directed_accumulation(graph, left, right):
    result = jnp.zeros(graph.wet.shape, dtype=left.dtype)
    for group in graph.incidence_groups:
        amounts = jnp.where(group.signs > 0., left[group.edges], right[group.edges])
        result = result.at[group.owners].set(jnp.sum(amounts, axis=1))
    return result


def weak_sparse_rate(concentration, coefficients, graph):
    safe = jnp.where(graph.wet, concentration.ravel(), 0.)
    coupled = directed_accumulation(graph, coefficients.forward * safe[graph.pair_right],
                                    coefficients.reverse * safe[graph.pair_left])
    return (coefficients.diagonal * safe + coupled).reshape(concentration.shape)


def mass_pair_entries(geometry, area, mass, graph):
    vertical = ((area * geometry.scale)[..., None] * mass.off_diagonal).ravel()[graph.vertical_entries]
    return jnp.concatenate((vertical, jnp.zeros(graph.east_entries.shape[0] + graph.north_entries.shape[0], dtype=vertical.dtype)))
