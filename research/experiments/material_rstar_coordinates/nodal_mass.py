"""Batched consistent nodal mass algebra, not a qualified transport operator."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from research.experiments.material_rstar_coordinates.pressure_work import make_potential_basis


class NodalMass(NamedTuple):
    diagonal: jax.Array
    off_diagonal: jax.Array
    wet: jax.Array


def make_nodal_mass(depths, params):
    depths = np.asarray(depths, dtype=np.float64)
    mask = np.asarray(params.wet_mask_z)
    if (depths.ndim not in (1, 3) or not depths.size or not np.all(np.isfinite(depths))):
        raise ValueError("nodal mass requires finite increasing reference depths starting at zero")
    if (mask.ndim != 3 or mask.shape[-1] != depths.shape[-1]
            or (depths.ndim == 3 and depths.shape != mask.shape) or not np.all((mask == 0.) | (mask == 1.))):
        raise ValueError("nodal mass requires matching binary wet mask")
    nodes = np.broadcast_to(depths, mask.shape)
    active = mask > 0.
    if (np.any((nodes[..., 0] != 0.) & active[..., 0])
            or np.any((np.diff(nodes, axis=-1) <= 0.) & active[..., :-1] & active[..., 1:])):
        raise ValueError("nodal mass requires finite increasing reference depths starting at zero")
    if not np.all(np.isfinite(np.asarray(params.dz_node))):
        raise ValueError("nodal mass requires finite reference widths")
    make_potential_basis(depths, params)
    wet = mask > 0.
    width = np.asarray(params.dz_node) * wet
    connected = wet[..., :-1] & wet[..., 1:]
    gap = np.diff(nodes, axis=-1) * connected
    diagonal = np.concatenate((gap / 3., np.zeros(wet.shape[:-1] + (1,))), axis=-1)
    diagonal += np.concatenate((np.zeros(wet.shape[:-1] + (1,)), gap / 3.), axis=-1)
    bottom = wet & ~np.concatenate((wet[..., 1:], np.zeros(wet.shape[:-1] + (1,), dtype=bool)), axis=-1)
    diagonal += np.where(bottom, width.sum(axis=-1, keepdims=True) - nodes, 0.)
    if np.any((diagonal <= 0.) & wet):
        raise ValueError("consistent nodal mass requires positive active diagonal")
    return NodalMass(jnp.asarray(diagonal), jnp.asarray(gap / 6.), jnp.asarray(wet))


def apply_nodal_mass(field, geometry, area, mass):
    safe = jnp.where(mass.wet, field, 0.)
    upper = mass.off_diagonal * safe[..., 1:]
    lower = mass.off_diagonal * safe[..., :-1]
    reference = (mass.diagonal * safe
                 + jnp.concatenate((upper, jnp.zeros_like(safe[..., :1])), axis=-1)
                 + jnp.concatenate((jnp.zeros_like(safe[..., :1]), lower), axis=-1))
    return area[..., None] * geometry.scale[..., None] * reference


def solve_nodal_mass(content, geometry, area, mass):
    """Thomas scans on independent columns; caller must reject invalid geometry."""
    factor = jnp.where(jnp.any(mass.wet, axis=-1), area * geometry.scale, 1.)
    rhs = jnp.where(mass.wet, content, 0.) / factor[..., None]
    diagonal = jnp.where(mass.wet, mass.diagonal, 1.)
    zeros = jnp.zeros_like(rhs[..., :1])
    lower = jnp.concatenate((zeros, mass.off_diagonal), axis=-1)
    upper = jnp.concatenate((mass.off_diagonal, zeros), axis=-1)

    def eliminate(previous, values):
        previous_upper, previous_rhs = previous
        below, center, above, value = values
        pivot = center - below * previous_upper
        updated = above / pivot, (value - below * previous_rhs) / pivot
        return updated, updated

    initial = jnp.zeros_like(rhs[..., 0]), jnp.zeros_like(rhs[..., 0])
    _, eliminated = jax.lax.scan(eliminate, initial, tuple(jnp.moveaxis(value, -1, 0) for value in (lower, diagonal, upper, rhs)))

    def substitute(next_value, values):
        above, value = values
        current = value - above * next_value
        return current, current

    _, solved = jax.lax.scan(substitute, initial[0], eliminated, reverse=True)
    return jnp.where(mass.wet, jnp.moveaxis(solved, 0, -1), 0.)
