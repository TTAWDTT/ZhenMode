"""Explicit changed nodal boundary contract, retaining the original physical bed."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from research.experiments.material_rstar_coordinates.nodal_mass import make_nodal_mass

REPRESENTATION_TAG = "bed_complete_nodal_v1"


class BedReference(NamedTuple):
    nodes: jax.Array
    widths: jax.Array
    wet: jax.Array
    added: jax.Array
    bed: jax.Array


def complete_bed_reference(depths, params):
    make_nodal_mass(depths, params)
    wet = np.asarray(params.wet_mask_z) > 0.
    nodes = np.broadcast_to(np.asarray(depths), wet.shape).copy()
    widths = np.asarray(params.dz_node) * wet
    bed = widths.sum(axis=-1)
    added = np.zeros(wet.shape, dtype=bool)
    for column in np.ndindex(wet.shape[:2]):
        count = int(wet[column].sum())
        if not count:
            continue
        if bed[column] > nodes[column + (count - 1,)]:
            if count == wet.shape[-1]:
                raise ValueError("bed completion needs a spare dry slot; cannot alter shape or bed")
            nodes[column + (count,)] = bed[column]
            wet[column + (count,)] = True
            added[column + (count,)] = True
    nodes = np.where(wet, nodes, 0.)
    gap = np.diff(nodes, axis=-1) * wet[..., :-1] * wet[..., 1:]
    widths = .5 * (np.concatenate((gap, np.zeros(wet.shape[:-1] + (1,))), axis=-1)
                   + np.concatenate((np.zeros(wet.shape[:-1] + (1,)), gap), axis=-1))
    floor = 64. * np.finfo(float).eps * np.maximum(bed, 1.)
    if np.any(np.abs(widths.sum(axis=-1) - bed) > floor) or np.any((widths <= 0.) & wet):
        raise ValueError("completed nodes require the same bed and positive linear-hat mass")
    return BedReference(jnp.asarray(nodes), jnp.asarray(widths), jnp.asarray(wet.astype(float)),
                        jnp.asarray(added), jnp.asarray(bed))
