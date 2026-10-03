"""Original-node pressure-work candidates; no accepted ocean-step interface."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ocean_solver.config.definitions import G_EARTH, RHO_0
from ocean_solver.dynamics.transport import _face_transport_divergence
from research.experiments.material_rstar_coordinates.kernel import (
    _face_gates,
    relative_vertical_transport,
)


class PotentialBasis(NamedTuple):
    mean_depth: jax.Array
    column_depth: jax.Array


def make_potential_basis(depths, params):
    """Exact nodal hat first moments, with constant extension at the wet bed."""
    wet = np.asarray(params.wet_mask_z) > 0.
    nodes = np.broadcast_to(np.asarray(depths), wet.shape)
    widths = np.broadcast_to(np.asarray(params.dz_node), wet.shape) * wet
    depth = widths.sum(axis=-1)
    connected = wet[..., :-1] & wet[..., 1:]
    gap = np.diff(nodes, axis=-1) * connected
    left = .5 * gap * nodes[..., :-1] + gap ** 2 / 6.
    right = .5 * gap * nodes[..., 1:] - gap ** 2 / 6.
    moments = np.concatenate((left, np.zeros(wet.shape[:-1] + (1,))), axis=-1)
    moments += np.concatenate((np.zeros(wet.shape[:-1] + (1,)), right), axis=-1)
    bottom = wet & ~np.concatenate((wet[..., 1:], np.zeros(wet.shape[:-1] + (1,), dtype=bool)), axis=-1)
    extension = np.where(bottom, depth[..., None] - nodes, 0.)
    moments += extension * nodes + .5 * extension ** 2
    actual_width = np.concatenate((.5 * gap, np.zeros(wet.shape[:-1] + (1,))), axis=-1)
    actual_width += np.concatenate((np.zeros(wet.shape[:-1] + (1,)), .5 * gap), axis=-1) + extension
    floor = 64. * np.finfo(float).eps * np.maximum(widths, 1.)
    if (np.any(np.diff(wet.astype(int), axis=-1) > 0) or np.any(extension < 0.)
            or np.any((widths <= 0.) & wet) or np.any(np.abs(actual_width - widths) > floor)):
        raise ValueError("potential basis requires matching contiguous nodal hat mass and declared bed")
    return PotentialBasis(jnp.asarray(np.where(wet, moments / np.where(widths > 0., widths, 1.), 0.)), jnp.asarray(depth))


def coordinate_layer_faces(velocity_x, velocity_y, geometry, params):
    gates = _face_gates(params)
    thickness = tuple(.5 * (geometry.thickness + jnp.roll(geometry.thickness, -1, axis=axis)) * gate
                      for axis, gate in enumerate(gates))
    east = .5 * (velocity_x + jnp.roll(velocity_x, -1, axis=0)) * thickness[0]
    cosine = .5 * (params.cos_lat + jnp.roll(params.cos_lat, -1))
    north = .5 * (velocity_y + jnp.roll(velocity_y, -1, axis=1)) * thickness[1] * cosine[None, :, None]
    return east, north


def centered_content_rate(density, faces, geometry, params):
    """Reversible diagnostic only; original vertical donor is not replaced."""
    safe = jnp.where(params.wet_mask_z > 0., density, 0.)
    horizontal = tuple(flux * .5 * (safe + jnp.roll(safe, -1, axis=axis)) for axis, flux in enumerate(faces))
    divergence = _face_transport_divergence(*faces, params)
    relative = relative_vertical_transport(divergence, geometry)
    internal = relative[..., 1:-1] * .5 * (safe[..., :-1] + safe[..., 1:]) * params.wet_mask_z[..., :-1] * params.wet_mask_z[..., 1:]
    upper = jnp.concatenate((jnp.zeros_like(internal[..., :1]), internal), axis=-1)
    lower = jnp.concatenate((internal, jnp.zeros_like(internal[..., :1])), axis=-1)
    area = (params.dx_2d * params.dy)[..., None]
    return -area * (_face_transport_divergence(*horizontal, params) + lower - upper), -jnp.sum(divergence, axis=-1)


def potential_conjugates(content, surface, geometry, basis, params):
    area = params.dx_2d * params.dy
    active_surface = jnp.where(params.wet_mask > 0., surface, 0.)
    mean = -active_surface[..., None] + geometry.scale[..., None] * basis.mean_depth
    safe = jnp.where(params.wet_mask_z > 0., content, 0.)
    derivative = -1. + basis.mean_depth / jnp.where(basis.column_depth > 0., basis.column_depth, 1.)[..., None]
    density_conjugate = jnp.where(params.wet_mask_z > 0., -G_EARTH * mean, 0.)
    surface_conjugate = RHO_0 * G_EARTH * area * active_surface - G_EARTH * jnp.sum(safe * derivative, axis=-1)
    return density_conjugate, surface_conjugate


def potential_energy(content, surface, geometry, basis, params):
    active_surface = jnp.where(params.wet_mask > 0., surface, 0.)
    mean = -active_surface[..., None] + geometry.scale[..., None] * basis.mean_depth
    safe = jnp.where(params.wet_mask_z > 0., content, 0.)
    return .5 * RHO_0 * G_EARTH * jnp.sum(params.dx_2d * params.dy * active_surface ** 2) - G_EARTH * jnp.sum(safe * mean)


def force_from_face_jumps(jumps, geometry, params):
    """Actual-mass negative adjoint of the declared coordinate-band transports."""
    gates = _face_gates(params)
    weighted = tuple(jump * .5 * (geometry.thickness + jnp.roll(geometry.thickness, -1, axis=axis)) * gate
                     for axis, (jump, gate) in enumerate(zip(jumps, gates, strict=True)))
    east, north = weighted
    cosine = .5 * (params.cos_lat + jnp.roll(params.cos_lat, -1))
    north = north * cosine[None, :, None]
    south = jnp.roll(north, 1, axis=1).at[:, 0].set(0.)
    thickness = jnp.where(params.wet_mask_z > 0., geometry.thickness, 1.)
    zonal = -params.inv_dx * .5 * (east + jnp.roll(east, 1, axis=0)) / (RHO_0 * thickness)
    meridional = -params.inv_dy / params.cos_lat[None, :, None] * .5 * (north + south) / (RHO_0 * thickness)
    return zonal * params.wet_mask_z, meridional * params.wet_mask_z


def paired_chain_rule_force(pressure, density, geometry, params):
    safe = jnp.where(params.wet_mask_z > 0., density, 0.)
    jumps = tuple(jnp.roll(pressure, -1, axis=axis) - pressure
                  - G_EARTH * .5 * (safe + jnp.roll(safe, -1, axis=axis))
                  * (jnp.roll(geometry.node_depth, -1, axis=axis) - geometry.node_depth) for axis in (0, 1))
    return force_from_face_jumps(jumps, geometry, params)


def energy_adjoint_force(density, content, surface, geometry, basis, params):
    """Work-paired candidate; physical resting-state qualification is separate."""
    safe = jnp.where(params.wet_mask_z > 0., density, 0.)
    density_conjugate, surface_conjugate = potential_conjugates(content, surface, geometry, basis, params)
    connected = params.wet_mask_z[..., :-1] * params.wet_mask_z[..., 1:]
    vertical = .5 * (safe[..., :-1] + safe[..., 1:]) * jnp.diff(density_conjugate, axis=-1) * connected
    suffix = jnp.concatenate((jnp.flip(jnp.cumsum(jnp.flip(vertical, axis=-1), axis=-1), axis=-1), jnp.zeros_like(vertical[..., :1])), axis=-1)
    prefix = jnp.cumsum(geometry.weights, axis=-1)[..., :-1]
    primitive = -suffix + jnp.sum(prefix * vertical, axis=-1, keepdims=True)
    surface_load = surface_conjugate / (params.dx_2d * params.dy)
    jumps = tuple(.5 * (safe + jnp.roll(safe, -1, axis=axis)) * (jnp.roll(density_conjugate, -1, axis=axis) - density_conjugate)
                  + primitive - jnp.roll(primitive, -1, axis=axis)
                  + (jnp.roll(surface_load, -1, axis=axis) - surface_load)[..., None] for axis in (0, 1))
    return force_from_face_jumps(jumps, geometry, params)


def transpose_force(density, conjugates, geometry, params):
    """Autodiff work oracle, independent of the explicit backward primitive."""
    def transport(velocities):
        faces = coordinate_layer_faces(*velocities, geometry, params)
        return centered_content_rate(density, faces, geometry, params)

    zeros = jnp.zeros_like(density)
    adjoint = jax.linear_transpose(transport, (zeros, zeros))(conjugates)[0]
    mass = (params.dx_2d * params.dy)[..., None] * geometry.thickness
    return tuple(-force / (RHO_0 * jnp.where(params.wet_mask_z > 0., mass, 1.)) for force in adjoint)
