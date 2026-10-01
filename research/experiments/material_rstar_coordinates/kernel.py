"""Isolated nodal r-star geometry and metric controls; no production caller."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from config import G_EARTH, RHO_0


class RStarGeometry(NamedTuple):
    scale: jax.Array
    thickness: jax.Array
    node_depth: jax.Array
    interface_depth: jax.Array
    node_spacing: jax.Array
    weights: jax.Array
    valid: jax.Array


class VerticalStencil(NamedTuple):
    indices: jax.Array
    coefficients: jax.Array


def rstar_geometry(eta, reference_depth, reference_width, wet):
    """Stretch between the actual surface and fixed bed, retaining invalid mass."""
    active = wet > 0.
    width = jnp.where(active, jnp.broadcast_to(reference_width, wet.shape), 0.)
    depth = jnp.sum(width, axis=-1)
    column = jnp.any(active, axis=-1)
    surface = jnp.where(column, eta, 0.)
    denominator = jnp.where(depth > 0., depth, 1.)
    scale = jnp.where(column, 1. + surface / denominator, 1.)
    thickness = width * scale[..., None]
    nodes = -surface[..., None] + scale[..., None] * reference_depth
    interfaces = jnp.concatenate((jnp.zeros_like(depth[..., None]), jnp.cumsum(width, axis=-1)), axis=-1)
    interfaces = -surface[..., None] + scale[..., None] * interfaces
    spacing = scale[..., None] * jnp.diff(reference_depth)
    connected = active[..., :-1] & active[..., 1:]
    valid = (jnp.all(jnp.where(column, jnp.isfinite(surface) & (scale > 0.), True))
             & jnp.all(jnp.where(active, jnp.isfinite(thickness) & (width > 0.) & (thickness > 0.), True))
             & jnp.all(jnp.where(active, jnp.isfinite(nodes), True))
             & jnp.all(jnp.where(connected, jnp.isfinite(spacing) & (spacing > 0.), True)))
    return RStarGeometry(scale, thickness, jnp.where(active, nodes, 0.), interfaces,
                         spacing, width / denominator[..., None], valid)


def relative_vertical_transport(layer_divergence, geometry):
    """Downward relative transport with its unmodified bottom residual."""
    surface_rate = -jnp.sum(layer_divergence, axis=-1)
    increment = -layer_divergence - geometry.weights * surface_rate[..., None]
    return jnp.concatenate((jnp.zeros_like(surface_rate[..., None]), jnp.cumsum(increment, axis=-1)), axis=-1)


def make_reference_stencil(reference_depth, wet):
    """Three-point physical-depth derivative without dry sentinel reads."""
    depths = np.asarray(reference_depth, dtype=np.float64)
    mask = np.asarray(wet)
    if depths.ndim != 1 or len(depths) < 2 or depths[0] != 0. or not np.all(np.diff(depths) > 0.):
        raise ValueError("reference nodes must start at zero and increase downward")
    if mask.ndim != 3 or mask.shape[-1] != len(depths) or not np.all((mask == 0.) | (mask == 1.)):
        raise ValueError("wet mask must be binary and match the reference nodes")
    if np.any(np.diff(mask, axis=-1) > 0.):
        raise ValueError("wet columns must be contiguous from the surface")
    indices = np.zeros(mask.shape + (3,), dtype=np.int32)
    coefficients = np.zeros(mask.shape + (3,), dtype=np.float64)
    for column in np.ndindex(mask.shape[:2]):
        count = int(np.sum(mask[column]))
        if count == 1:
            raise ValueError("a derivative needs at least two wet nodes")
        for level in range(count):
            if count == 2:
                selected = np.arange(2)
                weights = np.array([-1., 1.]) / (depths[1] - depths[0])
            else:
                start = min(max(level - 1, 0), count - 3)
                selected = np.arange(start, start + 3)
                offsets = depths[selected] - depths[level]
                matrix = np.stack((np.ones(3), offsets, offsets ** 2))
                weights = np.linalg.solve(matrix, np.array([0., 1., 0.]))
            indices[column + (level, slice(0, len(selected)))] = selected
            coefficients[column + (level, slice(0, len(selected)))] = weights
    return VerticalStencil(jnp.asarray(indices), jnp.asarray(coefficients))


def node_vertical_derivative(field, geometry, stencil, wet):
    safe = jnp.where(wet > 0., field, 0.)
    gathered = jnp.stack(tuple(jnp.take_along_axis(safe, stencil.indices[..., offset], axis=-1)
                               for offset in range(3)), axis=-1)
    derivative = jnp.sum(gathered * stencil.coefficients, axis=-1) / geometry.scale[..., None]
    return jnp.where(wet > 0., derivative, 0.)


def hydrostatic_pressure(rho_prime, eta, geometry, params):
    safe = jnp.where(params.wet_mask_z > 0., rho_prime, 0.)
    increment = G_EARTH * .5 * (safe[..., :-1] + safe[..., 1:]) * geometry.node_spacing
    column = jnp.concatenate((jnp.zeros_like(safe[..., :1]), jnp.cumsum(increment, axis=-1)), axis=-1)
    return column + RHO_0 * G_EARTH * jnp.where(params.wet_mask > 0., eta, 0.)[..., None]


def _face_gates(params):
    wet = params.wet_mask_z
    east = wet * jnp.roll(wet, -1, axis=0)
    north = (wet * jnp.roll(wet, -1, axis=1)).at[:, -1, :].set(0.)
    return east, north


def coordinate_pressure_gradient(pressure, rho_prime, geometry, params):
    """Face chain rule: subtract hydrostatic change due to coordinate slope."""
    safe = jnp.where(params.wet_mask_z > 0., rho_prime, 0.)
    differences = []
    for axis, gate in enumerate(_face_gates(params)):
        density = .5 * (safe + jnp.roll(safe, -1, axis=axis))
        slope = jnp.roll(geometry.node_depth, -1, axis=axis) - geometry.node_depth
        difference = jnp.roll(pressure, -1, axis=axis) - pressure - G_EARTH * density * slope
        differences.append(difference * gate)
    east, north = differences
    zonal = params.inv_dx * .5 * (east + jnp.roll(east, 1, axis=0))
    cosine_face = .5 * (params.cos_lat + jnp.roll(params.cos_lat, -1))
    north = north * cosine_face[None, :, None]
    south = jnp.roll(north, 1, axis=1).at[:, 0, :].set(0.)
    meridional = (params.inv_dy / params.cos_lat[None, :, None]) * .5 * (north + south)
    return -zonal / RHO_0, -meridional / RHO_0


def physical_diffusion_gradients(field, geometry, stencil, params):
    safe = jnp.where(params.wet_mask_z > 0., field, 0.)
    derivative = node_vertical_derivative(safe, geometry, stencil, params.wet_mask_z)
    gradients = []
    for axis, gate in enumerate(_face_gates(params)):
        slope = jnp.roll(geometry.node_depth, -1, axis=axis) - geometry.node_depth
        average_derivative = .5 * (derivative + jnp.roll(derivative, -1, axis=axis))
        difference = jnp.roll(safe, -1, axis=axis) - safe - average_derivative * slope
        inverse_spacing = params.inv_dx if axis == 0 else params.inv_dy
        gradients.append(difference * inverse_spacing * gate)
    connected = params.wet_mask_z[..., :-1] * params.wet_mask_z[..., 1:]
    spacing = jnp.where(connected > 0., geometry.node_spacing, 1.)
    gradients.append(jnp.diff(safe, axis=-1) / spacing * connected)
    return tuple(gradients)


def diffusion_weights(geometry, params, kappa_h, kappa_v):
    area = params.dx_2d * params.dy
    horizontal = jnp.broadcast_to(jnp.asarray(kappa_h), params.wet_mask_z.shape)
    vertical = jnp.broadcast_to(jnp.asarray(kappa_v), params.wet_mask_z.shape)
    east, north = _face_gates(params)
    weights = []
    for axis, gate in enumerate((east, north)):
        thickness = .5 * (geometry.thickness + jnp.roll(geometry.thickness, -1, axis=axis))
        coefficient = .5 * (horizontal + jnp.roll(horizontal, -1, axis=axis))
        measure = area[..., None]
        if axis == 1:
            cosine_face = .5 * (params.cos_lat + jnp.roll(params.cos_lat, -1))
            measure = measure * (cosine_face / params.cos_lat)[None, :, None]
        weights.append(measure * thickness * coefficient * gate)
    connected = params.wet_mask_z[..., :-1] * params.wet_mask_z[..., 1:]
    coefficient = .5 * (vertical[..., :-1] + vertical[..., 1:])
    weights.append(area[..., None] * geometry.node_spacing * coefficient * connected)
    return tuple(weights)


def diffusion_content_rhs(field, geometry, stencil, params, kappa_h, kappa_v):
    """Negative adjoint of the same physical gradients, including metric terms."""
    def gradient(values):
        return physical_diffusion_gradients(values, geometry, stencil, params)

    gradients = gradient(field)
    weights = diffusion_weights(geometry, params, kappa_h, kappa_v)
    adjoint = jax.linear_transpose(gradient, jnp.zeros_like(field))
    volume_rate = adjoint(tuple(-weight * value for weight, value in zip(weights, gradients, strict=True)))[0]
    return volume_rate / (params.dx_2d * params.dy)[..., None]
