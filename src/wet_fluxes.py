"""Trace-compatible enriched shared-Q reconstruction on frozen wet prisms.

These are mapped face-integrated fluxes, not physical point velocities or a
complete nonlinear moving-surface momentum scheme. See wet_trace_protocol.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from bounded_transport import _neighbor


class WetFluxReconstruction(NamedTuple):
    top: jnp.ndarray
    height: jnp.ndarray
    side_top: jnp.ndarray
    side_bottom: jnp.ndarray
    side_density: jnp.ndarray
    vertical_top: jnp.ndarray
    vertical_bottom: jnp.ndarray
    net_flux: jnp.ndarray
    bottom_closure_relative: jnp.ndarray
    valid: jnp.ndarray


class WetFluxPoint(NamedTuple):
    east_per_depth: jnp.ndarray
    north_per_depth: jnp.ndarray
    vertical: jnp.ndarray
    divergence_per_depth: jnp.ndarray
    valid: jnp.ndarray


def _expand(field, dimensions):
    return jnp.reshape(field, field.shape + (1,) * (dimensions - field.ndim))


def evaluate_wet_flux(reconstruction, longitude_fraction, latitude_area_fraction, depth):
    """Evaluate mapped flux in xi, normalized sin(latitude), physical depth.

    Inputs are scalar or cell-shaped, optionally with trailing sample axes.
    Horizontal outputs are m2/s per reference transverse coordinate; vertical
    is m3/s per reference horizontal area. They are NOT point velocities.
    Bad coordinates return false per-query validity, never accepted clipping.
    """
    longitude, latitude, depth = (jnp.asarray(value, jnp.float64) for value in
                                   (longitude_fraction, latitude_area_fraction, depth))
    dimensions = max(reconstruction.top.ndim, longitude.ndim, latitude.ndim, depth.ndim)
    longitude, latitude, depth = (_expand(field, dimensions) for field in (longitude, latitude, depth))
    top, height = (_expand(field, dimensions) for field in (reconstruction.top, reconstruction.height))
    safe_height = jnp.where(height > 0., height, 1.)
    traces, primitives = [], []
    for side in range(4):
        start, end, density = (_expand(field[..., side], dimensions) for field in
                               (reconstruction.side_top, reconstruction.side_bottom, reconstruction.side_density))
        traces.append(jnp.where((end > start) & (depth >= start) & (depth <= end), density, 0.))
        primitives.append(density * jnp.clip(depth - start, 0., end - start))
    east = (1. - longitude) * traces[0] + longitude * traces[1]
    north = (1. - latitude) * traces[2] + latitude * traces[3]
    net_flux = _expand(reconstruction.net_flux, dimensions)
    vertical = (_expand(reconstruction.vertical_top, dimensions) + (depth - top) / safe_height * net_flux
                - primitives[1] + primitives[0] - primitives[3] + primitives[2])
    divergence = net_flux / safe_height
    inside = ((height > 0.) & (depth >= top) & (depth <= top + height)
              & (longitude >= 0.) & (longitude <= 1.) & (latitude >= 0.) & (latitude <= 1.)
              & jnp.isfinite(depth) & jnp.isfinite(longitude) & jnp.isfinite(latitude))
    valid = (inside & reconstruction.valid & jnp.isfinite(east) & jnp.isfinite(north)
             & jnp.isfinite(vertical) & jnp.isfinite(divergence))
    return WetFluxPoint(*(jnp.where(inside, field, 0.) for field in (east, north, vertical, divergence)), valid)


def reconstruct_wet_fluxes(faces, fluxes):
    """Use actual MomentumGeometry intervals and actual VolumeFluxes once.

    W/E/S/N traces vanish on blocked wedges and integrate to the input Q.
    A piecewise-linear vertical primitive matches both interface fluxes and
    the cell net divergence. It changes no state, inventory source or force.
    The caller MUST check valid; geometry/Q precision is explicitly64.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("wet flux reconstruction requires explicit JAX X64")
    shape = faces.height.shape
    geometry_fields = tuple(jnp.asarray(getattr(faces, name)) for name in
                            ("top", "height", "east_top", "east_bottom", "north_top", "north_bottom", "east_area", "north_area"))
    east, north, vertical = (jnp.asarray(field) for field in fluxes)
    if (len(shape) != 3 or any(field.shape != shape for field in geometry_fields)
            or east.shape != shape or north.shape != shape or vertical.shape != shape[:-1] + (shape[-1] + 1,)):
        raise ValueError("wet geometry and flux shape must match physical cells/interfaces")
    if any(field.dtype != jnp.float64 for field in (*geometry_fields, east, north, vertical)):
        raise ValueError("wet geometry and actual fluxes require64 dtype")
    top, height, east_top, east_bottom, north_top, north_bottom, east_area, north_area = geometry_fields
    opened = jnp.stack((_neighbor(east_area, 0, -1) > 0., east_area > 0.,
                        _neighbor(north_area, 1, -1) > 0., north_area > 0.), axis=-1)
    raw_top = jnp.stack((_neighbor(east_top, 0, -1), east_top, _neighbor(north_top, 1, -1), north_top), axis=-1)
    raw_bottom = jnp.stack((_neighbor(east_bottom, 0, -1), east_bottom, _neighbor(north_bottom, 1, -1), north_bottom), axis=-1)
    side_top = jnp.where(opened, raw_top, top[..., None])
    side_bottom = jnp.where(opened, raw_bottom, top[..., None])
    side_height = side_bottom - side_top
    side_flux = jnp.stack((_neighbor(east, 0, -1), east, _neighbor(north, 1, -1), north), axis=-1)
    side_density = jnp.where(side_height > 0., side_flux / jnp.where(side_height > 0., side_height, 1.), 0.)
    vertical_top, vertical_bottom = vertical[..., :-1], vertical[..., 1:]
    net_flux = side_flux[..., 1] - side_flux[..., 0] + side_flux[..., 3] - side_flux[..., 2] + vertical_bottom - vertical_top
    floor = 64. * jnp.finfo(jnp.float64).eps * (jnp.abs(top) + jnp.abs(top + height) + height)
    intervals_valid = jnp.all(jnp.where(opened,
                                      (side_height > 0.) & (side_top >= top[..., None] - floor[..., None])
                                      & (side_bottom <= (top + height)[..., None] + floor[..., None]), side_flux == 0.))
    valid = (faces.valid & intervals_valid & jnp.all(height >= 0.)
             & jnp.all(jnp.where(height > 0., True, (vertical_top == 0.) & (vertical_bottom == 0.)))
             & jnp.all(north[:, -1] == 0.) & jnp.all(vertical[..., 0] == 0.) & jnp.all(vertical[..., -1] == 0.)
             & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(field)) for field in (*geometry_fields, east, north, vertical))))
             & jnp.all(east_area >= 0.) & jnp.all(north_area >= 0.) & jnp.all(jnp.isfinite(side_density)))
    result = WetFluxReconstruction(top, height, side_top, side_bottom, side_density, vertical_top, vertical_bottom,
                                   net_flux, jnp.asarray(0., jnp.float64), valid)
    endpoint = evaluate_wet_flux(result, .5, .5, top + height).vertical
    scale = jnp.sum(jnp.abs(side_flux), axis=-1) + jnp.abs(vertical_top) + jnp.abs(vertical_bottom)
    error = jnp.abs(endpoint - vertical_bottom)
    relative = jnp.max(error / jnp.where(scale > 0., scale, 1.))
    return result._replace(bottom_closure_relative=relative, valid=valid & jnp.all(error <= (1e-12 + 64. * jnp.finfo(jnp.float64).eps) * scale))
