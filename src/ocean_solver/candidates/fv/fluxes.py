"""Trace-compatible enriched shared-Q reconstruction on frozen wet prisms.

These are mapped face-integrated fluxes, not physical point velocities or a
complete nonlinear moving-surface momentum scheme. See wet_trace_protocol.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ocean_solver._compat import preserve_legacy_names
from ocean_solver.candidates.fv.geometry import VolumeFluxes
from ocean_solver.candidates.fv.transport import _neighbor


class WetFluxReconstruction(NamedTuple):
    top: jnp.ndarray
    height: jnp.ndarray
    side_top: jnp.ndarray
    side_bottom: jnp.ndarray
    side_density: jnp.ndarray
    vertical_top: jnp.ndarray
    vertical_bottom: jnp.ndarray
    net_flux: jnp.ndarray
    south_latitude: jnp.ndarray
    north_latitude: jnp.ndarray
    bottom_closure_relative: jnp.ndarray
    valid: jnp.ndarray


class WetFluxPoint(NamedTuple):
    east_per_depth: jnp.ndarray
    north_per_depth: jnp.ndarray
    vertical: jnp.ndarray
    divergence_per_depth: jnp.ndarray
    valid: jnp.ndarray


class HalfPrismTransport(NamedTuple):
    east_volume: jnp.ndarray
    north_volume: jnp.ndarray
    east_fluxes: VolumeFluxes
    north_fluxes: VolumeFluxes
    commutation_relative: jnp.ndarray
    valid: jnp.ndarray


def _expand(field, dimensions):
    return jnp.reshape(field, field.shape + (1,) * (dimensions - field.ndim))


def _latitude_metric(reconstruction, fraction, dimensions):
    south = _expand(reconstruction.south_latitude, dimensions)
    north = _expand(reconstruction.north_latitude, dimensions)
    width = north - south
    sine_width = jnp.sin(north) - jnp.sin(south)
    latitude = jnp.arcsin(jnp.sin(south) + fraction * sine_width)
    weight = sine_width / (width * jnp.cos(latitude))
    primitive = (latitude - south) / width
    return weight, primitive


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
    weight, latitude_primitive = _latitude_metric(reconstruction, latitude, dimensions)
    east = weight * ((1. - longitude) * traces[0] + longitude * traces[1])
    north = ((1. - latitude) * traces[2] + latitude * traces[3]
             + (traces[1] - traces[0]) * (latitude - latitude_primitive))
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


def reconstruct_wet_fluxes(geometry, faces, fluxes):
    """Use actual MomentumGeometry intervals and actual VolumeFluxes once.

    W/E/S/N traces vanish on blocked wedges and integrate to the input Q.
    Latitude arc weighting and its paired interior north correction preserve
    divergence and constant physical longitude-normal speed, not full vector
    velocity accuracy. Pole endpoints are unsupported, not cosine-floored.
    A piecewise-linear vertical primitive matches both interface fluxes and
    the cell net divergence. It changes no state, inventory source or force.
    The caller MUST check valid; geometry/Q precision is explicitly64.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("wet flux reconstruction requires explicit JAX X64")
    shape = faces.height.shape
    if len(shape) != 3:
        raise ValueError("wet geometry shape must contain three cell axes")
    latitude = jnp.asarray(geometry.latitude_edges)
    if latitude.shape != (shape[1] + 1,) or latitude.dtype != jnp.float64:
        raise ValueError("latitude edges must match cell shape with64 dtype")
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
    latitude_valid = (jnp.all(jnp.isfinite(latitude)) & jnp.all(jnp.diff(latitude) > 0.)
                      & (latitude[0] > -.5 * jnp.pi) & (latitude[-1] < .5 * jnp.pi))
    valid = (faces.valid & latitude_valid & intervals_valid & jnp.all(height >= 0.)
             & jnp.all(jnp.where(height > 0., True, (vertical_top == 0.) & (vertical_bottom == 0.)))
             & jnp.all(north[:, -1] == 0.) & jnp.all(vertical[..., 0] == 0.) & jnp.all(vertical[..., -1] == 0.)
             & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(field)) for field in (*geometry_fields, east, north, vertical))))
             & jnp.all(east_area >= 0.) & jnp.all(north_area >= 0.) & jnp.all(jnp.isfinite(side_density)))
    result = WetFluxReconstruction(top, height, side_top, side_bottom, side_density, vertical_top, vertical_bottom,
                                   net_flux, latitude[:-1][None, :, None], latitude[1:][None, :, None],
                                   jnp.asarray(0., jnp.float64), valid)
    endpoint = evaluate_wet_flux(result, .5, .5, top + height).vertical
    scale = jnp.sum(jnp.abs(side_flux), axis=-1) + jnp.abs(vertical_top) + jnp.abs(vertical_bottom)
    error = jnp.abs(endpoint - vertical_bottom)
    relative = jnp.max(error / jnp.where(scale > 0., scale, 1.))
    return result._replace(bottom_closure_relative=relative, valid=valid & jnp.all(error <= (1e-12 + 64. * jnp.finfo(jnp.float64).eps) * scale))


def _half_prism_map(field, south_fraction, component):
    if component == "east":
        return .5 * (field + jnp.roll(field, -1, axis=0))
    south_part = field * south_fraction
    north_part = field * (1. - south_fraction)
    return jnp.pad(south_part, ((0, 0), (0, 1), (0, 0))) + jnp.pad(north_part, ((0, 0), (1, 0), (0, 0)))


def _dual_net(fluxes):
    return (fluxes.east - jnp.roll(fluxes.east, 1, axis=0) + fluxes.north[:, 1:] - fluxes.north[:, :-1]
            + fluxes.vertical[..., 1:] - fluxes.vertical[..., :-1])


def reconstruct_half_prism_transport(geometry, volume, fluxes):
    """Integrate the metric wet reconstruction on ALL physical half-prisms.

    Each dual north flux includes both exterior wall entries. East stocks
    have primary shape; north stocks have ny+1 cells INCLUDING wall halves.
    These kinematic masses are NOT current force masses or nonlinear momentum.
    Transverse east Q uses arc halves, paired with the interior north bubble;
    mixing old area halves with that bubble breaks mass commutation.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("half-prism transport requires explicit JAX X64")
    volume = jnp.asarray(volume)
    east, north, vertical = (jnp.asarray(field) for field in fluxes)
    latitude = jnp.asarray(geometry.latitude_edges)
    shape = geometry.thickness.shape
    geometry_fields = tuple(jnp.asarray(getattr(geometry, name)) for name in ("thickness", "east_area", "north_area"))
    if (len(shape) != 3 or volume.shape != shape or east.shape != shape or north.shape != shape
            or vertical.shape != shape[:-1] + (shape[-1] + 1,) or latitude.shape != (shape[1] + 1,)
            or any(field.shape != shape for field in geometry_fields)):
        raise ValueError("half-prism volume/flux/latitude shape mismatch")
    if any(field.dtype != jnp.float64 for field in (volume, east, north, vertical, latitude, *geometry_fields)):
        raise ValueError("half-prism volume, flux and latitude require64 dtype")
    middle = .5 * (latitude[:-1] + latitude[1:])
    south_fraction = ((jnp.sin(middle) - jnp.sin(latitude[:-1])) / jnp.diff(jnp.sin(latitude)))[None, :, None]
    north_full = jnp.concatenate((jnp.zeros_like(north[:, :1]), north), axis=1)
    east_fluxes = VolumeFluxes(*(_half_prism_map(field, south_fraction, "east") for field in (east, north_full, vertical)))
    east_difference = east - jnp.roll(east, 1, axis=0)
    center_north = ((1. - south_fraction) * north_full[:, :-1] + south_fraction * north_full[:, 1:]
                    + (south_fraction - .5) * east_difference)
    north_fluxes = VolumeFluxes(_half_prism_map(east, jnp.full_like(south_fraction, .5), "north"),
                               jnp.pad(center_north, ((0, 0), (1, 1), (0, 0))),
                               _half_prism_map(vertical, south_fraction, "north"))
    primary_net = east_difference + north_full[:, 1:] - north_full[:, :-1] + vertical[..., 1:] - vertical[..., :-1]
    primary_scale = (jnp.abs(east) + jnp.abs(jnp.roll(east, 1, axis=0)) + jnp.abs(north_full[:, 1:])
                     + jnp.abs(north_full[:, :-1]) + jnp.abs(vertical[..., 1:]) + jnp.abs(vertical[..., :-1]))
    valid = (jnp.all(jnp.isfinite(latitude)) & jnp.all(jnp.diff(latitude) > 0.)
             & (latitude[0] > -.5 * jnp.pi) & (latitude[-1] < .5 * jnp.pi)
             & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(field)) for field in (volume, east, north, vertical, *geometry_fields))))
             & jnp.all(jnp.stack(tuple(jnp.all(field >= 0.) for field in geometry_fields)))
             & jnp.all(jnp.where(jnp.asarray(geometry.thickness) > 0., volume > 0., volume == 0.))
             & jnp.all(jnp.where(jnp.asarray(geometry.east_area) > 0., True, east == 0.))
             & jnp.all(jnp.where(jnp.asarray(geometry.north_area) > 0., True, north == 0.))
             & jnp.all(north[:, -1] == 0.) & jnp.all(vertical[..., 0] == 0.) & jnp.all(vertical[..., -1] == 0.))
    relative = jnp.asarray(0., jnp.float64)
    for component, dual_fluxes in (("east", east_fluxes), ("north", north_fluxes)):
        expected = _half_prism_map(primary_net, south_fraction, component)
        scale = _half_prism_map(primary_scale, south_fraction, component)
        error = jnp.abs(_dual_net(dual_fluxes) - expected)
        relative = jnp.maximum(relative, jnp.max(error / jnp.where(scale > 0., scale, 1.)))
        valid = valid & jnp.all(error <= (1e-12 + 64. * jnp.finfo(jnp.float64).eps) * scale)
    return HalfPrismTransport(_half_prism_map(volume, south_fraction, "east"),
                              _half_prism_map(volume, south_fraction, "north"), east_fluxes, north_fluxes, relative, valid)

preserve_legacy_names(globals(), 'wet_fluxes')
