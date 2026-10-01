"""Physical mean-Q velocity frame on frozen wet geometry, not full ALE dynamics."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from wet_fluxes import WetFluxReconstruction, _expand, evaluate_wet_flux


class PhysicalVelocityReconstruction(NamedTuple):
    transport: WetFluxReconstruction
    area: jnp.ndarray
    meridional_area_width: jnp.ndarray
    zonal_arc: jnp.ndarray
    surface_downward: jnp.ndarray
    source_speed: jnp.ndarray
    absolute_lift: jnp.ndarray
    bulk_continuity_relative: jnp.ndarray
    valid: jnp.ndarray


class PhysicalVelocityPoint(NamedTuple):
    east: jnp.ndarray
    north: jnp.ndarray
    downward: jnp.ndarray
    grid_downward: jnp.ndarray
    relative_downward: jnp.ndarray
    divergence: jnp.ndarray
    valid: jnp.ndarray


def physical_velocity_from_fluxes(geometry, transport, volume_source=None):
    """Use SAME mean-Q and top source as actual continuity, without changing V.

    Top grid motion follows (D-source)/A. Boundary inflow converts explicit
    top cell source to relative inflow ONLY in this view. Absolute fluid w
    lifts mapped qz by D*(1-alpha)/A; lower grids stay fixed. Source-only
    filling has zero bulk fluid w, a rising surface and correct relative rain.
    Incompressibility, material/step walls and metric identity are checked;
    only top sources and non-polar64 geometry are supported at this stage.
    This is not endpoint velocity, nonlinear flux or moving-time KE closure.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("physical velocity requires explicit JAX X64")
    shape = transport.top.shape
    if len(shape) != 3:
        raise ValueError("physical velocity requires three cell shape axes")
    cell_names = ("top", "height", "vertical_top", "vertical_bottom", "net_flux")
    side_names = ("side_top", "side_bottom", "side_density")
    transport_fields = tuple(jnp.asarray(getattr(transport, name)) for name in (*cell_names, *side_names))
    if (any(getattr(transport, name).shape != shape for name in cell_names)
            or any(getattr(transport, name).shape != shape + (4,) for name in side_names)):
        raise ValueError("physical transport shape must match cells and four contacts")
    if any(field.dtype != jnp.float64 for field in transport_fields):
        raise ValueError("physical transport requires64 dtype")
    area, meridional, zonal = (jnp.asarray(getattr(geometry, name)) for name in ("area", "north_width", "east_width"))
    latitude = jnp.asarray(geometry.latitude_edges)
    source = jnp.zeros(shape, jnp.float64) if volume_source is None else jnp.asarray(volume_source)
    if (any(field.shape != shape[:2] for field in (area, meridional, zonal))
            or latitude.shape != (shape[1] + 1,) or source.shape != shape):
        raise ValueError("physical metric/source shape must match transport")
    if any(field.dtype != jnp.float64 for field in (area, meridional, zonal, latitude, source)):
        raise ValueError("physical geometry/source must have64 dtype")
    width = jnp.diff(latitude)
    sine_width = jnp.diff(jnp.sin(latitude))
    middle = .5 * (latitude[:-1] + latitude[1:])
    meridional_area_width = meridional * (sine_width / width)[None, :]
    zonal_arc = zonal / jnp.cos(middle)[None, :]
    metric_error = jnp.abs(meridional_area_width * zonal_arc - area)
    radius = meridional[0, 0] / width[0]
    scale = (jnp.sum(jnp.abs(transport.side_density) * (transport.side_bottom - transport.side_top), axis=-1)
             + jnp.abs(transport.vertical_top) + jnp.abs(transport.vertical_bottom))
    lower_error = jnp.abs(transport.net_flux[..., 1:])
    relative = jnp.max(lower_error / jnp.where(scale[..., 1:] > 0., scale[..., 1:], 1.), initial=0.)
    floor = 1e-12 + 64. * jnp.finfo(jnp.float64).eps
    valid = (transport.valid & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(field)) for field in (area, meridional, zonal, latitude, source, *transport_fields))))
             & jnp.all(transport.height >= 0.)
             & jnp.all(area > 0.) & jnp.all(meridional > 0.) & jnp.all(zonal > 0.)
             & jnp.all(jnp.diff(latitude) > 0.) & (latitude[0] > -.5 * jnp.pi) & (latitude[-1] < .5 * jnp.pi)
             & jnp.all(transport.south_latitude == latitude[:-1][None, :, None])
             & jnp.all(transport.north_latitude == latitude[1:][None, :, None])
             & jnp.all(jnp.abs(meridional - radius * width[None, :]) <= floor * meridional)
             & jnp.all(jnp.abs(zonal_arc - zonal_arc[:, :1]) <= floor * zonal_arc)
             & (jnp.abs(jnp.sum(zonal_arc[:, 0]) / radius - 2. * jnp.pi) <= floor * 2. * jnp.pi)
             & jnp.all(metric_error <= floor * area) & jnp.all(lower_error <= floor * scale[..., 1:])
             & jnp.all(jnp.isfinite(transport.net_flux)) & jnp.all(jnp.isfinite(scale))
             & jnp.all(source[..., 1:] == 0.) & jnp.all(jnp.where(transport.height > 0., True, source == 0.)))
    absolute_lift = jnp.zeros(shape, jnp.float64).at[..., 0].set(transport.net_flux[..., 0] / area)
    source_speed = source / area[..., None]
    surface_downward = absolute_lift - source_speed
    return PhysicalVelocityReconstruction(transport, area[..., None], meridional_area_width[..., None], zonal_arc[..., None],
                                          surface_downward, source_speed, absolute_lift, relative, valid)


def evaluate_physical_velocity(reconstruction, longitude_fraction, latitude_area_fraction, depth):
    """Eulerian east/north/down m/s plus grid and boundary-relative frames.

    Coordinates/optional trailing samples follow evaluate_wet_flux. Returned
    bulk divergence is physical1/s, not mapped D/H. Invalid queries have false
    per-point validity and no accepted coordinate or cosine repair.
    """
    transport = reconstruction.transport
    point = evaluate_wet_flux(transport, longitude_fraction, latitude_area_fraction, depth)
    dimensions = point.vertical.ndim
    fraction = _expand(jnp.asarray(latitude_area_fraction, jnp.float64), dimensions)
    depth = _expand(jnp.asarray(depth, jnp.float64), dimensions)
    south, north = (_expand(field, dimensions) for field in (transport.south_latitude, transport.north_latitude))
    latitude = jnp.arcsin(jnp.sin(south) + fraction * (jnp.sin(north) - jnp.sin(south)))
    top, height = (_expand(field, dimensions) for field in (transport.top, transport.height))
    safe_height = jnp.where(height > 0., height, 1.)
    remainder = 1. - (depth - top) / safe_height
    area, meridional, zonal, surface, source, lift = (_expand(field, dimensions) for field in
                                                    (reconstruction.area, reconstruction.meridional_area_width, reconstruction.zonal_arc,
                                                     reconstruction.surface_downward, reconstruction.source_speed, reconstruction.absolute_lift))
    east = point.east_per_depth * jnp.cos(latitude) / meridional
    north = point.north_per_depth / (zonal * jnp.cos(latitude))
    downward = point.vertical / area + lift * remainder
    grid_downward = surface * remainder
    relative_downward = point.vertical / area + source * remainder
    divergence = point.divergence_per_depth / area - lift / safe_height
    fields = (east, north, downward, grid_downward, relative_downward, divergence)
    valid = point.valid & reconstruction.valid & jnp.all(jnp.stack(tuple(jnp.isfinite(field) for field in fields)), axis=0)
    return PhysicalVelocityPoint(*(jnp.where(valid, field, 0.) for field in fields), valid)
