"""Physical control volumes and extensive shared transport for mainline migration.

Scalars are cell averages; horizontal velocities are on east/north C-grid
faces. Interfaces are positive downward. This is not a reinterpretation of
the legacy collocated/node states. See D37 and the extensive-transport protocol.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from config import R_EARTH


class FiniteVolumeGeometry(NamedTuple):
    area: np.ndarray
    thickness: np.ndarray
    center_depth: np.ndarray
    east_area: np.ndarray
    north_area: np.ndarray
    east_distance: np.ndarray
    north_distance: np.ndarray
    east_width: np.ndarray
    north_width: np.ndarray


class ExtensiveState(NamedTuple):
    volume: jnp.ndarray
    content: jnp.ndarray


class VolumeFluxes(NamedTuple):
    east: jnp.ndarray
    north: jnp.ndarray
    vertical: jnp.ndarray


class MatchedTransport(NamedTuple):
    flux: jnp.ndarray
    valid: jnp.ndarray


class TransportResult(NamedTuple):
    state: ExtensiveState
    max_outflow_fraction: jnp.ndarray
    valid: jnp.ndarray


def build_geometry(longitude_edges, latitude_edges, interfaces, depth, radius=R_EARTH):
    """Build exact spherical areas and unrounded partial cells/open faces.

    Longitude covers one full periodic band. Cell/face thicknesses are in m,
    areas in m2, center/gradient distances in m. Dry center depth is zero.
    Bathymetry deeper than the last interface is rejected, not truncated.
    """
    longitude, latitude, levels = (
        np.asarray(values, dtype=np.float64)
        for values in (longitude_edges, latitude_edges, interfaces)
    )
    for name, values in (("longitude", longitude), ("latitude", latitude), ("interfaces", levels)):
        if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.):
            raise ValueError(f"{name} must be finite, strictly increasing edges")
    if len(longitude) < 3 or not np.isclose(longitude[-1] - longitude[0], 360., rtol=0., atol=1e-10):
        raise ValueError("longitude must cover one full periodic band with at least two cells")
    if latitude[0] < -90. or latitude[-1] > 90.:
        raise ValueError("latitude edges must lie within [-90, 90]")
    if levels[0] != 0.:
        raise ValueError("positive-down interfaces must start at zero")
    if not np.isfinite(radius) or radius <= 0.:
        raise ValueError("radius must be finite and positive")
    depth = np.asarray(depth, dtype=np.float64)
    expected_shape = (len(longitude) - 1, len(latitude) - 1)
    if depth.shape != expected_shape or not np.all(np.isfinite(depth)) or np.any(depth < 0.) or np.any(depth > levels[-1]):
        raise ValueError("depth must match horizontal cells and lie within the interfaces")
    delta_lon = np.radians(np.diff(longitude))
    latitude_rad = np.radians(latitude)
    delta_lat = np.diff(latitude_rad)
    centers_lat = .5 * (latitude_rad[:-1] + latitude_rad[1:])
    area = radius ** 2 * delta_lon[:, None] * np.diff(np.sin(latitude_rad))[None, :]
    thickness = np.clip(depth[..., None] - levels[:-1], 0., np.diff(levels))
    center_depth = np.where(thickness > 0., levels[:-1] + .5 * thickness, 0.)
    east_height = np.minimum(thickness, np.roll(thickness, -1, axis=0))
    north_height = np.minimum(thickness, np.concatenate((thickness[:, 1:], np.zeros_like(thickness[:, :1])), axis=1))
    east_area = east_height * radius * delta_lat[None, :, None]
    north_area = north_height * radius * delta_lon[:, None, None] * np.cos(latitude_rad[1:])[None, :, None]
    east_distance = radius * .5 * (delta_lon + np.roll(delta_lon, -1))[:, None] * np.cos(centers_lat)[None, :]
    north_spacing = np.concatenate((np.diff(centers_lat), delta_lat[-1:]))
    north_distance = np.broadcast_to(radius * north_spacing[None, :], expected_shape).copy()
    east_width = radius * delta_lon[:, None] * np.cos(centers_lat)[None, :]
    north_width = np.broadcast_to(radius * delta_lat[None, :], expected_shape).copy()
    return FiniteVolumeGeometry(area, thickness, center_depth, east_area,
                                north_area, east_distance, north_distance, east_width, north_width)


def surface_volume(geometry, eta):
    """Physical volume with a moving top cell; validity is checked at advance.

    Eta must be zero over land. Negative/depleted wet volume is not clipped.
    """
    eta = jnp.asarray(eta)
    if not jnp.issubdtype(eta.dtype, jnp.floating):
        raise ValueError("eta must have floating dtype")
    if eta.shape != geometry.area.shape:
        raise ValueError("eta shape must match horizontal cells")
    area = jnp.asarray(geometry.area, dtype=eta.dtype)
    volume = jnp.asarray(geometry.thickness, dtype=eta.dtype) * area[..., None]
    return volume.at[..., 0].add(area * eta)


def _physical_surface_height(geometry, volume):
    """Diagnose the physical ratio of stored V and effective stored area in64.

    X64 is required, not enabled globally here. State storage is unchanged;
    the mixed arithmetic avoids losing eta bits in V/A-h before pressure and
    acceptance checks. It cannot recover bits already lost in V storage.
    """
    volume = jnp.asarray(volume)
    if volume.shape != geometry.thickness.shape or not jnp.issubdtype(volume.dtype, jnp.floating):
        raise ValueError("volume must match cells with floating dtype")
    if not jax.config.jax_enable_x64:
        raise ValueError("physical surface diagnosis requires JAX X64 arithmetic enabled explicitly")
    area = jnp.asarray(geometry.area, dtype=volume.dtype).astype(jnp.float64)
    top_height = jnp.asarray(geometry.thickness[..., 0], dtype=volume.dtype).astype(jnp.float64)
    return volume[..., 0].astype(jnp.float64) / area - top_height


def surface_height(geometry, volume):
    """Eta from primary V, using explicit mixed arithmetic and original dtype."""
    volume = jnp.asarray(volume)
    return _physical_surface_height(geometry, volume).astype(volume.dtype)


def horizontal_divergence(east, north):
    """Net oriented outflow per cell, including a closed south boundary."""
    south = jnp.concatenate((jnp.zeros_like(north[:, :1]), north[:, :-1]), axis=1)
    return east - jnp.roll(east, 1, axis=0) + north - south


def closed_surface_fluxes(east, north):
    """Bottom-up fixed-z continuity with material top/bottom, Qz positive down."""
    net_outflow = horizontal_divergence(east, north)
    accumulated = jnp.flip(jnp.cumsum(jnp.flip(net_outflow, axis=-1), axis=-1), axis=-1)
    zero = jnp.zeros_like(accumulated[..., :1])
    vertical = jnp.concatenate((zero, accumulated[..., 1:], zero), axis=-1)
    return VolumeFluxes(east, north, vertical)


def match_column_transport(face_area, velocity, column_transport):
    """Locally match layer Q sums to the actual barotropic mean Q, not endpoint.

    A depth-uniform velocity correction is weighted by the same open areas.
    Closed velocity sentinels are ignored; nonzero closed targets are invalid.
    """
    velocity = jnp.asarray(velocity)
    if not jnp.issubdtype(velocity.dtype, jnp.floating):
        raise ValueError("face velocity must have floating dtype")
    area = jnp.asarray(face_area, dtype=velocity.dtype)
    target = jnp.asarray(column_transport, dtype=velocity.dtype)
    if area.shape != velocity.shape or target.shape != area.shape[:-1]:
        raise ValueError("layer velocity/areas and column transport shapes must agree")
    opened = area > 0.
    clean_velocity = jnp.where(opened, velocity, 0.)
    predicted = area * clean_velocity
    total_area = jnp.sum(area, axis=-1)
    correction = (target - jnp.sum(predicted, axis=-1)) / jnp.where(total_area > 0., total_area, 1.)
    flux = jnp.where(opened, predicted + area * correction[..., None], 0.)
    valid = (jnp.all(jnp.isfinite(clean_velocity)) & jnp.all(jnp.isfinite(target))
             & jnp.all(jnp.where(total_area > 0., True, target == 0.))
             & jnp.all(jnp.isfinite(flux)))
    return MatchedTransport(flux, valid)


def advance_contents(geometry, state, fluxes, dt, volume_source=None, content_source=None):
    """Donor baseline: evolve V and V*C with identical oriented fluxes.

    Source units are m3/s and C*m3/s per cell. No implicit surface tracer
    source, mean repair, concentration clamp or depth floor is applied.
    Callers of this JIT-safe kernel MUST check valid; the host checked wrapper
    rejects invalid steps. Higher-order bounded transport remains required.
    """
    volume, content = (jnp.asarray(field) for field in state)
    shape = geometry.thickness.shape
    if volume.shape != shape or content.ndim != 4 or content.shape[:-1] != shape or content.shape[-1] < 1:
        raise ValueError("V must match cells; content must be cells by tracer")
    if not jnp.issubdtype(volume.dtype, jnp.floating) or content.dtype != volume.dtype:
        raise ValueError("volume/content must share a floating state dtype")
    east, north, vertical = (jnp.asarray(field, dtype=volume.dtype) for field in fluxes)
    if east.shape != shape or north.shape != shape or vertical.shape != shape[:-1] + (shape[-1] + 1,):
        raise ValueError("flux shapes must match east/north cells and vertical interfaces")
    dt = jnp.asarray(dt, dtype=volume.dtype)
    if dt.ndim != 0:
        raise ValueError("dt must be scalar")
    source_volume = jnp.zeros_like(volume) if volume_source is None else jnp.asarray(volume_source, dtype=volume.dtype)
    source_content = jnp.zeros_like(content) if content_source is None else jnp.asarray(content_source, dtype=volume.dtype)
    if source_volume.shape != shape or source_content.shape != content.shape:
        raise ValueError("source shapes must match volume and content")
    wet = jnp.asarray(geometry.thickness) > 0.
    safe_volume = jnp.where(volume > 0., volume, 1.)
    concentration = jnp.where(wet[..., None], content / safe_volume[..., None], 0.)
    east_donor = jnp.where(east[..., None] >= 0., concentration, jnp.roll(concentration, -1, axis=0))
    north_neighbor = jnp.concatenate((concentration[:, 1:], jnp.zeros_like(concentration[:, :1])), axis=1)
    north_donor = jnp.where(north[..., None] >= 0., concentration, north_neighbor)
    internal_donor = jnp.where(vertical[..., 1:-1, None] >= 0., concentration[..., :-1, :], concentration[..., 1:, :])
    zero_content = jnp.zeros_like(content[..., :1, :])
    vertical_donor = jnp.concatenate((zero_content, internal_donor, zero_content), axis=-2)
    content_outflow = horizontal_divergence(east[..., None] * east_donor, north[..., None] * north_donor)
    vertical_content_flux = vertical[..., None] * vertical_donor
    content_outflow += vertical_content_flux[..., 1:, :] - vertical_content_flux[..., :-1, :]
    volume_outflow = horizontal_divergence(east, north) + vertical[..., 1:] - vertical[..., :-1]
    next_volume = volume + dt * (source_volume - volume_outflow)
    next_content = content + dt * (source_content - content_outflow)
    north_incoming = jnp.concatenate((jnp.zeros_like(north[:, :1]), north[:, :-1]), axis=1)
    outgoing = (jnp.maximum(east, 0.) + jnp.maximum(-jnp.roll(east, 1, axis=0), 0.)
                + jnp.maximum(north, 0.) + jnp.maximum(-north_incoming, 0.)
                + jnp.maximum(vertical[..., 1:], 0.) + jnp.maximum(-vertical[..., :-1], 0.))
    max_outflow_fraction = jnp.max(jnp.where(wet, dt * outgoing / safe_volume, 0.))
    vertical_open = jnp.concatenate((jnp.zeros_like(wet[..., :1]), wet[..., :-1] & wet[..., 1:], jnp.zeros_like(wet[..., :1])), axis=-1)
    valid = (jnp.isfinite(dt) & (dt > 0.) & (max_outflow_fraction <= 1.)
             & jnp.all(jnp.isfinite(volume)) & jnp.all(jnp.isfinite(content))
             & jnp.all(jnp.isfinite(source_volume)) & jnp.all(jnp.isfinite(source_content))
             & jnp.all(jnp.isfinite(east)) & jnp.all(jnp.isfinite(north)) & jnp.all(jnp.isfinite(vertical))
             & jnp.all(jnp.where(wet, volume > 0., volume == 0.))
             & jnp.all(jnp.where(wet[..., None], True, content == 0.))
             & jnp.all(jnp.where(wet, True, source_volume == 0.))
             & jnp.all(jnp.where(wet[..., None], True, source_content == 0.))
             & jnp.all(jnp.where(jnp.asarray(geometry.east_area) > 0., True, east == 0.))
             & jnp.all(jnp.where(jnp.asarray(geometry.north_area) > 0., True, north == 0.))
             & jnp.all(jnp.where(vertical_open, True, vertical == 0.))
             & jnp.all(jnp.isfinite(next_volume)) & jnp.all(jnp.isfinite(next_content))
             & jnp.all(jnp.where(wet, next_volume > 0., next_volume == 0.))
             & jnp.all(jnp.where(wet[..., None], True, next_content == 0.)))
    return TransportResult(ExtensiveState(next_volume, next_content), max_outflow_fraction, valid)


def checked_transport_step(geometry, state, fluxes, dt, volume_source=None, content_source=None):
    """Host fail-closed interface; no invalid state is returned as accepted."""
    result = advance_contents(geometry, state, fluxes, dt, volume_source, content_source)
    if not bool(result.valid):
        raise ValueError(f"invalid extensive transport step; outflow fraction={float(result.max_outflow_fraction)}")
    return result
