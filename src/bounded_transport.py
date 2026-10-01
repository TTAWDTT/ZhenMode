"""Metric linear reconstruction and shared multidimensional extensive FCT.

The SSPRK2 candidate uses prescribed/frozen Q and constant-in-step sources.
This component does not qualify coupled momentum accuracy or production climate.
Registered gates and method choice: bounded_extensive_transport protocol.
"""
import jax.numpy as jnp

from finite_volume import ExtensiveState, TransportResult, advance_contents


def _neighbor(field, axis, direction):
    """Adjacent cells: periodic longitude, zero beyond closed latitude/depth."""
    if axis == 0:
        return jnp.roll(field, -direction, axis=axis)
    selection = [slice(None)] * field.ndim
    selection[axis] = slice(0, 1)
    zero = jnp.zeros_like(field[tuple(selection)])
    selection[axis] = slice(1, None) if direction > 0 else slice(None, -1)
    parts = (field[tuple(selection)], zero) if direction > 0 else (zero, field[tuple(selection)])
    return jnp.concatenate(parts, axis=axis)


def _face_geometry(geometry, volume):
    dtype = volume.dtype
    wet = jnp.asarray(geometry.thickness) > 0.
    height = volume / jnp.asarray(geometry.area, dtype)[..., None]
    east_open = jnp.asarray(geometry.east_area) > 0.
    north_open = jnp.asarray(geometry.north_area) > 0.
    vertical_open = wet & _neighbor(wet, 2, 1)
    east_distance = jnp.broadcast_to(jnp.asarray(geometry.east_distance, dtype)[..., None], volume.shape)
    north_distance = jnp.broadcast_to(jnp.asarray(geometry.north_distance, dtype)[..., None], volume.shape)
    vertical_distance = .5 * (height + _neighbor(height, 2, 1))
    east_width = jnp.broadcast_to(jnp.asarray(geometry.east_width, dtype)[..., None], volume.shape)
    north_width = jnp.broadcast_to(jnp.asarray(geometry.north_width, dtype)[..., None], volume.shape)
    return ((east_open, east_distance, east_width),
            (north_open, north_distance, north_width),
            (vertical_open, vertical_distance, height))


def _linear_slope(concentration, opened, distance, axis):
    previous_distance = _neighbor(distance, axis, -1)
    left = (concentration - _neighbor(concentration, axis, -1)) / jnp.where(previous_distance > 0., previous_distance, 1.)[..., None]
    right = (_neighbor(concentration, axis, 1) - concentration) / jnp.where(distance > 0., distance, 1.)[..., None]
    denominator = previous_distance + distance
    slope = (distance[..., None] * left + previous_distance[..., None] * right) / jnp.where(denominator > 0., denominator, 1.)[..., None]
    return jnp.where((opened & _neighbor(opened, axis, -1))[..., None], slope, 0.)


def _local_bounds(concentration, low_concentration, faces, forced):
    lower, upper = concentration, concentration
    if forced:
        lower = jnp.minimum(lower, low_concentration)
        upper = jnp.maximum(upper, low_concentration)
    for axis, (opened, unused_distance, unused_width) in enumerate(faces):
        for direction in (-1, 1):
            connected = opened if direction > 0 else _neighbor(opened, axis, -1)
            adjacent = _neighbor(concentration, axis, direction)
            lower = jnp.minimum(lower, jnp.where(connected[..., None], adjacent, lower))
            upper = jnp.maximum(upper, jnp.where(connected[..., None], adjacent, upper))
            if forced:
                adjacent_low = _neighbor(low_concentration, axis, direction)
                lower = jnp.minimum(lower, jnp.where(connected[..., None], adjacent_low, lower))
                upper = jnp.maximum(upper, jnp.where(connected[..., None], adjacent_low, upper))
    return lower, upper


def euler_fct_step(geometry, state, fluxes, dt, volume_source=None, content_source=None):
    """One Euler FCT stage; shared face amounts, no final concentration repair.

    Positive/negative capacities collect all six faces before any face is
    limited. Concentration allowances multiply NEW V, not old V. Explicit
    forcing may extend bounds to the forced donor state and its wet neighbors.
    Every caller must check valid, including intermediate SSP stages.
    """
    low = advance_contents(geometry, state, fluxes, dt, volume_source, content_source)
    volume, content = (jnp.asarray(field) for field in state)
    wet = jnp.asarray(geometry.thickness) > 0.
    concentration = jnp.where(wet[..., None], content / jnp.where(volume > 0., volume, 1.)[..., None], 0.)
    low_volume, low_content = low.state
    low_concentration = low_content / jnp.where(low_volume > 0., low_volume, 1.)[..., None]
    faces = _face_geometry(geometry, volume)
    face_fluxes = (fluxes.east, fluxes.north, fluxes.vertical[..., 1:])
    antidiffusive = []
    positive, negative = jnp.zeros_like(content), jnp.zeros_like(content)
    for axis, ((opened, distance, width), flux) in enumerate(zip(faces, face_fluxes)):
        flux = jnp.asarray(flux, volume.dtype)
        slope = _linear_slope(concentration, opened, distance, axis)
        following = _neighbor(concentration, axis, 1)
        left = concentration + .5 * width[..., None] * slope
        right = following - .5 * _neighbor(width, axis, 1)[..., None] * _neighbor(slope, axis, 1)
        high_face = jnp.where(flux[..., None] >= 0., left, right)
        low_face = jnp.where(flux[..., None] >= 0., concentration, following)
        amount = jnp.where(opened[..., None], jnp.asarray(dt, volume.dtype) * flux[..., None] * (high_face - low_face), 0.)
        incoming = _neighbor(amount, axis, -1)
        positive += jnp.maximum(-amount, 0.) + jnp.maximum(incoming, 0.)
        negative += jnp.maximum(amount, 0.) + jnp.maximum(-incoming, 0.)
        antidiffusive.append(amount)
    lower, upper = _local_bounds(concentration, low_concentration, faces,
                                 volume_source is not None or content_source is not None)
    capacity_positive = jnp.maximum(low_volume[..., None] * upper - low_content, 0.)
    capacity_negative = jnp.maximum(low_content - low_volume[..., None] * lower, 0.)
    ratio_positive = jnp.minimum(1., capacity_positive / jnp.where(positive > 0., positive, 1.))
    ratio_negative = jnp.minimum(1., capacity_negative / jnp.where(negative > 0., negative, 1.))
    corrected = low_content
    for axis, amount in enumerate(antidiffusive):
        coefficient = jnp.where(amount >= 0.,
                                 jnp.minimum(ratio_negative, _neighbor(ratio_positive, axis, 1)),
                                 jnp.minimum(ratio_positive, _neighbor(ratio_negative, axis, 1)))
        limited = coefficient * amount
        corrected = corrected - limited + _neighbor(limited, axis, -1)
    next_concentration = corrected / jnp.where(low_volume > 0., low_volume, 1.)[..., None]
    tolerance = 2e-6 if jnp.finfo(volume.dtype).eps > 1e-10 else 1e-12
    valid = (low.valid & jnp.all(jnp.isfinite(corrected))
             & jnp.all(jnp.where(wet[..., None], True, corrected == 0.))
             & jnp.all(jnp.where(wet[..., None], (next_concentration >= lower - tolerance)
                                 & (next_concentration <= upper + tolerance), True)))
    return TransportResult(ExtensiveState(low_volume, corrected), low.max_outflow_fraction, valid)


def advance_bounded_contents(geometry, state, fluxes, dt, volume_source=None, content_source=None):
    """SSPRK2 of two checked FCT Euler stages with frozen Q and sources.

    Both volume and content are convexly combined. This preserves the volume
    change driven by mean Q; it is not second-order coupled momentum stepping.
    The spatial accuracy gate must qualify this candidate separately.
    """
    first = euler_fct_step(geometry, state, fluxes, dt, volume_source, content_source)
    second = euler_fct_step(geometry, first.state, fluxes, dt, volume_source, content_source)
    final = ExtensiveState(.5 * state.volume + .5 * second.state.volume,
                           .5 * state.content + .5 * second.state.content)
    valid = first.valid & second.valid & jnp.all(jnp.isfinite(final.volume)) & jnp.all(jnp.isfinite(final.content))
    return TransportResult(final, jnp.maximum(first.max_outflow_fraction, second.max_outflow_fraction), valid)
