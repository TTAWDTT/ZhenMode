"""C-grid substep transport and cell-content coupling for physical migration.

The gravity-wave reference uses static face geometry and omits full 3D
momentum/Coriolis/physics. It is not a qualified ocean production driver.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from bounded_transport import advance_bounded_contents
from finite_volume import (
    TransportResult,
    _physical_surface_height,
    advance_contents,
    closed_surface_fluxes,
    horizontal_divergence,
    horizontal_momentum_geometry,
    match_column_transport,
)


class BarotropicResult(NamedTuple):
    eta: jnp.ndarray
    east_velocity: jnp.ndarray
    north_velocity: jnp.ndarray
    mean_east: jnp.ndarray
    mean_north: jnp.ndarray
    gravity_cfl_bound: jnp.ndarray
    valid: jnp.ndarray


class CoupledResult(NamedTuple):
    barotropic: BarotropicResult
    transport: TransportResult
    surface_error: jnp.ndarray
    valid: jnp.ndarray


def subcycle_barotropic(geometry, eta, east_velocity, north_velocity, dt_sub, nsub,
                       gravity=9.81, drag=0., volume_source=None,
                       east_acceleration=None, north_acceleration=None):
    """Forward-backward wave steps and mean of Q actually used in eta.

    Gravity uses physical face_length/dual_area, paired with shared-Q work.
    A sufficient Gershgorin wave bound is dt_sub^2*g*max(diagonal)<=2.
    Material-top thickness must stay positive. No state or timestep repair.
    nsub is a static positive integer for JIT callers.
    """
    if not isinstance(nsub, int) or isinstance(nsub, bool) or nsub < 1:
        raise ValueError("nsub must be a positive static integer")
    eta = jnp.asarray(eta)
    shape = geometry.area.shape
    if eta.shape != shape or east_velocity.shape != shape or north_velocity.shape != shape:
        raise ValueError("barotropic state shapes must match horizontal cells")
    if not jnp.issubdtype(eta.dtype, jnp.floating):
        raise ValueError("eta must have floating dtype")
    area = jnp.asarray(geometry.area, eta.dtype)
    east_area = jnp.sum(jnp.asarray(geometry.east_area, eta.dtype), axis=-1)
    north_area = jnp.sum(jnp.asarray(geometry.north_area, eta.dtype), axis=-1)
    horizontal = horizontal_momentum_geometry(geometry)
    east_metric = jnp.asarray(horizontal.east_face_length / horizontal.east_dual_area, eta.dtype)
    north_metric = jnp.asarray(horizontal.north_face_length / horizontal.north_dual_area, eta.dtype)
    wet = jnp.asarray(geometry.thickness[..., 0]) > 0.
    top_height = jnp.asarray(geometry.thickness[..., 0], eta.dtype)
    dt_sub, gravity, drag = (jnp.asarray(value, eta.dtype) for value in (dt_sub, gravity, drag))
    if any(value.ndim != 0 for value in (dt_sub, gravity, drag)):
        raise ValueError("dt_sub, gravity and drag must be scalars")
    source = jnp.zeros_like(eta) if volume_source is None else jnp.asarray(volume_source, eta.dtype)
    if source.shape != shape:
        raise ValueError("barotropic volume source must match horizontal cells")
    force_east = jnp.zeros_like(eta) if east_acceleration is None else jnp.asarray(east_acceleration, eta.dtype)
    force_north = jnp.zeros_like(eta) if north_acceleration is None else jnp.asarray(north_acceleration, eta.dtype)
    if force_east.shape != shape or force_north.shape != shape:
        raise ValueError("barotropic acceleration must match horizontal faces")
    force_east = jnp.where(east_area > 0., force_east, 0.)
    force_north = jnp.where(north_area > 0., force_north, 0.)
    east = jnp.where(east_area > 0., jnp.asarray(east_velocity, eta.dtype), 0.)
    north = jnp.where(north_area > 0., jnp.asarray(north_velocity, eta.dtype), 0.)
    east_stiffness, north_stiffness = east_area * east_metric, north_area * north_metric
    south_stiffness = jnp.concatenate((jnp.zeros_like(north_stiffness[:, :1]), north_stiffness[:, :-1]), axis=1)
    diagonal = (east_stiffness + jnp.roll(east_stiffness, 1, axis=0) + north_stiffness + south_stiffness) / area
    cfl_bound = dt_sub ** 2 * gravity * jnp.max(diagonal)
    initial_valid = (jnp.isfinite(dt_sub) & (dt_sub > 0.) & jnp.isfinite(gravity) & (gravity >= 0.)
                     & jnp.isfinite(drag) & (drag >= 0.) & (cfl_bound <= 2.)
                     & jnp.all(jnp.isfinite(eta)) & jnp.all(jnp.isfinite(east)) & jnp.all(jnp.isfinite(north))
                     & jnp.all(jnp.isfinite(force_east)) & jnp.all(jnp.isfinite(force_north))
                     & jnp.all(jnp.isfinite(source)) & jnp.all(jnp.where(wet, True, source == 0.))
                     & jnp.all(jnp.where(wet, top_height + eta > 0., eta == 0.)))

    def advance(carry, unused):
        height, velocity_east, velocity_north, sum_east, sum_north, valid = carry
        flux_east, flux_north = east_area * velocity_east, north_area * velocity_north
        new_height = height + dt_sub * (source - horizontal_divergence(flux_east, flux_north)) / area
        east_gradient = (jnp.roll(new_height, -1, axis=0) - new_height) * east_metric
        north_neighbor = jnp.concatenate((new_height[:, 1:], new_height[:, -1:]), axis=1)
        north_gradient = (north_neighbor - new_height) * north_metric
        new_east = jnp.where(east_area > 0., (velocity_east - dt_sub * gravity * east_gradient + dt_sub * force_east) / (1. + dt_sub * drag), 0.)
        new_north = jnp.where(north_area > 0., (velocity_north - dt_sub * gravity * north_gradient + dt_sub * force_north) / (1. + dt_sub * drag), 0.)
        valid = (valid & jnp.all(jnp.isfinite(new_height)) & jnp.all(jnp.isfinite(new_east))
                 & jnp.all(jnp.isfinite(new_north)) & jnp.all(jnp.where(wet, top_height + new_height > 0., new_height == 0.)))
        return (new_height, new_east, new_north, sum_east + flux_east, sum_north + flux_north, valid), None

    initial = (eta, east, north, jnp.zeros_like(eta), jnp.zeros_like(eta), initial_valid)
    (eta, east, north, sum_east, sum_north, valid), _ = jax.lax.scan(advance, initial, None, length=nsub)
    return BarotropicResult(eta, east, north, sum_east / nsub, sum_north / nsub, cfl_bound, valid)


def coupled_surface_step(geometry, state, eta, east_velocity, north_velocity,
                         layer_east_velocity, layer_north_velocity, dt_sub, nsub,
                         gravity=9.81, drag=0., volume_source=None, content_source=None,
                         inventory_precision=None, transport_scheme="donor"):
    """Share actual substep mean Q between eta, layer volumes and V*C.

    Only top-layer sources are permitted in this fixed-z moving-top coupling.
    Low-level results must pass valid before accepting the next state.
    Explicit inventory_precision='float64' permits eta/velocities32 with V/N64;
    otherwise all state dtypes must match. No automatic inventory promotion.
    centered_fct selects the registered higher-order candidate, not production.
    """
    eta, east_velocity, north_velocity, layer_east_velocity, layer_north_velocity = (
        jnp.asarray(field) for field in (eta, east_velocity, north_velocity,
                                        layer_east_velocity, layer_north_velocity)
    )
    if inventory_precision not in (None, "float64"):
        raise ValueError("inventory_precision must be None or explicit float64")
    momentum_fields = (eta, east_velocity, north_velocity, layer_east_velocity, layer_north_velocity)
    if inventory_precision is None:
        if any(field.dtype != state.volume.dtype for field in momentum_fields):
            raise ValueError("coupled scalar and face states must share the volume dtype")
    elif (state.volume.dtype != jnp.float64 or state.content.dtype != jnp.float64
          or eta.dtype not in (jnp.float32, jnp.float64)
          or any(field.dtype != eta.dtype for field in momentum_fields)):
        raise ValueError("explicit float64 inventory requires V/N64 and a shared float32/64 momentum dtype")
    if transport_scheme not in ("donor", "centered_fct"):
        raise ValueError("transport_scheme must be donor or centered_fct")
    source = jnp.zeros_like(state.volume) if volume_source is None else jnp.asarray(volume_source, state.volume.dtype)
    if source.shape != state.volume.shape:
        raise ValueError("volume source must match cells")
    barotropic = subcycle_barotropic(geometry, eta, east_velocity, north_velocity,
                                    dt_sub, nsub, gravity, drag, source[..., 0])
    east = match_column_transport(geometry.east_area, layer_east_velocity, barotropic.mean_east)
    north = match_column_transport(geometry.north_area, layer_north_velocity, barotropic.mean_north)
    fluxes = closed_surface_fluxes(east.flux, north.flux)
    advance = advance_contents if transport_scheme == "donor" else advance_bounded_contents
    transport = advance(geometry, state, fluxes, dt_sub * nsub, volume_source, content_source)
    area = jnp.asarray(geometry.area, state.volume.dtype)
    initial_eta = _physical_surface_height(geometry, state.volume)
    content_eta = _physical_surface_height(geometry, transport.state.volume)
    scale = jnp.maximum(jnp.asarray(1e-8, state.volume.dtype), jnp.maximum(jnp.max(jnp.abs(eta)), jnp.max(jnp.abs(barotropic.eta - eta))))
    surface_error = jnp.maximum(jnp.max(jnp.abs(content_eta - barotropic.eta)), jnp.max(jnp.abs(initial_eta - eta))) / scale
    tolerance = jnp.where(jnp.finfo(eta.dtype).eps > 1e-10, 2e-6, 1e-12)
    lower_volume = jnp.asarray(geometry.thickness[..., 1:], state.volume.dtype) * area[..., None]
    lower_scale = jnp.maximum(lower_volume, 1.)
    lower_error = jnp.max(jnp.abs(state.volume[..., 1:] - lower_volume) / lower_scale, initial=0.)
    valid = (barotropic.valid & east.valid & north.valid & transport.valid
             & (surface_error <= tolerance) & (lower_error <= tolerance)
             & jnp.all(source[..., 1:] == 0.))
    return CoupledResult(barotropic, transport, surface_error, valid)
