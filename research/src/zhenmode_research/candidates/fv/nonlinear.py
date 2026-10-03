"""Conservative nonlinear FV dual momentum and moving-surface candidate.

The selected kinetic norm is sum(m*u**2)/2, NOT reconstructed physical L2.
Same actual 3-D Q advances momentum and extensive V/N. Held baroclinic work
is explicit; complete thermodynamic conversion and production remain pending.
Derivation and gates: nonlinear_dual_protocol in cgrid_hydrostatic_momentum.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres


from zhenmode_research.candidates.fv.fluxes import _half_prism_map, reconstruct_half_prism_transport
from zhenmode_research.candidates.fv.geometry import (
    TransportResult,
    VolumeFluxes,
    _physical_surface_height,
    closed_surface_fluxes,
)
from zhenmode_research.candidates.fv.momentum import (
    LayerState,
    hydrostatic_pressure_force,
    momentum_geometry,
)
from zhenmode_research.candidates.fv.transport import _neighbor, advance_bounded_contents
from ocean_solver.config.definitions import OMEGA


class DualVelocity(NamedTuple):
    east: jnp.ndarray
    north: jnp.ndarray


class NonlinearDiagnostics(NamedTuple):
    solve_relative_residual: jnp.ndarray
    outer_iterations: jnp.ndarray
    outer_change: jnp.ndarray
    momentum_residual: jnp.ndarray
    momentum_tolerance: jnp.ndarray
    continuity_residual: jnp.ndarray
    continuity_tolerance: jnp.ndarray
    surface_error: jnp.ndarray
    kinetic_change: jnp.ndarray
    surface_energy_change: jnp.ndarray
    held_force_work: jnp.ndarray
    surface_source_work: jnp.ndarray
    source_incoming_energy: jnp.ndarray
    source_mixing: jnp.ndarray
    source_outgoing_energy: jnp.ndarray
    advection_dissipation: jnp.ndarray
    energy_residual: jnp.ndarray
    energy_tolerance: jnp.ndarray
    cast_work: jnp.ndarray


class NonlinearStepResult(NamedTuple):
    state: LayerState
    eta: jnp.ndarray
    initial_eta: jnp.ndarray
    mean_eta: jnp.ndarray
    transport_velocity: DualVelocity
    uncast_velocity: DualVelocity
    held_force: DualVelocity
    wall_reaction: DualVelocity
    transport: TransportResult
    fluxes: VolumeFluxes
    diagnostics: NonlinearDiagnostics
    valid: jnp.ndarray


def _pair(operation, *pairs):
    return DualVelocity(*(operation(*fields) for fields in zip(*pairs)))


def _dot(left, right):
    return sum(jnp.sum(first * second) for first, second in zip(left, right))


def _norm(fields):
    return jnp.sqrt(sum(jnp.sum(field ** 2) for field in fields))


def _map(geometry, field):
    latitude = jnp.asarray(geometry.latitude_edges)
    middle = .5 * (latitude[:-1] + latitude[1:])
    split = ((jnp.sin(middle) - jnp.sin(latitude[:-1])) / jnp.diff(jnp.sin(latitude)))[None, :, None]
    return DualVelocity(*(_half_prism_map(field, split, component) for component in ("east", "north")))


def dual_advection(velocity, fluxes, upwind=False):
    """Oriented 3-D dual momentum outflow and nonnegative upwind KE loss."""
    divergence = jnp.zeros_like(velocity)
    dissipation = jnp.asarray(0., jnp.float64)
    for axis, flux in enumerate((fluxes.east, fluxes.north[:, 1:], fluxes.vertical[..., 1:])):
        following = _neighbor(velocity, axis, 1)
        jump = velocity - following
        face = .5 * (velocity + following)
        if upwind:
            face = face + .5 * jnp.sign(flux) * jump
            dissipation = dissipation + .5 * jnp.sum(jnp.abs(flux) * jump ** 2)
        momentum_flux = flux * face
        divergence = divergence + momentum_flux - _neighbor(momentum_flux, axis, -1)
    return divergence, dissipation


def dual_rotation(volume, velocity, frequency):
    """T^T V frequency J T, with full north wall halves and exact transpose."""
    mean_east = .5 * (velocity.east + jnp.roll(velocity.east, 1, axis=0))
    mean_north = .5 * (velocity.north[:, :-1] + velocity.north[:, 1:])
    east_cell = volume * frequency * mean_north
    north_cell = -volume * frequency * mean_east
    return DualVelocity(.5 * (east_cell + jnp.roll(east_cell, -1, axis=0)),
                        .5 * (jnp.pad(north_cell, ((0, 0), (0, 1), (0, 0)))
                              + jnp.pad(north_cell, ((0, 0), (1, 0), (0, 0)))))


def _pressure(faces, eta):
    return DualVelocity(faces.east_area * (eta - jnp.roll(eta, -1, axis=0))[..., None],
                        jnp.pad(faces.north_area * (eta - _neighbor(eta, 1, 1))[..., None], ((0, 0), (1, 0), (0, 0))))


def _fluxes(faces, velocity):
    return closed_surface_fluxes(faces.east_area * velocity.east, faces.north_area * velocity.north[:, 1:])


def _outflow(fluxes):
    return jnp.sum(fluxes.east - jnp.roll(fluxes.east, 1, axis=0)
                   + fluxes.north - _neighbor(fluxes.north, 1, -1), axis=-1)


def _metric_valid(geometry):
    area, latitude = jnp.asarray(geometry.area), jnp.asarray(geometry.latitude_edges)
    east_width, north_width = jnp.asarray(geometry.east_width), jnp.asarray(geometry.north_width)
    width = jnp.diff(latitude)
    middle = .5 * (latitude[:-1] + latitude[1:])
    radius = north_width[0, 0] / width[0]
    longitude = east_width / (radius * jnp.cos(middle)[None, :])
    floor = 1e-12 + 64. * jnp.finfo(jnp.float64).eps
    return (jnp.all(jnp.isfinite(area)) & jnp.all(area > 0.) & jnp.all(jnp.isfinite(latitude))
            & jnp.all(width > 0.) & (latitude[0] > -.5 * jnp.pi) & (latitude[-1] < .5 * jnp.pi)
            & jnp.all(jnp.isfinite(east_width)) & jnp.all(east_width > 0.)
            & jnp.all(jnp.isfinite(north_width)) & jnp.all(north_width > 0.)
            & jnp.all(jnp.abs(north_width - radius * width[None, :]) <= floor * north_width)
            & jnp.all(jnp.abs(longitude - longitude[:, :1]) <= floor * longitude)
            & (jnp.abs(jnp.sum(longitude[:, 0]) - 2. * jnp.pi) <= floor * 2. * jnp.pi)
            & jnp.all(jnp.abs(area - radius ** 2 * longitude * jnp.diff(jnp.sin(latitude))[None, :]) <= floor * area))


@partial(jax.jit, static_argnames=("curvature", "upwind", "outer_iterations", "restart", "maxiter"))
def nonlinear_momentum_surface_step(geometry, state, density_anomaly, dt, reference=None,
                                    gravity=9.81, pressure_gravity=None, rho0=1025., coriolis=None,
                                    volume_source=None, content_source=None, incoming_velocity=None,
                                    curvature=True, upwind=False, outer_iterations=20, restart=30, maxiter=20):
    """Simultaneously solve u_star/eta_bar, then advance V/N with SAME actual Q.

    Incoming physical cell velocity must be declared for positive water R.
    Negative R exits at the bulk transport velocity. Only top V may move;
    topology changes/poles/invalid inputs or unconverged solves fail closed.
    Momentum can be stored32/64; geometry, inventory, Q and solves require64.
    Caller MUST check valid before accepting returned state.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("nonlinear dual dynamics requires explicit JAX X64")
    for name, value in (("outer_iterations", outer_iterations), ("restart", restart), ("maxiter", maxiter)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive static integer")
    if not isinstance(curvature, bool) or not isinstance(upwind, bool):
        raise ValueError("curvature/upwind must be static bools")
    volume, content = jnp.asarray(state.inventory.volume), jnp.asarray(state.inventory.content)
    shape = geometry.thickness.shape
    if volume.shape != shape or volume.dtype != jnp.float64 or content.ndim != 4 or content.shape[:3] != shape or content.shape[-1] < 1 or content.dtype != jnp.float64:
        raise ValueError("nonlinear volume/content must match cells and have64 dtype")
    for name in ("area", "east_width", "north_width", "latitude_edges"):
        if jnp.asarray(getattr(geometry, name)).dtype != jnp.float64:
            raise ValueError("nonlinear physical metrics require64 dtype")
    if (any(jnp.asarray(getattr(geometry, name)).shape != shape[:2] for name in ("area", "east_width", "north_width"))
            or jnp.asarray(geometry.latitude_edges).shape != (shape[1] + 1,)):
        raise ValueError("nonlinear physical metrics must match cell shape")
    east, north = jnp.asarray(state.east_velocity), jnp.asarray(state.north_velocity)
    if east.shape != shape or north.shape != shape or east.dtype not in (jnp.float32, jnp.float64) or north.dtype != east.dtype:
        raise ValueError("nonlinear layer velocities must share32/64 dtype and match cells")
    source = jnp.zeros_like(volume) if volume_source is None else jnp.asarray(volume_source)
    if source.shape != shape or source.dtype != jnp.float64:
        raise ValueError("nonlinear volume source must match64 cells")
    if content_source is not None and (jnp.asarray(content_source).shape != content.shape or jnp.asarray(content_source).dtype != jnp.float64):
        raise ValueError("nonlinear content source must match64 inventory")
    incoming = DualVelocity(jnp.zeros_like(volume), jnp.zeros_like(volume)) if incoming_velocity is None else DualVelocity(*(jnp.asarray(field) for field in incoming_velocity))
    if any(field.shape != shape or field.dtype != jnp.float64 for field in incoming):
        raise ValueError("incoming physical east/north velocity must match64 primary cells")
    dt, gravity = (jnp.asarray(value, jnp.float64) for value in (dt, gravity))
    if dt.ndim != 0 or gravity.ndim != 0:
        raise ValueError("nonlinear dt/gravity must be scalar")
    frequency = (2. * OMEGA * jnp.sin(.5 * (jnp.asarray(geometry.latitude_edges[:-1]) + jnp.asarray(geometry.latitude_edges[1:])))[None, :, None]
                 if coriolis is None else jnp.asarray(coriolis, jnp.float64))
    if frequency.ndim == 2:
        frequency = frequency[..., None]
    frequency = jnp.broadcast_to(frequency, shape)
    area = jnp.asarray(geometry.area)
    initial_eta = _physical_surface_height(geometry, volume)
    initial_faces = momentum_geometry(geometry, volume)
    opened = DualVelocity(initial_faces.east_area > 0., jnp.pad(initial_faces.north_area > 0., ((0, 0), (1, 0), (0, 0))))
    initial_velocity = DualVelocity(east.astype(jnp.float64), jnp.pad(north.astype(jnp.float64), ((0, 0), (1, 0), (0, 0))))
    mass0 = _map(geometry, volume)
    pressure = hydrostatic_pressure_force(geometry, volume, density_anomaly, reference,
                                          gravity if pressure_gravity is None else pressure_gravity, rho0)
    held = DualVelocity(pressure.east * initial_faces.east_volume,
                        jnp.pad(pressure.north * initial_faces.north_volume, ((0, 0), (1, 0), (0, 0))))
    positive_source, negative_source = jnp.maximum(source, 0.), jnp.minimum(source, 0.)
    positive, negative = _map(geometry, positive_source), _map(geometry, negative_source)
    injection = DualVelocity(_map(geometry, positive_source * incoming.east).east,
                             _map(geometry, positive_source * incoming.north).north)
    incoming_energy = DualVelocity(_map(geometry, .5 * positive_source * incoming.east ** 2).east,
                                  _map(geometry, .5 * positive_source * incoming.north ** 2).north)
    root_area = jnp.sqrt(2. * area)
    surface_scale = jnp.sqrt(jnp.where(gravity > 0., gravity, 1.))
    roundoff = 64. * jnp.finfo(jnp.float64).eps
    latitude = jnp.asarray(geometry.latitude_edges)
    middle = .5 * (latitude[:-1] + latitude[1:])
    radius = jnp.asarray(geometry.north_width)[0, 0] / (latitude[1] - latitude[0])
    wet = jnp.asarray(geometry.thickness) > 0.
    input_valid = (_metric_valid(geometry) & initial_faces.valid & pressure.valid & jnp.isfinite(dt) & (dt > 0.)
                   & jnp.isfinite(gravity) & (gravity >= 0.) & jnp.all(jnp.isfinite(content)) & jnp.all(jnp.isfinite(frequency))
                   & jnp.all(jnp.isfinite(source)) & jnp.all(source[..., 1:] == 0.)
                   & jnp.all(jnp.where(wet, True, source == 0.))
                   & (jnp.all(source <= 0.) if incoming_velocity is None else jnp.asarray(True))
                   & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(field)) for field in (*initial_velocity, *incoming))))
                   & jnp.all(jnp.where(opened.east, True, initial_velocity.east == 0.))
                   & jnp.all(jnp.where(opened.north, True, initial_velocity.north == 0.)))

    def coefficients(velocity, eta_mean):
        proposed_volume = volume.at[..., 0].add(2. * area * (eta_mean - initial_eta))
        middle_volume = .5 * (volume + proposed_volume)
        faces = momentum_geometry(geometry, middle_volume)
        mass1 = _map(geometry, proposed_volume)
        diagonal = _pair(lambda first, last: last + jnp.sqrt(jnp.where(first * last > 0., first * last, 0.)), mass0, mass1)
        root = _pair(lambda field: jnp.sqrt(jnp.where(field > 0., field, 1.)), diagonal)
        fluxes = _fluxes(faces, velocity)
        dual = reconstruct_half_prism_transport(geometry, middle_volume, fluxes)
        angular = frequency
        if curvature:
            mean_east = .5 * (velocity.east + jnp.roll(velocity.east, 1, axis=0))
            angular = angular + mean_east * jnp.tan(middle)[None, :, None] / radius
        return proposed_volume, middle_volume, faces, mass1, root, fluxes, dual, angular

    def dynamic_force(velocity, eta_mean, middle_volume, faces, dual, angular):
        clean = _pair(lambda field, mask: jnp.where(mask, field, 0.), velocity, opened)
        spin = dual_rotation(middle_volume, clean, angular)
        surface = _pressure(faces, eta_mean)
        advection = DualVelocity(dual_advection(clean.east, dual.east_fluxes, upwind)[0],
                                 dual_advection(clean.north, dual.north_fluxes, upwind)[0])
        return _pair(lambda rotation, force, outflow, rate, speed: rotation + gravity * force - outflow + rate * speed,
                     spin, surface, advection, negative, clean)

    def iteration(carry):
        velocity, eta_mean, unused_done, count, unused_change, worst_residual, valid = carry
        proposed, middle_volume, faces, mass1, root, unused_fluxes, dual, angular = coefficients(velocity, eta_mean)

        def lhs(scaled):
            physical = DualVelocity(scaled[0] / root.east, scaled[1] / root.north)
            surface = scaled[2] / (root_area * surface_scale)
            force = dynamic_force(physical, surface, middle_volume, faces, dual, angular)
            momentum = _pair(lambda field, change, weight, mask: jnp.where(mask, field - dt * change / weight, field),
                             DualVelocity(scaled[0], scaled[1]), force, root, opened)
            outflow = _outflow(_fluxes(faces, physical))
            return (*momentum, scaled[2] + dt * surface_scale * outflow / root_area)

        base_force = dynamic_force(initial_velocity, initial_eta, middle_volume, faces, dual, angular)
        rhs_momentum = _pair(lambda first, last, speed, force, external, injected, weight, mask:
                             jnp.where(mask, ((first - last) * speed + dt * (force + external + injected)) / weight, 0.),
                             mass0, mass1, initial_velocity, base_force, held, injection, root, opened)
        rhs = (*rhs_momentum, dt * surface_scale * (source[..., 0] - _outflow(_fluxes(faces, initial_velocity))) / root_area)
        increment, unused_info = gmres(lhs, rhs, tol=1e-14, atol=0., restart=restart, maxiter=maxiter, solve_method="incremental")
        residual = tuple(first - second for first, second in zip(lhs(increment), rhs))
        residual_norm, rhs_norm = _norm(residual), _norm(rhs)
        solve_tolerance = 1e-12 * rhs_norm + roundoff * (rhs_norm + _norm(increment))
        relative = residual_norm / jnp.where(rhs_norm > 0., rhs_norm, 1.)
        next_velocity = _pair(lambda previous, change, weight, mask: jnp.where(mask, previous + change / weight, 0.),
                              initial_velocity, DualVelocity(increment[0], increment[1]), root, opened)
        next_eta = initial_eta + increment[2] / (root_area * surface_scale)
        velocity_change = _norm(_pair(lambda following, previous: following - previous, next_velocity, velocity))
        velocity_scale = jnp.maximum(_norm(next_velocity), _norm(initial_velocity))
        eta_change = _norm((next_eta - eta_mean,))
        eta_scale = jnp.maximum(_norm((next_eta,)), _norm((initial_eta,)))
        change = jnp.maximum(velocity_change / jnp.where(velocity_scale > 0., velocity_scale, 1.),
                             eta_change / jnp.where(eta_scale > 0., eta_scale, 1.))
        done = change <= 2e-13
        step_valid = (valid & faces.valid & dual.valid & momentum_geometry(geometry, proposed).valid
                      & jnp.isfinite(residual_norm) & (residual_norm <= solve_tolerance)
                      & jnp.isfinite(change))
        return next_velocity, next_eta, done, count + 1, change, jnp.maximum(worst_residual, relative), step_valid

    def body(unused_index, carry):
        return jax.lax.cond(carry[2], lambda previous: previous, iteration, carry)

    initial_carry = (initial_velocity, initial_eta, jnp.asarray(False), jnp.asarray(0), jnp.asarray(jnp.inf), jnp.asarray(0.), input_valid)
    velocity, mean_eta, converged, iterations, change, solve_residual, valid = jax.lax.fori_loop(0, outer_iterations, body, initial_carry)
    unused_proposed, middle_volume, faces, unused_mass, unused_root, fluxes, dual, angular = coefficients(velocity, mean_eta)
    active = geometry._replace(east_area=faces.east_area, north_area=faces.north_area)
    transport = advance_bounded_contents(active, state.inventory, fluxes, dt, source, content_source)
    mass1 = _map(geometry, transport.state.volume)
    root0, root1 = _pair(lambda mass: jnp.sqrt(jnp.where(mass > 0., mass, 0.)), mass0), _pair(lambda mass: jnp.sqrt(jnp.where(mass > 0., mass, 0.)), mass1)
    uncast = _pair(lambda speed, previous, first, last, mask:
                   jnp.where(mask, ((first + last) * speed - first * previous) / jnp.where(last > 0., last, 1.), 0.),
                   velocity, initial_velocity, root0, root1, opened)
    force = dynamic_force(velocity, mean_eta, middle_volume, faces, dual, angular)
    total_force = _pair(lambda internal, external, injected: internal + external + injected, force, held, injection)
    impulse = _pair(lambda last, speed, first, previous: last * speed - first * previous, mass1, uncast, mass0, initial_velocity)
    raw_residual = _pair(lambda actual, tendency: actual - dt * tendency, impulse, total_force)
    reaction = _pair(lambda field, mask: jnp.where(mask, 0., field), raw_residual, opened)
    participating = _pair(lambda actual, tendency, mask: jnp.where(mask, jnp.abs(actual) + dt * jnp.abs(tendency), 0.), impulse, total_force, opened)
    stored_scale = _pair(lambda last, speed, first, previous, mask:
                         jnp.where(mask, jnp.abs(last * speed) + jnp.abs(first * previous), 0.), mass1, uncast, mass0, initial_velocity, opened)
    momentum_residual = _norm(_pair(lambda field, mask: jnp.where(mask, field, 0.), raw_residual, opened))
    momentum_tolerance = 1e-11 * _norm(participating) + roundoff * _norm(stored_scale)
    eta = _physical_surface_height(geometry, transport.state.volume)
    outflow = _outflow(fluxes)
    surface_impulse = area * (eta - initial_eta)
    surface_tendency = dt * (source[..., 0] - outflow)
    continuity_residual = _norm((surface_impulse - surface_tendency,))
    continuity_tolerance = (1e-11 * _norm((surface_tendency,))
                            + roundoff * _norm((transport.state.volume[..., 0], volume[..., 0])))
    surface_error = jnp.max(jnp.abs(eta - (2. * mean_eta - initial_eta)))
    surface_tolerance = 1e-12 + roundoff * jnp.max(jnp.abs(eta) + jnp.abs(initial_eta) + jnp.asarray(geometry.thickness[..., 0]))
    energy0, energy1 = .5 * _dot(mass0, _pair(lambda field: field ** 2, initial_velocity)), .5 * _dot(mass1, _pair(lambda field: field ** 2, uncast))
    potential0, potential1 = .5 * gravity * jnp.sum(area * initial_eta ** 2), .5 * gravity * jnp.sum(area * eta ** 2)
    held_work = dt * _dot(velocity, held)
    surface_work = dt * gravity * jnp.sum(mean_eta * source[..., 0])
    injected_energy = dt * sum(jnp.sum(field) for field in incoming_energy)
    mixing = dt * sum(jnp.sum(energy - speed * injected + .5 * rate * speed ** 2)
                      for energy, speed, injected, rate in zip(incoming_energy, velocity, injection, positive))
    outgoing_energy = .5 * dt * _dot(negative, _pair(lambda field: field ** 2, velocity))
    dissipation = dt * (dual_advection(velocity.east, dual.east_fluxes, upwind)[1] + dual_advection(velocity.north, dual.north_fluxes, upwind)[1])
    kinetic_change, surface_change = energy1 - energy0, potential1 - potential0
    work = held_work + surface_work + injected_energy - mixing + outgoing_energy - dissipation
    energy_residual = kinetic_change + surface_change - work
    energy_tolerance = (1e-11 * (jnp.abs(kinetic_change) + jnp.abs(surface_change) + jnp.abs(held_work) + jnp.abs(surface_work)
                                + injected_energy + jnp.abs(mixing) + jnp.abs(outgoing_energy) + dissipation)
                        + roundoff * (jnp.abs(energy0) + jnp.abs(energy1) + jnp.abs(potential0) + jnp.abs(potential1)))
    stored = _pair(lambda field: field.astype(east.dtype), uncast)
    cast_work = .5 * _dot(mass1, _pair(lambda field: field.astype(jnp.float64) ** 2, stored)) - energy1
    valid = (valid & converged & transport.valid & faces.valid & dual.valid & momentum_geometry(geometry, transport.state.volume).valid
             & (momentum_residual <= momentum_tolerance) & (continuity_residual <= continuity_tolerance)
             & (surface_error <= surface_tolerance) & jnp.isfinite(energy_residual) & (jnp.abs(energy_residual) <= energy_tolerance)
             & (mixing >= -roundoff * (injected_energy + jnp.abs(mixing)))
             & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(field)) for field in (*uncast, *stored, *reaction)))))
    diagnostics = NonlinearDiagnostics(solve_residual, iterations, change, momentum_residual, momentum_tolerance,
                                       continuity_residual, continuity_tolerance, surface_error, kinetic_change, surface_change,
                                       held_work, surface_work, injected_energy, mixing, outgoing_energy, dissipation,
                                       energy_residual, energy_tolerance, cast_work)
    return NonlinearStepResult(LayerState(transport.state, stored.east, stored.north[:, 1:]), eta, initial_eta, mean_eta, velocity,
                               uncast, held, reaction, transport, fluxes, diagnostics, valid)
