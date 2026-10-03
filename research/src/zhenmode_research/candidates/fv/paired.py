"""Physical-field kinetic mass and paired frozen midpoint layer/surface step.

This actual migration kernel advances V/N and layer momentum together. Its
frozen energy identity is not a moving-mass, nonlinear or production claim.
See paired_dynamics_protocol for the method change and remaining scope.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres


from zhenmode_research.candidates.fv.fluxes import (
    HalfPrismTransport,
    WetFluxReconstruction,
    reconstruct_half_prism_transport,
    reconstruct_wet_fluxes,
)
from zhenmode_research.candidates.fv.geometry import (
    TransportResult,
    VolumeFluxes,
    _physical_surface_height,
    closed_surface_fluxes,
)
from zhenmode_research.candidates.fv.momentum import (
    LayerState,
    PressureForce,
    hydrostatic_pressure_force,
    momentum_geometry,
)
from zhenmode_research.candidates.fv.transport import _neighbor, advance_bounded_contents
from zhenmode_research.candidates.fv.velocity import (
    PhysicalVelocityReconstruction,
    physical_velocity_from_fluxes,
)
from ocean_solver.config.definitions import OMEGA


class VelocityPair(NamedTuple):
    east: jnp.ndarray
    north: jnp.ndarray


class PairedOperators(NamedTuple):
    mass: jnp.ndarray
    rotation: jnp.ndarray
    east_diagonal: jnp.ndarray
    north_diagonal: jnp.ndarray
    east_area: jnp.ndarray
    north_area: jnp.ndarray
    area: jnp.ndarray
    wet: jnp.ndarray
    valid: jnp.ndarray


class PairedSurfaceResult(NamedTuple):
    velocity: VelocityPair
    eta: jnp.ndarray
    initial_eta: jnp.ndarray
    mean_east: jnp.ndarray
    mean_north: jnp.ndarray
    mean_eta: jnp.ndarray
    uncast_velocity: VelocityPair
    solve_relative_residual: jnp.ndarray
    frozen_energy_change: jnp.ndarray
    external_work: jnp.ndarray
    energy_work_residual: jnp.ndarray
    energy_work_relative: jnp.ndarray
    cast_work: jnp.ndarray
    valid: jnp.ndarray


class PairedStepResult(NamedTuple):
    state: LayerState
    pressure: PressureForce
    held_force: VelocityPair
    surface: PairedSurfaceResult
    transport: TransportResult
    fluxes: VolumeFluxes
    flux_reconstruction: WetFluxReconstruction
    dual_transport: HalfPrismTransport
    physical_velocity: PhysicalVelocityReconstruction
    surface_error: jnp.ndarray
    moving_mass_energy_change: jnp.ndarray
    valid: jnp.ndarray


def _scatter(local):
    return VelocityPair(local[..., 1] + _neighbor(local[..., 0], 0, 1),
                        local[..., 3] + _neighbor(local[..., 2], 1, 1))


def _local_coefficients(operators, velocity):
    east = jnp.where(operators.east_area > 0., velocity.east, 0.)
    north = jnp.where(operators.north_area > 0., velocity.north, 0.)
    return jnp.stack((_neighbor(east, 0, -1), east, _neighbor(north, 1, -1), north), axis=-1)


def _action(operators, matrix, velocity):
    return _scatter(jnp.einsum("...ab,...b->...a", matrix, _local_coefficients(operators, velocity)))


def mass_action(operators, velocity):
    """Assembled matrix-free physical horizontal L2 mass action, units m3."""
    return _action(operators, operators.mass, velocity)


def rotation_action(operators, velocity):
    """Skew weak Coriolis force with the same field and wet intersections."""
    return _action(operators, operators.rotation, velocity)


def column_outflow(operators, velocity):
    """B*u from physical layer face areas, positive outward m3/s."""
    east = operators.east_area * velocity.east
    north = operators.north_area * velocity.north
    return jnp.sum(east - _neighbor(east, 0, -1) + north - _neighbor(north, 1, -1), axis=-1)


def surface_pressure_force(operators, eta):
    """B^T*eta; multiplying by g gives the paired pressure force per rho0."""
    return VelocityPair(operators.east_area * (eta - _neighbor(eta, 0, 1))[..., None],
                        operators.north_area * (eta - _neighbor(eta, 1, 1))[..., None])


def paired_operators(geometry, volume, coriolis=None, quadrature_order=24):
    """Integrate physical basis mass and rotation on actual common wet supports."""
    if not isinstance(quadrature_order, int) or isinstance(quadrature_order, bool) or quadrature_order < 2:
        raise ValueError("quadrature_order must be a static integer at least2")
    faces = momentum_geometry(geometry, volume)
    metrics = tuple(jnp.asarray(getattr(geometry, name)) for name in ("area", "east_width", "north_width", "latitude_edges"))
    if any(field.dtype != jnp.float64 for field in metrics):
        raise ValueError("paired physical metrics must have64 dtype")
    area, east_width, north_width, latitude = metrics
    if area.shape != volume.shape[:2] or east_width.shape != area.shape or north_width.shape != area.shape or latitude.shape != (volume.shape[1] + 1,):
        raise ValueError("paired physical metrics must match cells")
    delta_phi = jnp.diff(latitude)
    middle = .5 * (latitude[:-1] + latitude[1:])
    radius = north_width[0, 0] / delta_phi[0]
    delta_lambda = east_width / (radius * jnp.cos(middle)[None, :])
    sine_width = jnp.diff(jnp.sin(latitude))
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    fraction, weights = jnp.asarray((nodes + 1.) * .5), jnp.asarray(weights * .5)
    phi = jnp.arcsin(jnp.sin(latitude[:-1])[:, None] + fraction * sine_width[:, None])
    beta = (delta_phi[None, :, None] / delta_lambda[..., None]
            * (fraction - (phi - latitude[:-1, None]) / delta_phi[:, None])[None, ...] / jnp.cos(phi)[None, ...])
    south = (1. - fraction) * jnp.cos(latitude[:-1, None]) / jnp.cos(phi)
    north = fraction * jnp.cos(latitude[1:, None]) / jnp.cos(phi)
    north_basis = jnp.stack((-beta, beta, jnp.broadcast_to(south, beta.shape), jnp.broadcast_to(north, beta.shape)), axis=-1)
    east_gram = jnp.zeros((4, 4), jnp.float64).at[:2, :2].set(jnp.array([[1. / 3., 1. / 6.], [1. / 6., 1. / 3.]]))
    weighted = north_basis * jnp.sqrt(weights)[None, None, :, None]
    angular_mass = jnp.einsum("ijqa,ijqb->ijab", weighted, weighted) + east_gram
    frequency = (jnp.broadcast_to(2. * OMEGA * jnp.sin(phi)[None, ...], beta.shape) if coriolis is None
                 else jnp.broadcast_to(jnp.asarray(coriolis, jnp.float64), area.shape)[..., None] * jnp.ones_like(beta))
    mean_north = jnp.einsum("q,ijq,ijqa->ija", weights, frequency, north_basis)
    cross = jnp.array([.5, .5, 0., 0.])[None, None, :, None] * mean_north[..., None, :]
    angular_rotation = cross - jnp.swapaxes(cross, -1, -2)
    tops = jnp.stack((_neighbor(faces.east_top, 0, -1), faces.east_top,
                      _neighbor(faces.north_top, 1, -1), faces.north_top), axis=-1)
    bottoms = jnp.stack((_neighbor(faces.east_bottom, 0, -1), faces.east_bottom,
                         _neighbor(faces.north_bottom, 1, -1), faces.north_bottom), axis=-1)
    east_open, north_open = faces.east_area > 0., faces.north_area > 0.
    supports = jnp.stack((_neighbor(east_open, 0, -1), east_open,
                          _neighbor(north_open, 1, -1), north_open), axis=-1)
    overlap = jnp.maximum(jnp.minimum(bottoms[..., :, None], bottoms[..., None, :]) - jnp.maximum(tops[..., :, None], tops[..., None, :]), 0.)
    overlap = jnp.where(supports[..., :, None] & supports[..., None, :], overlap, 0.)
    mass = area[..., None, None, None] * overlap * angular_mass[..., None, :, :]
    rotation = area[..., None, None, None] * overlap * angular_rotation[..., None, :, :]
    diagonal = _scatter(jnp.diagonal(mass, axis1=-2, axis2=-1))
    floor = 1e-12 + 64. * jnp.finfo(jnp.float64).eps
    valid = (faces.valid & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(field)) for field in (*metrics, frequency, mass, rotation))))
             & jnp.all(area > 0.) & jnp.all(east_width > 0.) & jnp.all(north_width > 0.)
             & jnp.all(delta_phi > 0.) & (latitude[0] > -.5 * jnp.pi) & (latitude[-1] < .5 * jnp.pi)
             & jnp.all(jnp.abs(north_width - radius * delta_phi[None, :]) <= floor * north_width)
             & jnp.all(jnp.abs(delta_lambda - delta_lambda[:, :1]) <= floor * delta_lambda)
             & (jnp.abs(jnp.sum(delta_lambda[:, 0]) - 2. * jnp.pi) <= floor * 2. * jnp.pi)
             & jnp.all(jnp.abs(area - radius ** 2 * delta_lambda * sine_width[None, :]) <= floor * area)
             & jnp.all(jnp.where(faces.east_area > 0., diagonal.east > 0., diagonal.east == 0.))
             & jnp.all(jnp.where(faces.north_area > 0., diagonal.north > 0., diagonal.north == 0.)))
    return PairedOperators(mass, rotation, *diagonal, faces.east_area, faces.north_area, area, faces.height[..., 0] > 0., valid)


def _dot(left, right):
    return jnp.sum(left.east * right.east) + jnp.sum(left.north * right.north)


def kinetic_energy(operators, velocity):
    return .5 * _dot(velocity, mass_action(operators, velocity))


def advance_paired_surface(operators, eta, velocity, dt, nsub, gravity=9.81,
                           force=None, volume_source=None, restart=30, maxiter=20):
    """Coupled implicit midpoint; SAME substep-mean layer Q drives continuity.

    Solve/work gates apply before momentum storage casting. Cast work is
    reported separately; no explicit-wave CFL or moving-M qualification.
    """
    for name, value in (("nsub", nsub), ("restart", restart), ("maxiter", maxiter)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive static integer")
    shape = operators.east_area.shape
    eta = jnp.asarray(eta)
    velocity = VelocityPair(*(jnp.asarray(field) for field in velocity))
    if eta.shape != shape[:2] or eta.dtype != jnp.float64:
        raise ValueError("paired eta must match columns and have64 dtype")
    if any(field.shape != shape or field.dtype not in (jnp.float32, jnp.float64) for field in velocity) or velocity.east.dtype != velocity.north.dtype:
        raise ValueError("paired velocity must share32/64 dtype and match layers")
    force = VelocityPair(jnp.zeros(shape, jnp.float64), jnp.zeros(shape, jnp.float64)) if force is None else VelocityPair(*(jnp.asarray(field) for field in force))
    source = jnp.zeros(shape[:2], jnp.float64) if volume_source is None else jnp.asarray(volume_source)
    if source.shape != shape[:2] or source.dtype != jnp.float64 or any(field.shape != shape or field.dtype != jnp.float64 for field in force):
        raise ValueError("paired force/source must match layers/columns and have64 dtype")
    root_mass = VelocityPair(jnp.sqrt(jnp.where(operators.east_diagonal > 0., operators.east_diagonal, 1.)),
                             jnp.sqrt(jnp.where(operators.north_diagonal > 0., operators.north_diagonal, 1.)))
    root_area = jnp.sqrt(operators.area)
    surface_scale = jnp.sqrt(jnp.where(gravity > 0., gravity, 1.))
    opened = VelocityPair(operators.east_area > 0., operators.north_area > 0.)
    initial_velocity = VelocityPair(*(jnp.where(mask, field.astype(jnp.float64), 0.) for mask, field in zip(opened, velocity)))
    initial = (root_mass.east * initial_velocity.east, root_mass.north * initial_velocity.north, root_area * surface_scale * eta)
    sub_dt = dt / nsub

    def terms(state):
        physical = VelocityPair(state[0] / root_mass.east, state[1] / root_mass.north)
        mass = mass_action(operators, physical)
        spin = rotation_action(operators, physical)
        pressure = surface_pressure_force(operators, state[2] / (root_area * surface_scale))
        scaled_mass = (jnp.where(opened.east, mass.east / root_mass.east, state[0]),
                       jnp.where(opened.north, mass.north / root_mass.north, state[1]), state[2])
        dynamics = (jnp.where(opened.east, (spin.east + gravity * pressure.east) / root_mass.east, 0.),
                    jnp.where(opened.north, (spin.north + gravity * pressure.north) / root_mass.north, 0.),
                    -surface_scale * column_outflow(operators, physical) / root_area)
        return scaled_mass, dynamics

    def lhs(state):
        metric, dynamics = terms(state)
        return tuple(field - .5 * sub_dt * tendency for field, tendency in zip(metric, dynamics))

    scaled_source = (force.east / root_mass.east, force.north / root_mass.north, surface_scale * source / root_area)

    def substep(carry, unused_index):
        previous, east_sum, north_sum, eta_sum, worst_residual, work, valid = carry
        metric, dynamics = terms(previous)
        rhs = tuple(field + .5 * sub_dt * tendency + sub_dt * external for field, tendency, external in zip(metric, dynamics, scaled_source))
        increment_rhs = tuple(sub_dt * (tendency + external) for tendency, external in zip(dynamics, scaled_source))
        increment, unused_info = gmres(lhs, increment_rhs, tol=1e-14, atol=0., restart=restart,
                                      maxiter=maxiter, solve_method="incremental")
        final = tuple(field + change for field, change in zip(previous, increment))
        increment_residual = tuple(left - right for left, right in zip(lhs(increment), increment_rhs))
        increment_residual_norm = jnp.sqrt(sum(jnp.sum(field ** 2) for field in increment_residual))
        increment_rhs_norm = jnp.sqrt(sum(jnp.sum(field ** 2) for field in increment_rhs))
        increment_norm = jnp.sqrt(sum(jnp.sum(field ** 2) for field in increment))
        increment_tolerance = 1e-12 * increment_rhs_norm + 64. * jnp.finfo(jnp.float64).eps * (increment_rhs_norm + increment_norm)
        residual = tuple(left - right for left, right in zip(lhs(final), rhs))
        residual_norm = jnp.sqrt(sum(jnp.sum(field ** 2) for field in residual))
        rhs_norm = jnp.sqrt(sum(jnp.sum(field ** 2) for field in rhs))
        final_norm = jnp.sqrt(sum(jnp.sum(field ** 2) for field in final))
        tolerance = 1e-12 * rhs_norm + 64. * jnp.finfo(jnp.float64).eps * (rhs_norm + final_norm)
        relative = jnp.maximum(residual_norm / jnp.where(rhs_norm > 0., rhs_norm, 1.),
                               increment_residual_norm / jnp.where(increment_rhs_norm > 0., increment_rhs_norm, 1.))
        midpoint = VelocityPair(.5 * (previous[0] + final[0]) / root_mass.east,
                                .5 * (previous[1] + final[1]) / root_mass.north)
        midpoint_eta = .5 * (previous[2] + final[2]) / (root_area * surface_scale)
        external_work = sub_dt * (_dot(midpoint, force) + gravity * jnp.sum(midpoint_eta * source))
        return (final, east_sum + operators.east_area * midpoint.east / nsub,
                north_sum + operators.north_area * midpoint.north / nsub,
                eta_sum + midpoint_eta / nsub,
                jnp.maximum(worst_residual, relative), work + external_work,
                valid & jnp.isfinite(residual_norm) & (residual_norm <= tolerance)
                & jnp.isfinite(increment_residual_norm) & (increment_residual_norm <= increment_tolerance)), None

    input_valid = (operators.valid & jnp.isfinite(dt) & (dt > 0.) & jnp.isfinite(gravity) & (gravity >= 0.)
                   & jnp.all(jnp.isfinite(eta)) & jnp.all(jnp.where(operators.wet, True, eta == 0.))
                   & jnp.all(jnp.where(operators.wet, jnp.isfinite(source), source == 0.))
                   & jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(field)) for field in (*velocity, *force))))
                   & jnp.all(jnp.where(opened.east, True, velocity.east == 0.)) & jnp.all(jnp.where(opened.north, True, velocity.north == 0.))
                   & jnp.all(jnp.where(opened.east, True, force.east == 0.)) & jnp.all(jnp.where(opened.north, True, force.north == 0.)))
    final, east_mean, north_mean, eta_mean, residual, work, valid = jax.lax.scan(
        substep, (initial, jnp.zeros(shape, jnp.float64), jnp.zeros(shape, jnp.float64), jnp.zeros(shape[:2], jnp.float64), jnp.array(0.), jnp.array(0.), input_valid),
        xs=None, length=nsub)[0]
    physical_final = VelocityPair(final[0] / root_mass.east, final[1] / root_mass.north)
    final_eta = final[2] / (root_area * surface_scale)
    initial_energy = kinetic_energy(operators, initial_velocity) + .5 * gravity * jnp.sum(operators.area * eta ** 2)
    final_energy = kinetic_energy(operators, physical_final) + .5 * gravity * jnp.sum(operators.area * final_eta ** 2)
    change = final_energy - initial_energy
    energy_residual = change - work
    energy_scale = jnp.abs(initial_energy) + jnp.abs(final_energy) + jnp.abs(work)
    energy_relative = jnp.abs(energy_residual) / jnp.where(energy_scale > 0., energy_scale, 1.)
    stored = VelocityPair(*(jnp.where(mask, field, 0.).astype(velocity.east.dtype) for mask, field in zip(opened, physical_final)))
    cast_work = kinetic_energy(operators, VelocityPair(*(field.astype(jnp.float64) for field in stored))) - kinetic_energy(operators, physical_final)
    valid = valid & jnp.isfinite(energy_relative) & (energy_relative <= 1e-11)
    return PairedSurfaceResult(stored, final_eta, eta, east_mean, north_mean, eta_mean, physical_final,
                               residual, change, work, energy_residual, energy_relative, cast_work, valid)


def paired_momentum_surface_step(geometry, state, density_anomaly, dt, nsub,
                                 reference=None, gravity=9.81, pressure_gravity=None,
                                 rho0=1025., coriolis=None, volume_source=None, content_source=None,
                                 restart=30, maxiter=20):
    """Advance actual layer momentum and bounded inventories with paired mean Q."""
    volume = state.inventory.volume
    if state.inventory.content.dtype != jnp.float64:
        raise ValueError("paired content inventory requires explicit64 dtype")
    if content_source is not None:
        content_source = jnp.asarray(content_source)
        if content_source.shape != state.inventory.content.shape or content_source.dtype != jnp.float64:
            raise ValueError("paired content source must match64 inventory")
    source = jnp.zeros_like(volume) if volume_source is None else jnp.asarray(volume_source)
    if source.shape != volume.shape or source.dtype != jnp.float64:
        raise ValueError("paired cell volume source must match64 inventories")
    faces = momentum_geometry(geometry, volume)
    operators = paired_operators(geometry, volume, coriolis)
    pressure = hydrostatic_pressure_force(geometry, volume, density_anomaly, reference,
                                          gravity if pressure_gravity is None else pressure_gravity, rho0)
    held_force = VelocityPair(pressure.east * faces.east_volume, pressure.north * faces.north_volume)
    surface = advance_paired_surface(operators, _physical_surface_height(geometry, volume),
                                     VelocityPair(state.east_velocity, state.north_velocity), dt, nsub,
                                     gravity, held_force, source[..., 0], restart, maxiter)
    fluxes = closed_surface_fluxes(surface.mean_east, surface.mean_north)
    active = geometry._replace(east_area=faces.east_area, north_area=faces.north_area)
    transport = advance_bounded_contents(active, state.inventory, fluxes, dt, source, content_source)
    reconstruction = reconstruct_wet_fluxes(geometry, faces, fluxes)
    dual = reconstruct_half_prism_transport(geometry, volume, fluxes)
    physical = physical_velocity_from_fluxes(geometry, reconstruction, source)
    final_eta = _physical_surface_height(geometry, transport.state.volume)
    surface_error = jnp.max(jnp.abs(final_eta - surface.eta))
    surface_tolerance = 1e-12 + 1e-12 * jnp.maximum(jnp.max(jnp.abs(surface.eta)), jnp.max(jnp.abs(final_eta)))
    endpoint_operators = paired_operators(geometry, transport.state.volume, coriolis)
    endpoint_velocity = VelocityPair(*(field.astype(jnp.float64) for field in surface.velocity))
    moving_work = kinetic_energy(endpoint_operators, endpoint_velocity) - kinetic_energy(operators, endpoint_velocity)
    valid = (operators.valid & endpoint_operators.valid & pressure.valid & surface.valid & transport.valid
             & reconstruction.valid & dual.valid & physical.valid & (surface_error <= surface_tolerance)
             & jnp.all(source[..., 1:] == 0.))
    return PairedStepResult(LayerState(transport.state, *surface.velocity), pressure, held_force, surface,
                            transport, fluxes, reconstruction, dual, physical, surface_error, moving_work, valid)
