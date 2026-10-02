"""Common-depth hydrostatic pressure and actual C-grid layer momentum reference.

This migration stage freezes pressure/physical faces during a macrostep and
omits nonlinear momentum transport and full thermodynamics/physics. It is not
the completed production replacement. See cgrid_hydrostatic_momentum protocol.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from ocean_solver._compat import preserve_legacy_names
from ocean_solver.candidates.fv.barotropic import BarotropicResult, subcycle_barotropic
from ocean_solver.candidates.fv.fluxes import (
    HalfPrismTransport,
    WetFluxReconstruction,
    reconstruct_half_prism_transport,
    reconstruct_wet_fluxes,
)
from ocean_solver.candidates.fv.geometry import (
    ExtensiveState,
    TransportResult,
    VolumeFluxes,
    _physical_surface_height,
    closed_surface_fluxes,
    horizontal_momentum_geometry,
    match_column_transport,
)
from ocean_solver.candidates.fv.transport import _neighbor, advance_bounded_contents
from ocean_solver.candidates.fv.velocity import (
    PhysicalVelocityReconstruction,
    physical_velocity_from_fluxes,
)
from ocean_solver.configuration import OMEGA


class PressureForce(NamedTuple):
    east: jnp.ndarray
    north: jnp.ndarray
    east_area: jnp.ndarray
    north_area: jnp.ndarray
    valid: jnp.ndarray


class MomentumGeometry(NamedTuple):
    top: jnp.ndarray
    height: jnp.ndarray
    east_area: jnp.ndarray
    north_area: jnp.ndarray
    east_top: jnp.ndarray
    east_bottom: jnp.ndarray
    north_top: jnp.ndarray
    north_bottom: jnp.ndarray
    east_volume: jnp.ndarray
    north_volume: jnp.ndarray
    south_half_area: jnp.ndarray
    north_half_area: jnp.ndarray
    valid: jnp.ndarray


class RotationResult(NamedTuple):
    east_velocity: jnp.ndarray
    north_velocity: jnp.ndarray
    energy_relative_change: jnp.ndarray
    solve_relative_residual: jnp.ndarray
    valid: jnp.ndarray


class LayerState(NamedTuple):
    inventory: ExtensiveState
    east_velocity: jnp.ndarray
    north_velocity: jnp.ndarray


class MomentumResult(NamedTuple):
    state: LayerState
    pressure: PressureForce
    barotropic: BarotropicResult
    transport: TransportResult
    fluxes: VolumeFluxes
    flux_reconstruction: WetFluxReconstruction
    dual_transport: HalfPrismTransport
    physical_velocity: PhysicalVelocityReconstruction
    rotation_error: jnp.ndarray
    rotation_residual: jnp.ndarray
    surface_error: jnp.ndarray
    valid: jnp.ndarray


def momentum_geometry(geometry, volume=None):
    """Exact wet half-cell rectangles and shared physical face intervals.

    Latitude halves use spherical areas; velocities are zero outside their
    open-face wet interval. Only the top cell may move in this fixed-z stage.
    This defines the rotation mass, not full nonlinear dual-mass continuity.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("physical momentum geometry requires explicit JAX X64")
    for field in (geometry.area, geometry.thickness, geometry.interfaces, geometry.latitude_edges):
        if jnp.asarray(field).dtype != jnp.float64:
            raise ValueError("physical momentum geometry requires64 geometry dtype")
    area, base_height = jnp.asarray(geometry.area), jnp.asarray(geometry.thickness)
    volume = area[..., None] * base_height if volume is None else jnp.asarray(volume)
    if volume.shape != base_height.shape or volume.dtype != jnp.float64:
        raise ValueError("momentum volume must match physical cells with64 dtype")
    height = volume / area[..., None]
    eta = height[..., 0] - base_height[..., 0]
    top = jnp.broadcast_to(jnp.asarray(geometry.interfaces[:-1]), height.shape).at[..., 0].set(-eta)
    bottom = top + height
    horizontal = horizontal_momentum_geometry(geometry)
    face_areas, face_tops, face_bottoms, face_heights = [], [], [], []
    for axis, base_area in enumerate((geometry.east_area, geometry.north_area)):
        shared_top = jnp.maximum(top, _neighbor(top, axis, 1))
        shared_bottom = jnp.minimum(bottom, _neighbor(bottom, axis, 1))
        common = jnp.maximum(shared_bottom - shared_top, 0.)
        base_common = jnp.minimum(base_height, _neighbor(base_height, axis, 1))
        opened = jnp.asarray(base_area) > 0.
        face_areas.append(jnp.where(opened, jnp.asarray(base_area) * common / jnp.where(base_common > 0., base_common, 1.), 0.))
        face_tops.append(shared_top)
        face_bottoms.append(shared_bottom)
        face_heights.append(jnp.where(opened, common, 0.))
    east_volume = horizontal.east_dual_area[..., None] * face_heights[0]
    north_volume = horizontal.north_dual_area[..., None] * face_heights[1]
    lower_error = jnp.max(jnp.abs(height[..., 1:] - base_height[..., 1:]) / jnp.maximum(base_height[..., 1:], 1.), initial=0.)
    valid = (jnp.all(jnp.isfinite(volume)) & jnp.all(jnp.where(base_height > 0., volume > 0., volume == 0.))
             & (lower_error <= 1e-12) & jnp.all(jnp.isfinite(east_volume)) & jnp.all(jnp.isfinite(north_volume))
             & jnp.all(east_volume >= 0.) & jnp.all(north_volume >= 0.))
    return MomentumGeometry(top, height, *face_areas, face_tops[0], face_bottoms[0], face_tops[1], face_bottoms[1],
                            east_volume, north_volume, horizontal.south_half_area, horizontal.north_half_area, valid)


def hydrostatic_pressure_force(geometry, volume, density_anomaly, reference=None,
                               gravity=9.81, rho0=1025.):
    """Analytic pressure averages on the SAME physical wet depth on both sides.

    Contact force divides by physical wet momentum mass, not point distance.
    Density anomaly is a cell mean in kg/m3. reference is global coefficients
    of a+b*z+c*z^2 in positive-down z; its exact cell means are subtracted for
    reconstruction, but its physical surface load is restored. Geometry,
    inventory and pressure arithmetic require explicit X64/64-bit inputs.
    Only the top volume moves; dry density sentinels are ignored.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("hydrostatic pressure requires explicit JAX X64")
    volume, density = jnp.asarray(volume), jnp.asarray(density_anomaly)
    shape = geometry.thickness.shape
    if volume.shape != shape or density.shape != shape:
        raise ValueError("pressure volume/density must match physical cells")
    if volume.dtype != jnp.float64 or density.dtype != jnp.float64 or jnp.asarray(geometry.interfaces).dtype != jnp.float64:
        raise ValueError("pressure requires64 geometry interfaces, volume and density dtype")
    coefficients = jnp.zeros(3, jnp.float64) if reference is None else jnp.asarray(reference, jnp.float64)
    gravity, rho0 = (jnp.asarray(value, jnp.float64) for value in (gravity, rho0))
    if coefficients.shape != (3,) or gravity.ndim != 0 or rho0.ndim != 0:
        raise ValueError("reference needs three coefficients; gravity/rho0 must be scalar")
    base_height = jnp.asarray(geometry.thickness, jnp.float64)
    wet = base_height > 0.
    faces = momentum_geometry(geometry, volume)
    height, top = faces.height, faces.top
    eta = height[..., 0] - base_height[..., 0]
    center = top + .5 * height
    reference_mean = coefficients[0] + coefficients[1] * center + coefficients[2] * (center ** 2 + height ** 2 / 12.)
    density = jnp.where(wet, density, 0.)
    residual = jnp.where(wet, density - reference_mean, 0.)
    previous_center, next_center = _neighbor(center, 2, -1), _neighbor(center, 2, 1)
    left_open, right_open = wet & _neighbor(wet, 2, -1), wet & _neighbor(wet, 2, 1)
    distance_left, distance_right = center - previous_center, next_center - center
    gradient_left = (residual - _neighbor(residual, 2, -1)) / jnp.where(left_open, distance_left, 1.)
    gradient_right = (_neighbor(residual, 2, 1) - residual) / jnp.where(right_open, distance_right, 1.)
    denominator = distance_left + distance_right
    central = (distance_right * gradient_left + distance_left * gradient_right) / jnp.where(left_open & right_open, denominator, 1.)
    slope = jnp.where(left_open & right_open, central,
                      jnp.where(left_open, gradient_left, jnp.where(right_open, gradient_right, 0.)))
    integrated = gravity * residual * height
    pressure_top = jnp.concatenate((jnp.zeros_like(integrated[..., :1]), jnp.cumsum(integrated[..., :-1], axis=-1)), axis=-1)
    surface_depth = -eta
    reference_load = -gravity * (coefficients[0] * surface_depth + .5 * coefficients[1] * surface_depth ** 2
                                  + coefficients[2] * surface_depth ** 3 / 3.)

    def pressure_average(interval_top, interval_bottom, cell_top, cell_height, cell_residual, cell_slope, cell_pressure):
        start, end = interval_top - cell_top, interval_bottom - cell_top
        return cell_pressure + gravity * ((cell_residual - .5 * cell_slope * cell_height) * .5 * (start + end)
                                          + cell_slope * (start ** 2 + start * end + end ** 2) / 6.)

    forces, face_areas = [], []
    for axis, (face_area, interval_top, interval_bottom, mass) in enumerate(
            ((faces.east_area, faces.east_top, faces.east_bottom, faces.east_volume),
             (faces.north_area, faces.north_top, faces.north_bottom, faces.north_volume))):
        pressure_left = pressure_average(interval_top, interval_bottom, top, height, residual, slope, pressure_top) + reference_load[..., None]
        pressure_right = pressure_average(interval_top, interval_bottom, _neighbor(top, axis, 1),
                                          _neighbor(height, axis, 1), _neighbor(residual, axis, 1),
                                          _neighbor(slope, axis, 1), _neighbor(pressure_top, axis, 1)) + _neighbor(reference_load, axis, 1)[..., None]
        force = jnp.where(face_area > 0., -face_area * (pressure_right - pressure_left) / (rho0 * jnp.where(mass > 0., mass, 1.)), 0.)
        forces.append(force)
        face_areas.append(face_area)
    valid = (faces.valid & jnp.all(jnp.isfinite(density))
             & jnp.all(jnp.isfinite(coefficients)) & jnp.isfinite(gravity) & (gravity >= 0.)
             & jnp.isfinite(rho0) & (rho0 > 0.)
             & jnp.all(jnp.where(wet, rho0 + density > 0., True))
             & jnp.all(jnp.isfinite(forces[0])) & jnp.all(jnp.isfinite(forces[1]))
             & jnp.all(jnp.isfinite(face_areas[0])) & jnp.all(jnp.isfinite(face_areas[1])))
    return PressureForce(*forces, *face_areas, valid)


def _rotation_system(geometry, east_velocity, north_velocity, coriolis, volume=None):
    if not jax.config.jax_enable_x64:
        raise ValueError("weighted rotation requires explicit JAX X64 arithmetic")
    east_velocity, north_velocity = jnp.asarray(east_velocity), jnp.asarray(north_velocity)
    if east_velocity.shape != geometry.thickness.shape or north_velocity.shape != east_velocity.shape:
        raise ValueError("layer face velocities must match physical cells")
    if east_velocity.dtype not in (jnp.float32, jnp.float64) or north_velocity.dtype != east_velocity.dtype:
        raise ValueError("layer face velocities must share float32/64 dtype")
    faces = momentum_geometry(geometry, volume)
    east_weight, north_weight = faces.east_volume, faces.north_volume
    east_open, north_open = east_weight > 0., north_weight > 0.
    root_east = jnp.where(east_open, jnp.sqrt(jnp.where(east_open, east_weight, 1.)), 0.)
    root_north = jnp.where(north_open, jnp.sqrt(jnp.where(north_open, north_weight, 1.)), 0.)
    clean_east = jnp.where(east_open, east_velocity.astype(jnp.float64), 0.)
    clean_north = jnp.where(north_open, north_velocity.astype(jnp.float64), 0.)
    if coriolis is None:
        latitude = jnp.asarray(geometry.latitude_edges, jnp.float64)
        centers = .5 * (latitude[:-1] + latitude[1:])
        upper = jnp.broadcast_to(OMEGA * (jnp.sin(centers) + jnp.sin(latitude[1:]))[None, :], geometry.area.shape)
        lower = jnp.broadcast_to(OMEGA * (jnp.sin(centers) + jnp.sin(latitude[:-1]))[None, :], geometry.area.shape)
    else:
        center_f = jnp.asarray(coriolis, jnp.float64)
        if center_f.ndim == 0:
            center_f = jnp.full(geometry.area.shape, center_f)
        if center_f.shape != geometry.area.shape:
            raise ValueError("Coriolis must be scalar or horizontal cell field")
        upper, lower = center_f, center_f
    coefficients = []
    for longitude, latitude_shift, half_area, coriolis_mean in (
            (0, 0, faces.north_half_area, upper), (1, 0, _neighbor(faces.north_half_area, 0, 1), _neighbor(upper, 0, 1)),
            (0, -1, faces.south_half_area, lower), (1, -1, _neighbor(faces.south_half_area, 0, 1), _neighbor(lower, 0, 1))):
        neighbor_top, neighbor_bottom, neighbor_root = faces.north_top, faces.north_bottom, root_north
        if longitude:
            neighbor_top, neighbor_bottom, neighbor_root = (_neighbor(field, 0, 1) for field in (neighbor_top, neighbor_bottom, neighbor_root))
        if latitude_shift:
            neighbor_top, neighbor_bottom, neighbor_root = (_neighbor(field, 1, -1) for field in (neighbor_top, neighbor_bottom, neighbor_root))
        common_height = jnp.maximum(jnp.minimum(faces.east_bottom, neighbor_bottom) - jnp.maximum(faces.east_top, neighbor_top), 0.)
        pair_open = east_open & (neighbor_root > 0.)
        denominator = jnp.where(pair_open, root_east * neighbor_root, 1.)
        coefficients.append(jnp.where(pair_open, .5 * half_area[..., None] * coriolis_mean[..., None] * common_height / denominator, 0.))
    upper_west, upper_east, lower_west, lower_east = coefficients

    def cross(north):
        north = jnp.where(north_open, north, 0.)
        south = _neighbor(north, 1, -1)
        return jnp.where(east_open, upper_west * north + upper_east * _neighbor(north, 0, 1)
                         + lower_west * south + lower_east * _neighbor(south, 0, 1), 0.)

    def transpose(east):
        east = jnp.where(east_open, east, 0.)
        upper_pair = upper_west * east + _neighbor(upper_east * east, 0, -1)
        lower_pair = _neighbor(lower_west * east + _neighbor(lower_east * east, 0, -1), 1, 1)
        return jnp.where(north_open, upper_pair + lower_pair, 0.)

    valid = (faces.valid & jnp.all(jnp.isfinite(clean_east)) & jnp.all(jnp.isfinite(clean_north))
             & jnp.all(jnp.isfinite(east_weight)) & jnp.all(jnp.isfinite(north_weight))
             & jnp.all(east_weight >= 0.) & jnp.all(north_weight >= 0.)
             & jnp.all(jnp.isfinite(upper)) & jnp.all(jnp.isfinite(lower)))
    return root_east, root_north, clean_east, clean_north, cross, transpose, valid


def coriolis_tendency(geometry, east_velocity, north_velocity, coriolis=None, volume=None):
    """Physical rectangle-overlap rotation; returned tendencies use64 guard math."""
    root_east, root_north, east, north, cross, transpose, unused_valid = _rotation_system(geometry, east_velocity, north_velocity, coriolis, volume)
    return (cross(root_north * north) / jnp.where(root_east > 0., root_east, 1.),
            -transpose(root_east * east) / jnp.where(root_north > 0., root_north, 1.))


def rotate_coriolis(geometry, east_velocity, north_velocity, dt, coriolis=None, maxiter=100, volume=None):
    """Implicit midpoint rotation, independently checked residual/weighted energy.

    Energy uses actual wet half-cell rectangle volumes, not face area*distance
    or a square-root neighbor-mass interpolation.32 velocity storage is explicit;
    arithmetic/solve are64. No damping or force clipping is applied.
    """
    if not isinstance(maxiter, int) or isinstance(maxiter, bool) or maxiter < 1:
        raise ValueError("rotation maxiter must be a positive static integer")
    root_east, root_north, east, north, cross, transpose, valid = _rotation_system(geometry, east_velocity, north_velocity, coriolis, volume)
    dt = jnp.asarray(dt, jnp.float64)
    if dt.ndim != 0:
        raise ValueError("rotation dt must be scalar")
    half = .5 * dt
    initial_east, initial_north = root_east * east, root_north * north
    cross_north = cross(initial_north)
    rhs = initial_east - half ** 2 * cross(transpose(initial_east)) + 2. * half * cross_north

    def matrix(east_scaled):
        return east_scaled + half ** 2 * cross(transpose(east_scaled))

    rhs_norm = jnp.linalg.norm(rhs)
    solution, unused_info = cg(matrix, rhs, tol=1e-13, atol=0., maxiter=maxiter)
    rotated_north = initial_north - half * transpose(initial_east + solution)
    dtype = jnp.asarray(east_velocity).dtype
    final_east = (solution / jnp.where(root_east > 0., root_east, 1.)).astype(dtype)
    final_north = (rotated_north / jnp.where(root_north > 0., root_north, 1.)).astype(dtype)
    final_east = jnp.where(root_east > 0., final_east, 0.)
    final_north = jnp.where(root_north > 0., final_north, 0.)
    residual_norm = jnp.linalg.norm(rhs - matrix(solution))
    relative_residual = residual_norm / jnp.where(rhs_norm > 0., rhs_norm, 1.)
    arithmetic_floor = 32. * jnp.finfo(jnp.float64).eps * (jnp.linalg.norm(initial_east) + jnp.abs(half) * jnp.linalg.norm(cross_north))
    initial_energy = jnp.sum(initial_east ** 2 + initial_north ** 2)
    final_energy = jnp.sum((root_east * final_east.astype(jnp.float64)) ** 2 + (root_north * final_north.astype(jnp.float64)) ** 2)
    energy_error = jnp.abs(final_energy - initial_energy) / jnp.where(initial_energy > 0., initial_energy, 1.)
    tolerance = 2e-6 if dtype == jnp.float32 else 1e-12
    valid = (valid & jnp.isfinite(dt) & (dt >= 0.) & jnp.all(jnp.isfinite(final_east)) & jnp.all(jnp.isfinite(final_north))
             & (residual_norm <= 1e-12 * rhs_norm + arithmetic_floor) & (energy_error <= tolerance))
    return RotationResult(final_east, final_north, energy_error, relative_residual, valid)


def linear_momentum_surface_step(geometry, state, density_anomaly, dt, nsub,
                                 reference=None, gravity=9.81, pressure_gravity=None,
                                 rho0=1025., coriolis=None, volume_source=None, content_source=None):
    """Actual layer pressure/rotation coupled to shared fast mean Q and V/N.

    Pressure and physical face intersections are frozen during this reference
    macrostep. This is NOT full nonlinear momentum or nonlinear split physics.
    Pressure's layer-area mean enters fast momentum once; deviations enter3D.
    Fast wave/pressure arithmetic64,3D velocity storage32/64, inventory64.
    Only top volume sources; callers MUST check all returned validity flags.
    """
    if not isinstance(nsub, int) or isinstance(nsub, bool) or nsub < 1:
        raise ValueError("nsub must be a positive static integer")
    density_gravity = gravity if pressure_gravity is None else pressure_gravity
    pressure = hydrostatic_pressure_force(geometry, state.inventory.volume, density_anomaly,
                                           reference, density_gravity, rho0)
    active = geometry._replace(east_area=pressure.east_area, north_area=pressure.north_area)
    first = rotate_coriolis(geometry, state.east_velocity, state.north_velocity, .5 * dt, coriolis, volume=state.inventory.volume)
    areas = (pressure.east_area, pressure.north_area)
    total_areas = tuple(jnp.sum(area, axis=-1) for area in areas)
    velocities = (first.east_velocity.astype(jnp.float64), first.north_velocity.astype(jnp.float64))
    mean_velocities = tuple(jnp.sum(area * velocity, axis=-1) / jnp.where(total > 0., total, 1.)
                            for area, velocity, total in zip(areas, velocities, total_areas))
    mean_forces = tuple(jnp.sum(area * force, axis=-1) / jnp.where(total > 0., total, 1.)
                        for area, force, total in zip(areas, (pressure.east, pressure.north), total_areas))
    deviations = tuple(jnp.where(area > 0., force - mean[..., None], 0.)
                        for area, force, mean in zip(areas, (pressure.east, pressure.north), mean_forces))
    midpoint_layers = tuple(velocity + .5 * dt * deviation for velocity, deviation in zip(velocities, deviations))
    eta = _physical_surface_height(geometry, state.inventory.volume)
    source = jnp.zeros_like(state.inventory.volume) if volume_source is None else jnp.asarray(volume_source, jnp.float64)
    if source.shape != state.inventory.volume.shape:
        raise ValueError("layer volume source must match physical cells")
    barotropic = subcycle_barotropic(active, eta, *mean_velocities, dt / nsub, nsub,
                                      gravity=gravity, volume_source=source[..., 0],
                                      east_acceleration=mean_forces[0], north_acceleration=mean_forces[1])
    matched = tuple(match_column_transport(area, layer, mean)
                    for area, layer, mean in zip(areas, midpoint_layers, (barotropic.mean_east, barotropic.mean_north)))
    fluxes = closed_surface_fluxes(matched[0].flux, matched[1].flux)
    flux_reconstruction = reconstruct_wet_fluxes(geometry, momentum_geometry(geometry, state.inventory.volume), fluxes)
    dual_transport = reconstruct_half_prism_transport(geometry, state.inventory.volume, fluxes)
    physical_velocity = physical_velocity_from_fluxes(geometry, flux_reconstruction, source)
    transport = advance_bounded_contents(active, state.inventory, fluxes, dt, volume_source, content_source)
    endpoint_layers = tuple(jnp.where(area > 0., velocity + dt * deviation + (endpoint - initial_mean)[..., None], 0.).astype(state.east_velocity.dtype)
                            for area, velocity, deviation, endpoint, initial_mean
                            in zip(areas, velocities, deviations, (barotropic.east_velocity, barotropic.north_velocity), mean_velocities))
    second = rotate_coriolis(geometry, *endpoint_layers, .5 * dt, coriolis, volume=state.inventory.volume)
    final_eta = _physical_surface_height(geometry, transport.state.volume)
    surface_error = jnp.max(jnp.abs(final_eta - barotropic.eta))
    surface_tolerance = 1e-12 + 1e-12 * jnp.maximum(jnp.max(jnp.abs(eta)), jnp.max(jnp.abs(barotropic.eta)))
    valid = (pressure.valid & first.valid & second.valid & barotropic.valid & transport.valid & flux_reconstruction.valid & dual_transport.valid & physical_velocity.valid
             & matched[0].valid & matched[1].valid & (surface_error <= surface_tolerance)
             & jnp.all(source[..., 1:] == 0.))
    final_state = LayerState(transport.state, second.east_velocity, second.north_velocity)
    return MomentumResult(final_state, pressure, barotropic, transport, fluxes, flux_reconstruction, dual_transport, physical_velocity,
                           jnp.maximum(first.energy_relative_change, second.energy_relative_change),
                           jnp.maximum(first.solve_relative_residual, second.solve_relative_residual), surface_error, valid)

preserve_legacy_names(globals(), 'cgrid_momentum')
