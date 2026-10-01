"""Conservative pair-limited consistent content on prescribed physical weak fluxes."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from research.experiments.material_rstar_coordinates.kernel import rstar_geometry
from research.experiments.material_rstar_coordinates.nodal_mass import (
    apply_nodal_mass,
    solve_nodal_mass,
)
from research.experiments.material_rstar_coordinates.sparse_diffusion import (
    accumulate_edges,
    limited_negative_exchange,
    neighbor_bounds,
)
from research.experiments.material_rstar_coordinates.weak_sparse import (
    directed_accumulation,
    mass_pair_entries,
    weak_coefficients,
    weak_sparse_rate,
)


class WeakStageResult(NamedTuple):
    content: jax.Array
    surface: jax.Array
    attempted_content: jax.Array
    high_content: jax.Array
    low_row_content: jax.Array
    source: jax.Array
    valid: jax.Array
    fraction: jax.Array
    minimum_limiter: jax.Array
    decomposition_residual: jax.Array
    mass_continuity_residual: jax.Array
    inventory_residual: jax.Array
    lower_bound: jax.Array
    upper_bound: jax.Array


def bounded_weak_euler(content, surface, velocities, source_rate, duration, params, depths, mass, graph):
    if content.dtype != jnp.float64 or surface.dtype != jnp.float64 or content.shape != params.wet_mask_z.shape:
        raise ValueError("bounded weak content requires the frozen float64 grid")
    if any(value.shape != content.shape or value.dtype != jnp.float64 for value in velocities):
        raise ValueError("bounded weak velocities must match the float64 nodal layout")
    geometry = rstar_geometry(surface, depths, params.dz_node, params.wet_mask_z)
    area = params.dx_2d * params.dy
    coefficients = weak_coefficients(velocities, surface, geometry, params, graph)
    endpoint = surface + duration * coefficients.surface_rate
    next_geometry = rstar_geometry(endpoint, depths, params.dz_node, params.wet_mask_z)
    wet = mass.wet
    active_content = jnp.where(wet, content, 0.)
    concentration = solve_nodal_mass(content, geometry, area, mass)
    current_mass, endpoint_mass = area[..., None] * geometry.thickness, area[..., None] * next_geometry.thickness
    denominator = jnp.where(wet & (endpoint_mass > 0.), endpoint_mass, 1.)
    rate = weak_sparse_rate(concentration, coefficients, graph)
    viscosity = jnp.maximum(0., jnp.maximum(-coefficients.forward, -coefficients.reverse))
    safe = concentration.ravel()
    difference = safe[graph.pair_left] - safe[graph.pair_right]
    diffusion = accumulate_edges(graph, -duration * viscosity * difference).reshape(content.shape)
    source = duration * jnp.broadcast_to(jnp.asarray(source_rate, dtype=content.dtype), content.shape)
    low_row = current_mass * concentration + duration * rate + diffusion + source
    high_content = active_content + duration * rate + source
    high = solve_nodal_mass(high_content, next_geometry, area, mass)
    old_pairs = mass_pair_entries(geometry, area, mass, graph)
    new_pairs = mass_pair_entries(next_geometry, area, mass, graph)
    new_difference = high.ravel()[graph.pair_left] - high.ravel()[graph.pair_right]
    anti = new_pairs * new_difference - old_pairs * difference + duration * viscosity * difference
    unlimited = accumulate_edges(graph, anti).reshape(content.shape)
    decomposition = endpoint_mass * high - low_row - unlimited
    decomposition_scale = (jnp.abs(endpoint_mass * high) + jnp.abs(low_row)
                           + accumulate_edges(graph, jnp.abs(new_pairs * new_difference) + jnp.abs(old_pairs * difference)
                                              + jnp.abs(duration * viscosity * difference), "unsigned").reshape(content.shape))
    support = jnp.abs(coefficients.forward) + jnp.abs(coefficients.reverse) + old_pairs
    lower, upper = neighbor_bounds(graph, safe, support)
    lower_content = denominator.ravel() * lower + source.ravel()
    upper_content = denominator.ravel() * upper + source.ravel()
    correction, minimum = limited_negative_exchange(graph, anti, low_row.ravel(), lower_content, upper_content)
    corrected = (low_row + correction.reshape(content.shape)) / denominator
    attempted = apply_nodal_mass(corrected, next_geometry, area, mass)
    decoded_attempt = solve_nodal_mass(attempted, next_geometry, area, mass)
    lower_bound = lower.reshape(content.shape) + source / denominator
    upper_bound = upper.reshape(content.shape) + source / denominator
    low_concentration = low_row / denominator
    eps = 64. * jnp.finfo(content.dtype).eps
    bound_floor = eps * (1. + jnp.abs(concentration) + jnp.abs(low_concentration) + jnp.abs(corrected)
                         + jnp.abs(decoded_attempt) + jnp.abs(source / denominator))
    outgoing = directed_accumulation(graph, coefficients.forward + viscosity, coefficients.reverse + viscosity).reshape(content.shape)
    fraction = jnp.max(jnp.where(wet, duration * outgoing / denominator, 0.))
    row_rate = coefficients.diagonal + directed_accumulation(graph, coefficients.forward, coefficients.reverse)
    column_rate = coefficients.diagonal + directed_accumulation(graph, coefficients.reverse, coefficients.forward)
    operator_scale = jnp.abs(coefficients.diagonal) + directed_accumulation(graph, jnp.abs(coefficients.reverse), jnp.abs(coefficients.forward))
    continuity = endpoint_mass - current_mass - duration * row_rate.reshape(content.shape)
    continuity_scale = jnp.abs(endpoint_mass) + jnp.abs(current_mass) + jnp.abs(duration * row_rate.reshape(content.shape))
    old_row = apply_nodal_mass(jnp.ones_like(content), geometry, area, mass)
    new_row = apply_nodal_mass(jnp.ones_like(content), next_geometry, area, mass)
    identity_valid = (jnp.all(jnp.abs(column_rate) <= eps * operator_scale)
                      & jnp.all(jnp.abs(old_row - current_mass) <= eps * (jnp.abs(old_row) + jnp.abs(current_mass)))
                      & jnp.all(jnp.abs(new_row - endpoint_mass) <= eps * (jnp.abs(new_row) + jnp.abs(endpoint_mass)))
                      & jnp.all(jnp.abs(continuity) <= eps * continuity_scale)
                      & jnp.all(jnp.abs(decomposition) <= eps * decomposition_scale))
    inventory = jnp.sum(attempted) - jnp.sum(active_content) - jnp.sum(source)
    inventory_floor = eps * (jnp.sum(jnp.abs(attempted)) + jnp.sum(jnp.abs(active_content)) + jnp.sum(jnp.abs(source)))
    active_inputs_valid = (jnp.all(jnp.where(wet, jnp.isfinite(content), True))
                           & jnp.all(jnp.stack([jnp.all(jnp.where(wet, jnp.isfinite(value), True)) for value in velocities])))
    source_valid = jnp.all(jnp.isfinite(source)) & jnp.all(jnp.where(wet, True, source == 0.))
    mass_valid = jnp.all(wet == (params.wet_mask_z > 0.)) & jnp.all(graph.wet == wet.ravel())
    bounds_valid = jnp.all(jnp.where(wet, jnp.isfinite(corrected) & jnp.isfinite(decoded_attempt) & jnp.isfinite(low_concentration)
                                   & (low_concentration >= lower_bound - bound_floor) & (low_concentration <= upper_bound + bound_floor)
                                   & (corrected >= lower_bound - bound_floor) & (corrected <= upper_bound + bound_floor)
                                   & (decoded_attempt >= lower_bound - bound_floor) & (decoded_attempt <= upper_bound + bound_floor), True))
    valid = (geometry.valid & next_geometry.valid & active_inputs_valid & source_valid & mass_valid & identity_valid & bounds_valid
             & jnp.isfinite(duration) & (duration > 0.) & (fraction >= 0.) & (fraction <= .5) & (jnp.abs(inventory) <= inventory_floor))
    return WeakStageResult(jnp.where(valid & wet, attempted, content), jnp.where(valid & (params.wet_mask > 0.), endpoint, surface),
                           attempted, high_content, low_row, source, valid, fraction, minimum, decomposition, continuity,
                           inventory, lower_bound, upper_bound)
