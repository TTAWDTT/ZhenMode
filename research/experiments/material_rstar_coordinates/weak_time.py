"""Stage-time-field Heun candidate; qualification is separate from its SSP form."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from research.experiments.material_rstar_coordinates.kernel import rstar_geometry
from research.experiments.material_rstar_coordinates.nodal_mass import (
    apply_nodal_mass,
    solve_nodal_mass,
)
from research.experiments.material_rstar_coordinates.weak_bounded import bounded_weak_euler


class WeakHeunResult(NamedTuple):
    content: jax.Array
    surface: jax.Array
    attempted_content: jax.Array
    attempted_surface: jax.Array
    source: jax.Array
    valid: jax.Array
    first_valid: jax.Array
    second_valid: jax.Array
    fraction: jax.Array
    minimum_limiter: jax.Array
    lower_bound: jax.Array
    upper_bound: jax.Array
    inventory_residual: jax.Array
    mass_combination_residual: jax.Array


def bounded_weak_heun(content, surface, time, duration, params, depths, mass, graph, fields):
    geometry = rstar_geometry(surface, depths, params.dz_node, params.wet_mask_z)
    velocities, source_rate = fields(time, content, surface, geometry)
    first = bounded_weak_euler(content, surface, velocities, source_rate, duration, params, depths, mass, graph)
    middle_geometry = rstar_geometry(first.surface, depths, params.dz_node, params.wet_mask_z)
    next_velocities, next_source_rate = fields(time + duration, first.content, first.surface, middle_geometry)
    second = bounded_weak_euler(first.content, first.surface, next_velocities, next_source_rate,
                                duration, params, depths, mass, graph)
    wet, ocean = mass.wet, params.wet_mask > 0.
    attempted = jnp.where(wet, .5 * (content + second.content), content)
    attempted_surface = jnp.where(ocean, .5 * (surface + second.surface), surface)
    source = .5 * (first.source + second.source)
    area = params.dx_2d * params.dy
    final_geometry = rstar_geometry(attempted_surface, depths, params.dz_node, params.wet_mask_z)
    stage_geometry = rstar_geometry(second.surface, depths, params.dz_node, params.wet_mask_z)
    original = solve_nodal_mass(content, geometry, area, mass)
    decoded = solve_nodal_mass(attempted, final_geometry, area, mass)
    decoded_second = solve_nodal_mass(second.content, stage_geometry, area, mass)
    before_mass = area[..., None] * geometry.thickness
    after_mass = area[..., None] * stage_geometry.thickness
    final_mass = area[..., None] * final_geometry.thickness
    denominator = jnp.where(wet & (before_mass + after_mass > 0.), before_mass + after_mass, 1.)
    lower = (before_mass * original + after_mass * second.lower_bound) / denominator
    upper = (before_mass * original + after_mass * second.upper_bound) / denominator
    convex = (before_mass * original + after_mass * decoded_second) / denominator
    eps = 64. * jnp.finfo(content.dtype).eps
    bound_floor = eps * (1. + jnp.abs(original) + jnp.abs(decoded) + jnp.abs(decoded_second))
    mass_residual = final_mass - .5 * (before_mass + after_mass)
    actual_row = apply_nodal_mass(jnp.ones_like(content), final_geometry, area, mass)
    mass_valid = (jnp.all(jnp.abs(mass_residual) <= eps * (jnp.abs(final_mass) + .5 * (jnp.abs(before_mass) + jnp.abs(after_mass))))
                  & jnp.all(jnp.abs(actual_row - final_mass) <= eps * (jnp.abs(actual_row) + jnp.abs(final_mass))))
    bounds_valid = jnp.all(jnp.where(wet, jnp.isfinite(decoded) & jnp.isfinite(convex)
                                    & (jnp.abs(decoded - convex) <= bound_floor)
                                    & (decoded >= lower - bound_floor) & (decoded <= upper + bound_floor), True))
    inventory = jnp.sum(jnp.where(wet, attempted - content, 0.)) - jnp.sum(source)
    inventory_floor = eps * (jnp.sum(jnp.abs(jnp.where(wet, attempted, 0.)))
                             + jnp.sum(jnp.abs(jnp.where(wet, content, 0.))) + jnp.sum(jnp.abs(source)))
    valid = (first.valid & second.valid & final_geometry.valid & stage_geometry.valid & mass_valid & bounds_valid
             & jnp.isfinite(time) & jnp.isfinite(time + duration) & (jnp.abs(inventory) <= inventory_floor))
    return WeakHeunResult(jnp.where(valid & wet, attempted, content),
                          jnp.where(valid & ocean, attempted_surface, surface), attempted, attempted_surface, source,
                          valid, first.valid, second.valid, jnp.maximum(first.fraction, second.fraction),
                          jnp.minimum(first.minimum_limiter, second.minimum_limiter), lower, upper, inventory, mass_residual)
