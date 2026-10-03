"""Consistent velocity kinetic contract for instantaneous weak pressure only."""
import jax
import jax.numpy as jnp

from ocean_solver.config.definitions import RHO_0
from research.experiments.material_rstar_coordinates.nodal_mass import (
    apply_nodal_mass,
    solve_nodal_mass,
)
from research.experiments.material_rstar_coordinates.pressure_work import potential_conjugates
from research.experiments.material_rstar_coordinates.weak_transport import weak_content_rate


def consistent_pressure_force(density, content, surface, geometry, basis, params, mass):
    def transport(velocities):
        return weak_content_rate(density, velocities, surface, geometry, params)

    zeros = jnp.zeros_like(density)
    conjugates = potential_conjugates(content, surface, geometry, basis, params)
    dual = jax.linear_transpose(transport, (zeros, zeros))(conjugates)[0]
    area = params.dx_2d * params.dy
    return tuple(-solve_nodal_mass(value, geometry, area, mass) / RHO_0 for value in dual)


def velocity_kinetic_energy(velocities, geometry, params, mass):
    area = params.dx_2d * params.dy
    return .5 * RHO_0 * sum(jnp.sum(jnp.where(mass.wet, velocity, 0.) * apply_nodal_mass(velocity, geometry, area, mass))
                            for velocity in velocities)


def velocity_pressure_power(velocities, accelerations, geometry, params, mass):
    area = params.dx_2d * params.dy
    return RHO_0 * sum(jnp.sum(jnp.where(mass.wet, acceleration, 0.) * apply_nodal_mass(velocity, geometry, area, mass))
                       for velocity, acceleration in zip(velocities, accelerations, strict=True))
