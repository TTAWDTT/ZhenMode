"""Consistent kinetic mass: original work/rest gates and physical point accuracy."""

import jax
import jax.numpy as jnp
import numpy as np

from ocean_solver.config.definitions import RHO_0
from research.experiments.material_rstar_coordinates.nodal_mass import make_nodal_mass
from research.experiments.material_rstar_coordinates.weak_momentum import (
    consistent_pressure_force,
)
from tests.support.rstar.representation import _gauss_mass
from tests.support.rstar.weak_transport import _weak_case, _weak_diagnostic

_compiled_force = jax.jit(consistent_pressure_force)

def _consistent_case(stairs=True, flat=True, kind="random"):
    case = _weak_case(stairs=stairs, flat=flat, kind=kind, complete=True)
    params, _, geometry, _, surface, _, _, _ = case
    nodes = (np.asarray(geometry.node_depth) + np.asarray(surface)[..., None]) / np.asarray(geometry.scale)[..., None]
    nodes = np.where(np.asarray(params.wet_mask_z) > 0., nodes, 0.)
    nodes[..., 0] = 0.
    mass = make_nodal_mass(nodes, params)
    matrix, _ = _gauss_mass(nodes, np.asarray(params.dz_node), np.asarray(params.wet_mask_z) > 0.)
    actual_matrix = (np.asarray(params.dx_2d) * float(params.dy) * np.asarray(geometry.scale))[..., None, None] * matrix
    return case, mass, actual_matrix

def _consistent_diagnostic(stairs, flat, kind):
    case, mass, matrix = _consistent_case(stairs, flat, kind)
    params, weak_params, geometry, basis, surface, density, content, velocities = case
    result, arrays = _weak_diagnostic(False, stairs, flat, kind, complete=True)
    forces = _compiled_force(density, content, surface, geometry, basis, weak_params, mass)
    terms = np.stack([RHO_0 * np.asarray(velocity)[..., :, None] * matrix * np.asarray(force)[..., None, :]
                      for velocity, force in zip(velocities, forces, strict=True)])
    kinetic = float(np.sum(terms))
    potential_terms = (arrays["content_rate_kg_per_s"] * arrays["content_conjugate_m2_per_s2"],
                       arrays["surface_rate_m_per_s"] * arrays["surface_conjugate_J_per_m"])
    residual = kinetic + sum(float(np.sum(term)) for term in potential_terms)
    floor = 64. * np.finfo(float).eps * (float(np.sum(np.abs(terms))) + sum(float(np.sum(np.abs(term))) for term in potential_terms))
    maximum = max(float(jnp.max(jnp.abs(force))) for force in forces)
    result.update(velocity_mass_contract="consistent hat velocity kinetic mass",
                  pressure_work_residual_watts=residual, pressure_work_64eps_floor_watts=floor,
                  pressure_work_passed=bool(abs(residual) <= floor), maximum_force_m_per_s2=maximum,
                  physical_affine_rest_passed=bool(maximum <= result["rest_force_64eps_floor_m_per_s2"]) if result["rest_applicable"] else None)
    arrays.update(force_x_m_per_s2=np.asarray(forces[0]), force_y_m_per_s2=np.asarray(forces[1]),
                  numpy_actual_kinetic_matrix_m3=matrix)
    return result, arrays
