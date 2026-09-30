"""Independent physical nodal accuracy of the completed-bed weak pressure."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from config import G_EARTH, R_EARTH, RHO_0
from research.experiments.material_rstar_coordinates.kernel import rstar_geometry
from research.experiments.material_rstar_coordinates.nodal_mass import (
    apply_nodal_mass,
    make_nodal_mass,
)
from research.experiments.material_rstar_coordinates.weak_momentum import (
    consistent_pressure_force,
)
from research.experiments.material_rstar_coordinates.weak_transport import (
    WeakParameters,
    consistent_potential_basis,
    weak_pressure_force,
)

_compiled_force = jax.jit(weak_pressure_force)
_compiled_consistent_force = jax.jit(consistent_pressure_force)


def pressure_accuracy_case(nx, node_count, consistent=False):
    depth = 2000.
    nodes = np.linspace(0., depth, node_count)
    widths = np.concatenate((np.diff(nodes) / 2., [0.]))
    widths += np.concatenate(([0.], np.diff(nodes) / 2.))
    latitude = np.linspace(-np.pi / 6., np.pi / 6., 4)
    longitude = 2. * np.pi * np.arange(nx) / nx
    wet = jnp.ones((nx, len(latitude), node_count))
    dx = np.broadcast_to(R_EARTH * np.cos(latitude) * 2. * np.pi / nx, wet.shape[:2]).copy()
    dy = R_EARTH * (latitude[1] - latitude[0])
    params = SimpleNamespace(wet_mask_z=wet, wet_mask=wet[..., 0],
                             dz_node=jnp.broadcast_to(jnp.asarray(widths), wet.shape),
                             dx_2d=jnp.asarray(dx), dy=float(dy), cos_lat=jnp.asarray(np.cos(latitude)))
    surface = jnp.zeros(wet.shape[:2])
    geometry = rstar_geometry(surface, jnp.asarray(nodes), params.dz_node, wet)
    mass = make_nodal_mass(nodes, params)
    basis = consistent_potential_basis(nodes, params, geometry, mass)
    density = 1.5 + .002 * geometry.node_depth + .2 * jnp.sin(jnp.asarray(longitude))[:, None, None] * (1. + geometry.node_depth / depth)
    area = params.dx_2d * params.dy
    content = apply_nodal_mass(density, geometry, area, mass)
    weak_params = WeakParameters(*(getattr(params, name) for name in WeakParameters._fields))
    arguments = density, content, surface, geometry, basis, weak_params
    actual = _compiled_consistent_force(*arguments, mass) if consistent else _compiled_force(*arguments)
    forces = tuple(np.asarray(value) for value in actual)
    factor = -G_EARTH * .2 * np.cos(longitude)[:, None] / (RHO_0 * R_EARTH * np.cos(latitude)[None, :])
    point = factor[..., None] * (nodes + .5 * nodes ** 2 / depth)
    projected_profile = np.zeros(node_count)
    kinetic_matrix = np.zeros((node_count, node_count))
    gauss, weights = np.polynomial.legendre.leggauss(5)
    for level, (left, right) in enumerate(zip(nodes[:-1], nodes[1:], strict=True)):
        positions = .5 * (left + right + (right - left) * gauss)
        measures = .5 * (right - left) * weights
        right_hat = (positions - left) / (right - left)
        hats = np.stack((1. - right_hat, right_hat))
        kinetic_matrix[level:level + 2, level:level + 2] += (hats * measures) @ hats.T
        profile = positions + .5 * positions ** 2 / depth
        projected_profile[level] += np.sum(measures * profile * (1. - right_hat))
        projected_profile[level + 1] += np.sum(measures * profile * right_hat)
    decoded_profile = np.linalg.solve(kinetic_matrix, projected_profile) if consistent else projected_profile / widths
    projected = factor[..., None] * decoded_profile
    discrete_factor = np.sin(2. * np.pi / nx) / (2. * np.pi / nx)
    nodal_mass = np.asarray(area)[..., None] * np.asarray(geometry.thickness)

    def error(expected, selected=slice(None)):
        difference = (forces[0] - expected)[..., selected] ** 2 + forces[1][..., selected] ** 2
        weight = nodal_mass[..., selected]
        return float(np.sqrt(np.sum(weight * difference) / np.sum(weight)))

    functional_error = max(float(np.max(np.abs(forces[0] - projected * discrete_factor))), float(np.max(np.abs(forces[1]))))
    functional_floor = 64. * np.finfo(float).eps * G_EARTH * depth * float(jnp.max(jnp.abs(density))) / (RHO_0 * float(np.min(dx)))
    result = {"longitude_count": nx, "latitude_count": len(latitude), "node_count": node_count,
              "vertical_point_l2_m_per_s2": error(point * discrete_factor),
              "vertical_endpoint_l2_m_per_s2": error(point * discrete_factor, [0, -1]),
              "vertical_interior_l2_m_per_s2": error(point * discrete_factor, slice(1, -1)),
              "horizontal_projected_l2_m_per_s2": error(projected),
              "physical_point_l2_m_per_s2": error(point),
              "independent_functional_max_error_m_per_s2": functional_error,
              "independent_functional_64eps_floor_m_per_s2": functional_floor,
              "independent_functional_passed": bool(functional_error <= functional_floor)}
    arrays = {"force_x_m_per_s2": forces[0], "force_y_m_per_s2": forces[1],
              "analytic_point_force_m_per_s2": point, "analytic_centered_point_force_m_per_s2": point * discrete_factor,
              "numpy_hat_projected_force_m_per_s2": projected, "centered_angular_factor": discrete_factor,
              "node_depth_m": np.asarray(geometry.node_depth), "thickness_m": np.asarray(geometry.thickness),
              "area_m2": np.asarray(area), "density_kg_per_m3": np.asarray(density)}
    arrays["numpy_reference_kinetic_matrix_m"] = kinetic_matrix
    return result, arrays


def pressure_accuracy_diagnostic(consistent=False):
    records, arrays = [], {}
    for nx, count in ((16, 9), (16, 17), (16, 33), (32, 17), (64, 17)):
        record, witness = pressure_accuracy_case(nx, count, consistent)
        records.append(record)
        arrays[f"nx{nx}_nz{count}"] = witness
    vertical = [record for record in records if record["longitude_count"] == 16]
    horizontal = [record for record in records if record["node_count"] == 17]

    def ratios(selected, field):
        return [before[field] / after[field] for before, after in zip(selected[:-1], selected[1:], strict=True)]

    vertical_ratios = ratios(vertical, "vertical_point_l2_m_per_s2")
    horizontal_ratios = ratios(horizontal, "horizontal_projected_l2_m_per_s2")
    return {"records": records, "vertical_ratios": vertical_ratios, "horizontal_ratios": horizontal_ratios,
            "vertical_second_order_passed": all(3.3 <= ratio <= 4.5 for ratio in vertical_ratios),
            "horizontal_second_order_passed": all(3.3 <= ratio <= 4.5 for ratio in horizontal_ratios),
            "independent_functionals_passed": all(record["independent_functional_passed"] for record in records),
            "velocity_mass_contract": "consistent hat velocity kinetic mass" if consistent else "lumped nodal velocity kinetic mass", "production_promotion": False,
            "accepted_ocean_steps": 0}, arrays
