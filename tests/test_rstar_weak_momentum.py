"""Consistent kinetic mass: original work/rest gates and physical point accuracy."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_rstar_representation import _gauss_mass
from test_rstar_weak_transport import _weak_case, _weak_diagnostic

from config import RHO_0
from research.experiments.material_rstar_coordinates.nodal_mass import make_nodal_mass
from research.experiments.material_rstar_coordinates.pressure_accuracy import (
    pressure_accuracy_diagnostic,
)
from research.experiments.material_rstar_coordinates.weak_momentum import (
    consistent_pressure_force,
    velocity_kinetic_energy,
    velocity_pressure_power,
)

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


@pytest.mark.parametrize("stairs", [False, True])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("kind", ["zero", "constant", "affine", "random"])
def test_consistent_pressure_keeps_independent_work_and_original_physical_rest_gates(stairs, flat, kind):
    result, _ = _consistent_diagnostic(stairs, flat, kind)
    assert result["pressure_work_passed"]
    assert result["rhs_numpy_passed"] and result["eta_numpy_passed"]
    assert result["physical_affine_rest_passed"] is not False
    assert result["constant_tracer_geometry_passed"] is not False
    assert abs(result["global_inventory_residual_kg_per_s"]) <= result["global_inventory_floor_kg_per_s"]
    assert abs(result["global_volume_residual_m3_per_s"]) <= result["global_volume_floor_m3_per_s"]
    assert not result["production_promotion"]


def test_kinetic_velocity_derivative_and_pressure_power_match_independent_matrix():
    case, mass, matrix = _consistent_case()
    params, weak_params, geometry, basis, surface, density, content, velocities = case
    gradient = jax.grad(velocity_kinetic_energy)(velocities, geometry, weak_params, mass)
    for derivative, velocity in zip(gradient, velocities, strict=True):
        contributions = RHO_0 * matrix * np.asarray(velocity)[..., None, :]
        expected = np.sum(contributions, axis=-1)
        floor = 64. * np.finfo(float).eps * (1. + np.sum(np.abs(contributions), axis=-1))
        assert np.all(np.abs(np.asarray(derivative) - expected) <= floor)
    force = _compiled_force(density, content, surface, geometry, basis, weak_params, mass)
    terms = np.stack([RHO_0 * np.asarray(velocity)[..., :, None] * matrix * np.asarray(acceleration)[..., None, :]
                      for velocity, acceleration in zip(velocities, force, strict=True)])
    actual = float(velocity_pressure_power(velocities, force, geometry, weak_params, mass))
    assert abs(actual - float(terms.sum())) <= 64. * np.finfo(float).eps * float(np.sum(np.abs(terms)))
    zeros = tuple(jnp.zeros_like(value) for value in velocities)
    assert float(velocity_kinetic_energy(velocities, geometry, weak_params, mass)) > 0.
    assert float(velocity_kinetic_energy(zeros, geometry, weak_params, mass)) == 0.


def test_consistent_pressure_dry_sentinels_preserve_raw_output_bytes():
    case, mass, _ = _consistent_case()
    _, weak_params, geometry, basis, surface, density, content, velocities = case
    expected = _compiled_force(density, content, surface, geometry, basis, weak_params, mass)
    wet = np.asarray(mass.wet)
    for sentinel in (123., -1e6, np.nan):
        changed = jnp.asarray(np.where(wet, density, sentinel))
        poisoned_content = jnp.asarray(np.where(wet, content, sentinel))
        actual = _compiled_force(changed, poisoned_content, surface, geometry, basis, weak_params, mass)
        for current, reference in zip(actual, expected, strict=True):
            assert np.asarray(current).tobytes() == np.asarray(reference).tobytes()
        shifted = tuple(jnp.asarray(np.where(wet, value, sentinel)) for value in velocities)
        assert np.asarray(velocity_kinetic_energy(shifted, geometry, weak_params, mass)).tobytes() == np.asarray(velocity_kinetic_energy(velocities, geometry, weak_params, mass)).tobytes()


@pytest.mark.parametrize("consistent", [False, True])
def test_physical_nodal_order_gate_distinguishes_velocity_mass_without_excluding_endpoints(consistent):
    diagnostic, _ = pressure_accuracy_diagnostic(consistent)
    assert diagnostic["independent_functionals_passed"]
    assert diagnostic["horizontal_second_order_passed"]
    assert diagnostic["vertical_second_order_passed"] is consistent
    assert not diagnostic["production_promotion"]
