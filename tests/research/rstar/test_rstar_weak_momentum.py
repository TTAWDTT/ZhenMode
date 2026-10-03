"""Consistent kinetic mass: original work/rest gates and physical point accuracy."""

from tests.support.rstar.weak_momentum import _compiled_force, _consistent_case, _consistent_diagnostic

import jax

import jax.numpy as jnp

import numpy as np

import pytest

from tests.support.rstar.representation import _gauss_mass

from tests.support.rstar.weak_transport import _weak_case, _weak_diagnostic

from ocean_solver.config.definitions import RHO_0

from research.experiments.material_rstar_coordinates.nodal_mass import make_nodal_mass

from research.experiments.material_rstar_coordinates.pressure_accuracy import (
    pressure_accuracy_diagnostic,
)

from research.experiments.material_rstar_coordinates.weak_momentum import (
    consistent_pressure_force,
    velocity_kinetic_energy,
    velocity_pressure_power,
)

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
