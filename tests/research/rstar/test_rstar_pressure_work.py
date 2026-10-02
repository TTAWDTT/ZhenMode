"""Independent pressure-work controls; physical candidate gates remain separate."""

from tests.support.rstar.pressure_work import _candidate_diagnostics, _case, _numpy_basis, _numpy_rates, _vertical_refinement, _work

import jax

import jax.numpy as jnp

import numpy as np

import pytest

from tests.support.rstar.metric_controls import _fixture, _geometry

from config import G_EARTH, RHO_0

from jax_solver_global import _advection_scalar, _face_transport_divergence

from research.experiments.material_rstar_coordinates.kernel import (
    coordinate_pressure_gradient,
    hydrostatic_pressure,
    relative_vertical_transport,
)

from research.experiments.material_rstar_coordinates.pressure_work import (
    centered_content_rate,
    coordinate_layer_faces,
    energy_adjoint_force,
    make_potential_basis,
    paired_chain_rule_force,
    potential_conjugates,
    potential_energy,
    transpose_force,
)

@pytest.mark.parametrize("stairs", [False, True])
def test_pressure_potential_basis_matches_independent_gauss_moments_mass_and_derivatives(stairs):
    params, depths, geometry, basis, surface, _, content, _ = _case(stairs)
    mass, mean = _numpy_basis(depths, params)
    widths = np.asarray(params.dz_node) * np.asarray(params.wet_mask_z)
    assert np.max(np.abs(mass - widths)) <= 64. * np.finfo(float).eps * np.max(widths)
    assert np.max(np.abs(mean - np.asarray(basis.mean_depth))) <= 64. * np.finfo(float).eps * np.max(np.abs(mean))
    area = np.asarray(params.dx_2d) * float(params.dy)
    actual_mean = -np.asarray(surface)[..., None] + np.asarray(geometry.scale)[..., None] * mean
    expected = .5 * RHO_0 * G_EARTH * np.sum(area * np.asarray(surface) ** 2) - G_EARTH * np.sum(np.asarray(content) * actual_mean)
    actual = float(potential_energy(content, surface, geometry, basis, params))
    assert abs(actual - expected) <= 64. * np.finfo(float).eps * abs(expected)

    def energy(values, eta):
        return potential_energy(values, eta, _geometry(params, depths, eta), basis, params)

    automatic = jax.grad(energy, argnums=(0, 1))(content, surface)
    explicit = potential_conjugates(content, surface, geometry, basis, params)
    for actual, expected in zip(automatic, explicit, strict=True):
        assert float(jnp.max(jnp.abs(actual - expected))) <= 64. * np.finfo(float).eps * float(jnp.max(jnp.abs(expected)))

@pytest.mark.parametrize("stairs", [False, True])
@pytest.mark.parametrize("flat", [False, True])
def test_pressure_energy_adjoint_matches_explicit_transpose_and_independent_transport(stairs, flat, record_property):
    params, _, geometry, basis, surface, density, content, velocities = _case(stairs, flat)
    faces = coordinate_layer_faces(*velocities, geometry, params)
    rates = centered_content_rate(density, faces, geometry, params)
    independent = _numpy_rates(np.asarray(density), velocities, geometry, params)
    for actual, expected in zip(rates, independent, strict=True):
        assert float(jnp.max(jnp.abs(actual - expected))) <= 64. * np.finfo(float).eps * max(1e-30, float(np.max(np.abs(expected))))
    conjugates = potential_conjugates(content, surface, geometry, basis, params)
    explicit = jax.jit(lambda field: energy_adjoint_force(field, content, surface, geometry, basis, params))(density)
    automatic = transpose_force(density, conjugates, geometry, params)
    for actual, expected in zip(explicit, automatic, strict=True):
        assert float(jnp.max(jnp.abs(actual - expected))) <= 64. * np.finfo(float).eps * float(jnp.max(jnp.abs(expected)))
    residual, floor = _work(explicit, velocities, *independent, conjugates, geometry, params)
    record_property("reversible_work_residual_watts", residual)
    record_property("reversible_work_64eps_floor_watts", floor)
    assert abs(residual) <= floor

@pytest.mark.parametrize("kind", ["zero", "constant"])
def test_actual_width_paired_chain_rule_surface_work_closes(kind, record_property):
    params, _, geometry, basis, surface, density, content, velocities = _case(kind=kind)
    pressure = hydrostatic_pressure(density, surface, geometry, params)
    rates = _numpy_rates(np.asarray(density), velocities, geometry, params)
    forces = paired_chain_rule_force(pressure, density, geometry, params)
    conjugates = potential_conjugates(content, surface, geometry, basis, params)
    residual, floor = _work(forces, velocities, *rates, conjugates, geometry, params)
    record_property("paired_chain_surface_work_residual_watts", residual)
    record_property("paired_chain_surface_work_64eps_floor_watts", floor)
    assert abs(residual) <= floor

def test_pressure_density_force_and_centered_rates_do_not_read_finite_dry_sentinels():
    params, _, geometry, basis, surface, density, content, velocities = _case()
    evaluate = jax.jit(lambda field: energy_adjoint_force(field, content, surface, geometry, basis, params))
    reference = evaluate(density)
    for sentinel in (123., -1e6):
        actual = evaluate(jnp.where(params.wet_mask_z > 0., density, sentinel))
        for changed, expected in zip(actual, reference, strict=True):
            np.testing.assert_array_equal(changed, expected)

@pytest.mark.parametrize("sentinel", [123., -1e6, np.nan])
def test_potential_conjugate_ignores_dry_surface_values(sentinel):
    params, depths, geometry, basis, surface, _, content, _ = _case()
    changed = jnp.where(params.wet_mask > 0., surface, sentinel)

    def energy(eta):
        return potential_energy(content, eta, _geometry(params, depths, eta), basis, params)

    derivative = jax.grad(energy)(changed)
    assert bool(jnp.isfinite(energy(changed)))
    explicit = potential_conjugates(content, changed, geometry, basis, params)[1]
    np.testing.assert_array_equal(np.asarray(explicit)[np.asarray(params.wet_mask) == 0.], np.asarray(derivative)[np.asarray(params.wet_mask) == 0.])

def test_pressure_surface_weight_mismatch_is_detected_without_relaxing_work_gate():
    result = _candidate_diagnostics(True, False, "zero")["candidate_gates"]
    assert not result["existing_chain_rule"]["pressure_work_passed"]
    assert result["actual_width_chain_rule"]["pressure_work_passed"]
    assert result["energy_adjoint"]["pressure_work_passed"]

def test_energy_pairing_alone_does_not_qualify_affine_physical_rest_on_stairs():
    result = _candidate_diagnostics(True, True, "affine")["candidate_gates"]
    assert result["actual_width_chain_rule"]["rest_gate_passed"]
    assert not result["actual_width_chain_rule"]["pressure_work_passed"]
    assert result["energy_adjoint"]["pressure_work_passed"]
    assert not result["energy_adjoint"]["rest_gate_passed"]

def test_pressure_full_wet_vertical_refinement_does_not_pass_registered_second_order_gate():
    result = _vertical_refinement()
    assert not result["second_order_diagnostic_passed"]
    assert not result["changes_interface_gate"]
