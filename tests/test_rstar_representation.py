"""Representation-only localization; unchanged flux is not FE weak transport."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_rstar_metric_controls import _geometry
from test_rstar_pressure_work import _case, _numpy_rates, _work

from config import G_EARTH, RHO_0
from research.experiments.material_rstar_coordinates.kernel import hydrostatic_pressure
from research.experiments.material_rstar_coordinates.pressure_work import (
    PotentialBasis,
    energy_adjoint_force,
    make_potential_basis,
    paired_chain_rule_force,
    potential_conjugates,
    potential_energy,
    transpose_force,
)


def _gauss_mass(depths, widths, wet):
    matrix = np.zeros(wet.shape + (wet.shape[-1],))
    moment = np.zeros(wet.shape)
    quadrature, weights = np.polynomial.legendre.leggauss(3)
    for column in np.ndindex(wet.shape[:2]):
        count = int(wet[column].sum())
        for level in range(count - 1):
            low, high = depths[level:level + 2]
            positions = .5 * ((high - low) * quadrature + high + low)
            measure = .5 * (high - low) * weights
            hats = ((high - positions) / (high - low), (positions - low) / (high - low))
            for local, hat in enumerate(hats):
                node = level + local
                moment[column + (node,)] += np.sum(measure * positions * hat)
                for other, other_hat in enumerate(hats):
                    matrix[column + (node, level + other)] += np.sum(measure * hat * other_hat)
        if count:
            low, high = depths[count - 1], widths[column].sum()
            positions = .5 * ((high - low) * quadrature + high + low)
            measure = .5 * (high - low) * weights
            matrix[column + (count - 1, count - 1)] += measure.sum()
            moment[column + (count - 1,)] += np.sum(measure * positions)
    return matrix, moment


def _representation_case(truncate, consistent, flat):
    params, depths, _, _, surface, _, _, velocities = _case(True, flat, "affine")
    wet = np.asarray(params.wet_mask_z) > 0.
    widths = np.asarray(params.dz_node).copy() * wet
    original_depth = widths.sum(axis=-1)
    if truncate:
        for column in np.ndindex(wet.shape[:2]):
            count = int(wet[column].sum())
            if count:
                widths[column + (count - 1,)] -= widths[column].sum() - depths[count - 1]
        params = SimpleNamespace(**{**vars(params), "dz_node": jnp.asarray(widths)})
    geometry = _geometry(params, depths, surface)
    density = (1.5 + .002 * geometry.node_depth) * params.wet_mask_z
    area = np.asarray(params.dx_2d) * float(params.dy)
    matrix, moment = _gauss_mass(depths, widths, wet)
    mean = np.zeros(wet.shape)
    if consistent:
        for column in np.ndindex(wet.shape[:2]):
            count = int(wet[column].sum())
            if count:
                mean[column][:count] = np.linalg.solve(matrix[column][:count, :count], moment[column][:count])
        content = area[..., None] * np.asarray(geometry.scale)[..., None] * np.einsum("...ij,...j->...i", matrix, np.asarray(density))
        basis = PotentialBasis(jnp.asarray(mean), jnp.asarray(widths.sum(axis=-1)))
    else:
        content = area[..., None] * np.asarray(geometry.thickness) * np.asarray(density)
        basis = make_potential_basis(depths, params)
    return params, depths, geometry, basis, surface, density, jnp.asarray(content), velocities, matrix, moment, original_depth


@pytest.mark.parametrize("truncate", [False, True])
@pytest.mark.parametrize("consistent", [False, True])
def test_representation_quadrature_mass_energy_and_actual_conjugates(truncate, consistent):
    params, depths, geometry, basis, surface, density, content, _, matrix, moment, _ = _representation_case(truncate, consistent, True)
    wet = np.asarray(params.wet_mask_z) > 0.
    widths = np.asarray(params.dz_node) * wet
    floor = 64. * np.finfo(float).eps * max(float(widths.max()), 1.)
    assert np.max(np.abs(matrix.sum(axis=-1) - widths)) <= floor
    assert np.max(np.abs(matrix - matrix.swapaxes(-1, -2))) <= floor
    for column in np.ndindex(wet.shape[:2]):
        count = int(wet[column].sum())
        if count:
            assert np.linalg.eigvalsh(matrix[column][:count, :count]).min() > 0.
    if consistent:
        assert np.max(np.abs(np.einsum("...ij,...j->...i", matrix, np.asarray(basis.mean_depth)) - moment)) <= floor * depths[-1]
        if truncate:
            assert np.max(np.abs(np.asarray(basis.mean_depth)[wet] - np.broadcast_to(depths, wet.shape)[wet])) <= floor
    area = np.asarray(params.dx_2d) * float(params.dy)
    scale = np.asarray(geometry.scale)
    actual_moment = -np.asarray(surface)[..., None] * widths + scale[..., None] * moment
    expected = .5 * RHO_0 * G_EARTH * np.sum(area * np.asarray(surface) ** 2)
    expected -= G_EARTH * np.sum(area[..., None] * scale[..., None] * np.asarray(density) * actual_moment)
    actual = float(potential_energy(content, surface, geometry, basis, params))
    assert abs(actual - expected) <= 64. * np.finfo(float).eps * abs(expected)

    def energy(saved, eta):
        return potential_energy(saved, eta, _geometry(params, depths, eta), basis, params)

    automatic = jax.grad(energy, argnums=(0, 1))(content, surface)
    explicit = potential_conjugates(content, surface, geometry, basis, params)
    for calculated, oracle in zip(explicit, automatic, strict=True):
        np.testing.assert_allclose(calculated, oracle, rtol=64. * np.finfo(float).eps, atol=1e-12)


@pytest.mark.parametrize("truncate", [False, True])
def test_consistent_content_changes_local_semantics_not_column_total_or_same_profile_energy(truncate):
    lumped = _representation_case(truncate, False, True)
    consistent = _representation_case(truncate, True, True)
    old, new = np.asarray(lumped[6]), np.asarray(consistent[6])
    floor = 64. * np.finfo(float).eps * np.sum(np.abs(old), axis=-1)
    assert np.all(np.abs(old.sum(axis=-1) - new.sum(axis=-1)) <= floor)
    assert np.max(np.abs(old - new)) > np.max(floor)
    energies = [float(potential_energy(case[6], case[4], case[2], case[3], case[0])) for case in (lumped, consistent)]
    assert abs(energies[0] - energies[1]) <= 64. * np.finfo(float).eps * abs(energies[0])


def _representation_diagnostic(truncate, consistent, flat):
    params, _, geometry, basis, surface, density, content, velocities, matrix, moment, old_depth = _representation_case(truncate, consistent, flat)
    pressure = hydrostatic_pressure(density, surface, geometry, params)
    rates = _numpy_rates(np.asarray(density), velocities, geometry, params)
    conjugates = potential_conjugates(content, surface, geometry, basis, params)
    candidates = {"actual_width_chain_rule": paired_chain_rule_force(pressure, density, geometry, params),
                  "energy_adjoint": energy_adjoint_force(density, content, surface, geometry, basis, params)}
    oracle = transpose_force(density, conjugates, geometry, params)
    rest_floor = (64. * np.finfo(float).eps * float(jnp.max(jnp.abs(pressure))) / RHO_0
                  * max(float(jnp.max(params.inv_dx)), float(params.inv_dy)))
    results = {}
    for name, forces in candidates.items():
        residual, floor = _work(forces, velocities, *rates, conjugates, geometry, params)
        maximum = max(float(jnp.max(jnp.abs(force))) for force in forces)
        results[name] = {"pressure_work_residual_watts": residual, "pressure_work_64eps_floor_watts": floor,
                         "pressure_work_passed": bool(abs(residual) <= floor), "maximum_force_m_per_s2": maximum,
                         "rest_force_64eps_floor_m_per_s2": rest_floor, "rest_gate_applicable": flat,
                         "rest_passed": bool(maximum <= rest_floor) if flat else None}
    arrays = {"mass_matrix_m": matrix, "hat_first_moment_m2": moment, "content_kg": content,
              "reference_energy_conjugate_depth_m": basis.mean_depth, "surface_m": surface,
              "density_kg_per_m3": density, "original_bed_m": old_depth, "diagnostic_bed_m": basis.column_depth}
    return {"truncated_bed": truncate, "consistent_mass": consistent, "flat_eta": flat,
            "maximum_removed_bed_depth_m": float(np.max(old_depth - np.asarray(basis.column_depth))),
            "adjoint_oracle_max_force_difference_m_per_s2": max(float(jnp.max(jnp.abs(value - reference))) for value, reference in zip(candidates["energy_adjoint"], oracle, strict=True)),
            "candidates": results, "unchanged_flux_is_consistent_weak_transport": False,
            "production_promotion": False}, {name: np.asarray(value) for name, value in arrays.items()}
