"""Representation-only localization; unchanged flux is not FE weak transport."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from ocean_solver.config.definitions import RHO_0
from research.experiments.material_rstar_coordinates.kernel import hydrostatic_pressure
from research.experiments.material_rstar_coordinates.pressure_work import (
    PotentialBasis,
    energy_adjoint_force,
    make_potential_basis,
    paired_chain_rule_force,
    potential_conjugates,
    transpose_force,
)
from tests.support.rstar.metric_controls import _geometry
from tests.support.rstar.pressure_work import _case, _numpy_rates, _work


def _gauss_mass(depths, widths, wet):
    matrix = np.zeros(wet.shape + (wet.shape[-1],))
    moment = np.zeros(wet.shape)
    quadrature, weights = np.polynomial.legendre.leggauss(3)
    node_depths = np.broadcast_to(depths, wet.shape)
    for column in np.ndindex(wet.shape[:2]):
        count = int(wet[column].sum())
        for level in range(count - 1):
            low, high = node_depths[column][level:level + 2]
            positions = .5 * ((high - low) * quadrature + high + low)
            measure = .5 * (high - low) * weights
            hats = ((high - positions) / (high - low), (positions - low) / (high - low))
            for local, hat in enumerate(hats):
                node = level + local
                moment[column + (node,)] += np.sum(measure * positions * hat)
                for other, other_hat in enumerate(hats):
                    matrix[column + (node, level + other)] += np.sum(measure * hat * other_hat)
        if count:
            low, high = node_depths[column][count - 1], widths[column].sum()
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
