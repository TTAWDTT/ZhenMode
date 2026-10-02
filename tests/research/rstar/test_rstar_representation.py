"""Representation-only localization; unchanged flux is not FE weak transport."""

from tests.support.rstar.representation import _gauss_mass, _representation_case, _representation_diagnostic

from types import SimpleNamespace

import jax

import jax.numpy as jnp

import numpy as np

import pytest

from tests.support.rstar.metric_controls import _geometry

from tests.support.rstar.pressure_work import _case, _numpy_rates, _work

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
