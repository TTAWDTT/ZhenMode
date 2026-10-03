"""Original physical bed, explicit extra unknown and joint weak-interface gates."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from tests.support.rstar.metric_controls import _fixture
from tests.support.rstar.representation import _gauss_mass
from tests.support.rstar.weak_transport import _weak_case, _weak_diagnostic

from ocean_solver.config.definitions import G_EARTH, RHO_0
from research.experiments.material_rstar_coordinates.bed_completion import complete_bed_reference
from research.experiments.material_rstar_coordinates.nodal_mass import make_nodal_mass
from research.experiments.material_rstar_coordinates.pressure_work import potential_energy


def test_bed_completion_preserves_shape_old_wet_nodes_and_original_physical_bed():
    _, params, depths = _fixture()
    completed = complete_bed_reference(depths, params)
    old = np.asarray(params.wet_mask_z) > 0.
    wet = np.asarray(completed.wet) > 0.
    assert wet.shape == old.shape
    np.testing.assert_array_equal(np.asarray(completed.nodes)[old], np.broadcast_to(depths, old.shape)[old])
    np.testing.assert_array_equal(completed.bed, np.sum(np.asarray(params.dz_node) * old, axis=-1))
    assert np.all(old <= wet)
    assert np.all((wet.sum(axis=-1) - old.sum(axis=-1)) <= 1)
    assert bool(jnp.any(completed.added))
    assert np.all(np.asarray(completed.widths)[wet] > 0.)
    np.testing.assert_allclose(np.sum(completed.widths, axis=-1), completed.bed, rtol=64. * np.finfo(float).eps, atol=0.)


def test_completed_column_mass_matches_independent_quadrature_on_local_node_coordinates():
    _, params, depths = _fixture()
    completed = complete_bed_reference(depths, params)
    updated = SimpleNamespace(**{**vars(params), "wet_mask_z": completed.wet, "dz_node": completed.widths})
    mass = make_nodal_mass(completed.nodes, updated)
    matrix, _ = _gauss_mass(completed.nodes, np.asarray(completed.widths), np.asarray(completed.wet) > 0.)
    assert np.all(np.abs(np.diagonal(matrix, axis1=-2, axis2=-1) - mass.diagonal) <= 64. * np.finfo(float).eps * np.maximum(completed.widths, 1.))
    assert np.max(np.abs(np.diagonal(matrix, offset=1, axis1=-2, axis2=-1) - mass.off_diagonal)) <= 64. * np.finfo(float).eps * float(jnp.max(completed.widths))


def test_bed_completion_refuses_missing_slot_instead_of_truncating_column():
    _, params, depths = _fixture(stairs=False)
    widths = np.asarray(params.dz_node).copy()
    widths[..., -1] += 1.
    updated = SimpleNamespace(**{**vars(params), "dz_node": widths})
    with pytest.raises(ValueError, match="spare dry slot"):
        complete_bed_reference(depths, updated)


@pytest.mark.parametrize("flat", [False, True])
def test_completed_affine_profile_inventory_and_potential_match_same_physical_domain(flat):
    params, _, geometry, basis, surface, _, content, _ = _weak_case(flat=flat, complete=True)
    depth = np.asarray(basis.column_depth)
    eta = np.asarray(surface)
    area = np.asarray(params.dx_2d) * float(params.dy)
    wet = np.asarray(params.wet_mask) > 0.
    exact_mass = area * (1.5 * (depth + eta) + .001 * (depth ** 2 - eta ** 2))
    actual_mass = np.asarray(content).sum(axis=-1)
    assert np.all(np.abs(actual_mass[wet] - exact_mass[wet]) <= 64. * np.finfo(float).eps * np.abs(exact_mass[wet]))
    exact = .5 * RHO_0 * G_EARTH * np.sum(area[wet] * eta[wet] ** 2)
    exact -= G_EARTH * np.sum(area[wet] * (.75 * (depth[wet] ** 2 - eta[wet] ** 2) + .002 / 3. * (depth[wet] ** 3 + eta[wet] ** 3)))
    actual = float(potential_energy(content, surface, geometry, basis, params))
    assert abs(actual - exact) <= 64. * np.finfo(float).eps * abs(exact)


def test_completed_weak_interface_has_both_original_pressure_work_and_affine_rest_gates():
    result, _ = _weak_diagnostic(False, True, True, "affine", complete=True)
    assert result["pressure_work_passed"]
    assert result["physical_affine_rest_passed"]
    assert not result["production_promotion"]


def test_original_constant_tail_rest_failure_is_retained_after_weak_transport_change():
    result, _ = _weak_diagnostic(False, True, True, "affine")
    assert result["pressure_work_passed"]
    assert not result["physical_affine_rest_passed"]
