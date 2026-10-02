"""Same-reconstruction physical weak rates and separate pressure/rest gates."""

from tests.support.rstar.weak_transport import _compiled_force, _compiled_rates, _weak_case, _weak_diagnostic

from types import SimpleNamespace

import jax

import jax.numpy as jnp

import numpy as np

import pytest

from tests.support.rstar.metric_controls import _fixture, _geometry

from tests.support.rstar.pressure_work import _work

from tests.support.rstar.representation import _representation_case

from config import RHO_0

from research.experiments.material_rstar_coordinates.bed_completion import (
    REPRESENTATION_TAG,
    complete_bed_reference,
)

from research.experiments.material_rstar_coordinates.kernel import hydrostatic_pressure

from research.experiments.material_rstar_coordinates.nodal_mass import (
    apply_nodal_mass,
    make_nodal_mass,
)

from research.experiments.material_rstar_coordinates.pressure_work import potential_conjugates

from research.experiments.material_rstar_coordinates.weak_oracle import physical_weak_rates

from research.experiments.material_rstar_coordinates.weak_transport import (
    WeakParameters,
    consistent_potential_basis,
    weak_content_rate,
    weak_pressure_force,
)

@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("truncate", [False, True])
def test_physical_weak_rhs_matches_independent_five_point_gauss_and_global_inventory(flat, truncate):
    params, weak_params, geometry, _, surface, density, _, velocities = _weak_case(truncate=truncate, flat=flat, kind="random")
    expected, eta_rate, scale, eta_scale = physical_weak_rates(density, velocities, surface, geometry, params)
    actual = _compiled_rates(density, velocities, surface, geometry, weak_params)
    assert np.all(np.abs(np.asarray(actual[0]) - expected) <= 64. * np.finfo(float).eps * (1. + scale))
    assert np.all(np.abs(np.asarray(actual[1]) - eta_rate) <= 64. * np.finfo(float).eps * (1e-20 + eta_scale))
    assert abs(float(jnp.sum(actual[0]))) <= 64. * np.finfo(float).eps * np.sum(scale)

@pytest.mark.parametrize("flat", [False, True])
def test_weak_constant_content_rate_is_exact_actual_hat_mass_geometry_rate(flat):
    params, weak_params, geometry, _, surface, density, _, velocities = _weak_case(flat=flat, kind="constant")
    actual, eta_rate = _compiled_rates(density, velocities, surface, geometry, weak_params)
    _, _, scale, _ = physical_weak_rates(density, velocities, surface, geometry, params)
    expected = 1.5 * (params.dx_2d * params.dy)[..., None] * geometry.weights * eta_rate[..., None]
    assert np.all(np.abs(np.asarray(actual - expected)) <= 64. * np.finfo(float).eps * (1. + scale + np.abs(np.asarray(expected))))

@pytest.mark.parametrize("truncate", [False, True])
def test_weak_pressure_power_pairs_with_independent_rates_without_rest_promotion(truncate):
    params, weak_params, geometry, basis, surface, density, content, velocities = _weak_case(truncate=truncate)
    expected, eta_rate, _, _ = physical_weak_rates(density, velocities, surface, geometry, params)
    forces = _compiled_force(density, content, surface, geometry, basis, weak_params)
    conjugates = potential_conjugates(content, surface, geometry, basis, params)
    residual, floor = _work(forces, velocities, expected, eta_rate, conjugates, geometry, params)
    assert abs(residual) <= floor

def test_weak_traces_do_not_read_dry_density_or_velocity_sentinels():
    params, weak_params, geometry, basis, surface, density, content, velocities = _weak_case(kind="random")
    wet = np.asarray(params.wet_mask_z) > 0.
    original = _compiled_rates(density, velocities, surface, geometry, weak_params)
    force = _compiled_force(density, content, surface, geometry, basis, weak_params)
    for sentinel in (123., -1e6, np.nan):
        changed = jnp.asarray(np.where(wet, density, sentinel))
        shifted = tuple(jnp.asarray(np.where(wet, value, sentinel)) for value in velocities)
        actual = _compiled_rates(changed, shifted, surface, geometry, weak_params)
        for result, reference in zip(actual, original, strict=True):
            np.testing.assert_array_equal(result, reference)
        for result, reference in zip(_compiled_force(changed, content, surface, geometry, basis, weak_params), force, strict=True):
            np.testing.assert_array_equal(result, reference)
