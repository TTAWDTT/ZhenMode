"""Consistent hat mass algebra; no weak transport or pressure qualification."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.support.rstar.metric_controls import _fixture, _geometry
from tests.support.rstar.representation import _gauss_mass

from research.experiments.material_rstar_coordinates.nodal_mass import (
    apply_nodal_mass,
    make_nodal_mass,
    solve_nodal_mass,
)


@pytest.mark.parametrize("stairs", [False, True])
def test_tridiagonal_mass_and_solve_match_independent_gauss_and_dense_columns(stairs):
    _, params, depths = _fixture(stairs=stairs)
    wet = np.asarray(params.wet_mask_z) > 0.
    mass = make_nodal_mass(depths, params)
    matrix, _ = _gauss_mass(depths, np.asarray(params.dz_node) * wet, wet)
    generator = np.random.default_rng(930200)
    eta = jnp.asarray(generator.uniform(-3., 2., wet.shape[:2])) * params.wet_mask
    geometry = _geometry(params, depths, eta)
    area = params.dx_2d * params.dy
    field = generator.normal(size=wet.shape)
    expected = np.asarray(area * geometry.scale)[..., None] * np.einsum("...ij,...j->...i", matrix, np.where(wet, field, 0.))
    apply = jax.jit(lambda values: apply_nodal_mass(values, geometry, area, mass))
    solve = jax.jit(lambda values: solve_nodal_mass(values, geometry, area, mass))
    result = apply(jnp.asarray(field))
    scale = np.asarray(area * geometry.scale)[..., None] * np.einsum("...ij,...j->...i", np.abs(matrix), np.abs(np.where(wet, field, 0.)))
    floor = 64. * np.finfo(float).eps * (1. + scale)
    assert np.all(np.abs(np.asarray(result) - expected) <= floor)
    reconstructed = np.zeros_like(matrix)
    for level in range(wet.shape[-1]):
        reconstructed[..., level, level] = np.asarray(mass.diagonal)[..., level]
        if level + 1 < wet.shape[-1]:
            reconstructed[..., level, level + 1] = np.asarray(mass.off_diagonal)[..., level]
            reconstructed[..., level + 1, level] = np.asarray(mass.off_diagonal)[..., level]
    assert np.all(np.abs(reconstructed - matrix) <= 64. * np.finfo(float).eps * np.maximum(np.asarray(params.dz_node)[..., None], 1.))
    np.testing.assert_allclose(solve(result), np.where(wet, field, 0.), rtol=64. * np.finfo(float).eps, atol=64. * np.finfo(float).eps)
    for sentinel in (123., -1e6, np.nan):
        np.testing.assert_array_equal(apply(jnp.asarray(np.where(wet, field, sentinel))), result)
        np.testing.assert_array_equal(solve(jnp.asarray(np.where(wet, np.asarray(result), sentinel))), solve(result))
    for column in np.ndindex(wet.shape[:2]):
        count = int(wet[column].sum())
        if count:
            reference = np.linalg.solve(matrix[column][:count, :count], expected[column][:count] / float((area * geometry.scale)[column]))
            np.testing.assert_allclose(np.asarray(solve(result))[column][:count], reference, rtol=64. * np.finfo(float).eps, atol=64. * np.finfo(float).eps)


def test_nodal_mass_geometry_derivatives_and_transpose_are_supported():
    _, params, depths = _fixture()
    mass = make_nodal_mass(depths, params)
    area = params.dx_2d * params.dy
    generator = np.random.default_rng(930201)
    eta = jnp.asarray(generator.uniform(-3., 2., params.wet_mask.shape)) * params.wet_mask
    field = jnp.asarray(generator.normal(size=params.wet_mask_z.shape)) * params.wet_mask_z
    content = apply_nodal_mass(field, _geometry(params, depths, eta), area, mass)

    def inverse(surface):
        return solve_nodal_mass(content, _geometry(params, depths, surface), area, mass)

    direction = jnp.asarray(generator.normal(size=eta.shape)) * params.wet_mask
    _, tangent = jax.jvp(inverse, (eta,), (direction,))
    delta = 1e-3
    finite = (inverse(eta + delta * direction) - inverse(eta - delta * direction)) / (2. * delta)
    np.testing.assert_allclose(tangent, finite, rtol=2e-7, atol=1e-12)
    cotangent = jnp.asarray(generator.normal(size=field.shape)) * params.wet_mask_z
    adjoint = jax.vjp(inverse, eta)[1](cotangent)[0]
    forward, reverse = float(jnp.sum(tangent * cotangent)), float(jnp.sum(adjoint * direction))
    floor = 64. * np.finfo(float).eps * float(jnp.sum(jnp.abs(tangent * cotangent)) + jnp.sum(jnp.abs(adjoint * direction)))
    assert abs(forward - reverse) <= floor


def test_invalid_geometry_is_not_clipped_or_accepted_by_mass_algebra():
    _, params, depths = _fixture()
    mass = make_nodal_mass(depths, params)
    depth = jnp.sum(params.dz_node * params.wet_mask_z, axis=-1)
    geometry = _geometry(params, depths, -1.01 * depth)
    assert not bool(geometry.valid)
    content = apply_nodal_mass(jnp.ones(params.wet_mask_z.shape), geometry, params.dx_2d * params.dy, mass)
    assert bool(jnp.all(jnp.where(mass.wet, content < 0., True)))


def test_positive_consistent_content_is_not_a_nodal_positivity_guarantee():
    _, params, depths = _fixture(stairs=False)
    mass = make_nodal_mass(depths, params)
    geometry = _geometry(params, depths, jnp.zeros(params.wet_mask.shape))
    pulse = jnp.zeros(params.wet_mask_z.shape).at[..., 2].set(1.)
    result = solve_nodal_mass(pulse, geometry, params.dx_2d * params.dy, mass)
    assert float(jnp.min(result)) < 0.
    assert float(jnp.max(result)) > 0.


@pytest.mark.parametrize("case", ["unsorted_depth", "nonfinite_depth", "nonzero_top", "fractional_wet", "nonfinite_width"])
def test_nodal_mass_constructor_rejects_invalid_reference_inputs(case):
    _, params, depths = _fixture()
    depths = depths.copy()
    values = dict(vars(params))
    if case == "unsorted_depth":
        depths[2] = depths[1]
    elif case == "nonfinite_depth":
        depths[-1] = np.nan
    elif case == "nonzero_top":
        depths[0] = 1.
    elif case == "fractional_wet":
        wet = np.asarray(params.wet_mask_z).copy()
        wet[0, 0, 1] = .5
        values["wet_mask_z"] = wet
    else:
        width = np.asarray(params.dz_node).copy()
        width[0, 0, 1] = np.nan
        values["dz_node"] = width
    with pytest.raises(ValueError, match="nodal mass requires"):
        make_nodal_mass(depths, SimpleNamespace(**values))
