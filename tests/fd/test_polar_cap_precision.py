"""Candidate filter precision gates, not physical moving-inventory qualification."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ocean_solver.numerics.horizontal import _apply_polar_cap


def _mask(rank):
    longitude = np.arange(180)[:, None, None]
    latitude = np.arange(14)[None, :, None]
    level = np.arange(4)[None, None, :]
    wet = (longitude < 19 + (17 * latitude + 7 * level) % 151).astype(np.float32)
    return wet[..., 0] if rank == 2 else wet


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("constant", [15., 34.7, -1.8])
def test_candidate_float32_polar_filter_preserves_a_constant_exactly(rank, constant):
    wet = jnp.asarray(_mask(rank))
    params = SimpleNamespace(polar_cap_rows=2, polar_cap_taper=3, process_time_scheme="symmetric_fast_v3")
    initial = jnp.full(wet.shape, constant, dtype=jnp.float32)
    apply = jax.jit(lambda value: _apply_polar_cap(value, wet, params))
    once = apply(initial)
    repeated = jax.jit(lambda value: jax.lax.fori_loop(0, 1000, lambda index, current: apply(current), value))(initial)
    selected = np.asarray(wet, dtype=bool)
    for result in (once, repeated):
        assert result.dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(result)[selected], np.asarray(initial)[selected])


def _reference_cap(field, wet, params):
    width = params.polar_cap_rows + params.polar_cap_taper
    weights = np.concatenate((np.ones(params.polar_cap_rows),
                              np.cos(np.linspace(0., np.pi / 2., params.polar_cap_taper + 2)[1:-1]) ** 2)).astype(field.dtype).astype(np.float64)
    output = field.copy()
    for band, fractions in ((slice(0, width), weights), (slice(-width, None), weights[::-1])):
        values = field[:, band].astype(np.float64) * wet[:, band]
        mean = np.sum(values, axis=0, keepdims=True) / np.maximum(np.sum(wet[:, band], axis=0, keepdims=True), 1.)
        fraction = fractions.reshape((1, width) + (1,) * (field.ndim - 2))
        output[:, band] = (values + fraction * (mean * wet[:, band] - values)).astype(field.dtype)
    return output


@pytest.mark.parametrize("rank", [2, 3])
def test_candidate_polar_filter_rounds_the_explicit_double_band_formula_to_float32(rank):
    wet = _mask(rank)
    params = SimpleNamespace(polar_cap_rows=2, polar_cap_taper=3, process_time_scheme="symmetric_fast_v3")
    initial = np.asarray(34.7 + np.random.default_rng(627).normal(size=wet.shape) * .03, dtype=np.float32)
    result = jax.jit(lambda value: _apply_polar_cap(value, jnp.asarray(wet), params))(jnp.asarray(initial))
    expected = _reference_cap(initial, wet, params)
    assert result.dtype == jnp.float32
    np.testing.assert_array_equal(result, expected)


def test_candidate_polar_filter_handles_empty_bands_and_dry_sentinels():
    wet = _mask(3)
    wet[:, :5] = 0.
    wet[:, -5:, -1] = 0.
    params = SimpleNamespace(polar_cap_rows=2, polar_cap_taper=3, process_time_scheme="symmetric_fast_v3")
    initial = np.full(wet.shape, 34.7, dtype=np.float32)
    initial[wet == 0.] = 1e20
    result = jax.jit(lambda value: _apply_polar_cap(value, jnp.asarray(wet), params))(jnp.asarray(initial))
    expected = _reference_cap(initial, wet, params)
    np.testing.assert_array_equal(result, expected)
    assert np.isfinite(result).all()


def test_candidate_float32_polar_filter_tangent_and_adjoint_are_consistent():
    wet = jnp.asarray(_mask(3))
    params = SimpleNamespace(polar_cap_rows=2, polar_cap_taper=3, process_time_scheme="symmetric_fast_v3")
    generator = np.random.default_rng(834)
    initial = jnp.asarray(generator.normal(size=wet.shape) + 34.7, dtype=jnp.float32)
    tangent = jnp.asarray(generator.normal(size=wet.shape), dtype=jnp.float32)
    dual = jnp.asarray(generator.normal(size=wet.shape), dtype=jnp.float32)
    apply = jax.jit(lambda value: _apply_polar_cap(value, wet, params))
    _, pushed = jax.jvp(apply, (initial,), (tangent,))
    _, pullback = jax.vjp(apply, initial)
    pulled = pullback(dual)[0]
    expected = _reference_cap(np.asarray(tangent), np.asarray(wet), params)
    np.testing.assert_allclose(pushed, expected, rtol=3e-6, atol=3e-6)
    first = np.sum(np.asarray(pushed, dtype=np.float64) * np.asarray(dual, dtype=np.float64))
    second = np.sum(np.asarray(tangent, dtype=np.float64) * np.asarray(pulled, dtype=np.float64))
    scale = np.sum(np.abs(np.asarray(pushed, dtype=np.float64) * np.asarray(dual, dtype=np.float64)))
    assert abs(first - second) <= 3e-6 * scale
    assert pushed.dtype == pulled.dtype == jnp.float32


def test_candidate_float32_polar_filter_tangent_matches_resolvable_finite_difference():
    wet = jnp.asarray(_mask(3))
    params = SimpleNamespace(polar_cap_rows=2, polar_cap_taper=3, process_time_scheme="symmetric_fast_v3")
    generator = np.random.default_rng(835)
    initial = jnp.asarray(generator.normal(size=wet.shape) * .1 + 34.7, dtype=jnp.float32)
    tangent = jnp.asarray(generator.normal(size=wet.shape), dtype=jnp.float32)
    apply = jax.jit(lambda value: _apply_polar_cap(value, wet, params))
    _, pushed = jax.jvp(apply, (initial,), (tangent,))
    interval = .01
    difference = np.asarray(apply(initial + interval * tangent), dtype=np.float64)
    difference -= np.asarray(apply(initial - interval * tangent), dtype=np.float64)
    difference /= 2. * interval
    assert np.linalg.norm(difference - np.asarray(pushed)) <= 1e-3 * np.linalg.norm(pushed)
