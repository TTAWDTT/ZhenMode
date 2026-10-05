"""Unmodified Fortran oracle, independent units and common-pressure controls."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zhenmode.model.solver.numerics.backend import jax, jnp
from zhenmode.model.solver.physics import teos10
from zhenmode.model.solver.physics.eos import _density_anomaly

REFERENCE_SHA = 'd6965433e4955ce2d933a78bcd30c492fae888b06bb77458ec9708853b9159ec'


def reference():
    directory = Path(__file__).resolve().parents[1] / 'support'
    payload = (directory/'teos10_reference.json').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == REFERENCE_SHA
    result = json.loads(payload)
    assert hashlib.sha256((directory/'teos10_reference.f90').read_bytes()).hexdigest() == result['driver_sha256']
    return np.asarray(result['rows'], dtype=float)


@pytest.mark.parametrize('precision,rho_atol,derivative_atol', [
    ('float64', 2e-10, 2e-10), ('float32', 5e-4, 1e-5),
])
def test_independent_gsw_density_and_analytic_derivatives(precision, rho_atol, derivative_atol):
    expected = reference()
    sa, ct, p = jnp.asarray(expected[:, :3], dtype=getattr(jnp, precision)).T
    teos10.validate_state(sa, ct, p)
    rho = jax.jit(teos10.density)(sa, ct, p)
    derivatives = jax.jit(teos10.density_derivatives)(sa, ct, p)
    np.testing.assert_allclose(rho, expected[:, 3], rtol=0, atol=rho_atol)
    np.testing.assert_allclose(jnp.stack(derivatives, axis=-1), expected[:, 4:7],
                               rtol=0, atol=derivative_atol)
    np.testing.assert_allclose(teos10.reference_salinity(sa), expected[:, 7], rtol=0,
                               atol=1e-5 if precision == 'float32' else 2e-13)
    assert rho.dtype == getattr(jnp, precision)


def test_scalar_broadcast_derivatives_are_elementwise_not_sums():
    expected = reference()[7:9]
    derivatives = jax.jit(teos10.density_derivatives)(35.16504, jnp.array([10., 11.]), 500.)
    assert all(value.shape == (2,) for value in derivatives)
    np.testing.assert_allclose(jnp.stack(derivatives, axis=-1), expected[:, 4:7], rtol=0, atol=2e-10)


def test_common_pressure_controls_convection():
    expected = reference()
    contrast = jax.jit(teos10.parcel_density_contrast)(35.16504, 10., 35.16504, 11., 500.)
    assert contrast == pytest.approx(expected[7, 3]-expected[8, 3], abs=2e-10)
    assert contrast > 0  # colder upper parcel, warmer lower parcel: unstable
    false_test = teos10.density(35.16504, 10., 0.) - teos10.density(35.16504, 11., 1000.)
    assert false_test == pytest.approx(expected[9, 3]-expected[10, 3], abs=2e-10)
    assert false_test < 0  # different in-situ pressures falsely suggest stability


def test_neutral_gradient_removes_pressure_compressibility():
    expected = reference()[7]
    actual = teos10.neutral_density_gradient(35.16504, 10., 500., .1, 1.25)
    assert actual == pytest.approx(expected[4]*.1 + expected[5]*1.25, abs=2e-10)
    pressure = jnp.array([0., 1000., 4000.])
    np.testing.assert_array_equal(teos10.neutral_density_gradient(35.16504, 10., pressure, 0., 0.), 0.)
    assert float(teos10.density(35.16504, 10., pressure)[-1]) > float(teos10.density(35.16504, 10., pressure)[0]) + 10


def test_legacy_eos_remains_linear():
    t, s = np.array([14., 20., 5.]), np.array([34., 36., 38.])
    expected = 1025 * (-2e-4 * (t-15.) + 7.6e-4 * (s-35.))
    np.testing.assert_array_equal(_density_anomaly(jnp.asarray(t), jnp.asarray(s),
                                                  SimpleNamespace(T_ref=15., S_ref=35.)), expected)
    legacy_fresh_water = 1025 * (1 + 2e-4 * 15 - 7.6e-4 * 35)
    assert abs(float(teos10.density(0., 0., 0.)) - legacy_fresh_water) > .5


@pytest.mark.parametrize('inputs,reason', [
    ((-1., 10., 0.), 'SA'),
    ((50., 10., 0.), 'SA'),
    ((35., 290., 0.), 'CT'),  # Kelvin is not degC
    ((35., 10., 1e6), 'dbar'),  # a typical Pa input cannot pass this range
    ((35., 10., -1.), 'dbar'),
    ((35., float('nan'), 0.), 'finite'),
    ((True, 10., 0.), 'finite real'),
    ((np.ma.array([35.], mask=[True]), 10., 0.), 'masked'),
    (([], [], []), 'empty'),
])
def test_host_preflight_refuses_invalid_values_without_clipping(inputs, reason):
    with pytest.raises(ValueError, match=reason):
        teos10.validate_state(*inputs)


def test_host_preflight_refuses_incompatible_shapes():
    with pytest.raises(ValueError, match='shape mismatch'):
        teos10.validate_state(np.ones((2, 3))*35, np.ones((4,)), 0.)
