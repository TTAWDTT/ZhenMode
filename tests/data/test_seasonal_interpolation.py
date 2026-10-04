"""Calendar boundaries checked against independently specified month values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zhenmode.model.inputs.forcing.seasonal import (
    interp_monthly_field,
    interp_monthly_field_jit,
    interp_seasonal_wind,
    interp_seasonal_wind_jit,
)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("blend,day,expected", [
    (0., 0., 0.), (0., 15., 0.), (0., 30., 1.),
    (0., 359., 11.), (0., 360., 0.), (0., 720., 0.),
    (5., 0., 5.5), (5., 1., 3.3), (5., 15., 0.), (5., 29., .3),
    (5., 30., .5), (5., 31., .7), (5., 359., 7.7), (5., 360., 5.5),
])
def test_monthly_calendar_host_and_traced_backends(dtype, blend, day, expected):
    fields = np.broadcast_to(np.arange(12, dtype=dtype)[:, None, None], (12, 2, 3)).copy()
    wind = np.stack((fields, -fields), axis=1)
    advance = jax.jit(lambda day, width: (
        interp_monthly_field_jit(jnp.asarray(fields), day, width),
        interp_seasonal_wind_jit(jnp.asarray(wind), day, width),
    ))
    device_field, device_wind = advance(jnp.asarray(day, dtype=dtype),
                                        jnp.asarray(blend, dtype=dtype))
    values = [interp_monthly_field(fields, day, blend), device_field,
              *interp_seasonal_wind(list(wind), day, blend), *device_wind]
    for index, actual in enumerate(values):
        actual = np.asarray(actual)
        assert actual.shape == (2, 3)
        assert actual.dtype == np.dtype(dtype)
        assert np.isfinite(actual).all()
        sign = -1. if index in (3, 5) else 1.
        np.testing.assert_allclose(actual, sign * expected,
                                   atol=2e-6 if dtype == "float32" else 1e-12, rtol=0.)
