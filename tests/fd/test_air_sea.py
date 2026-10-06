"""Independent author-Fortran coefficients and analytic SI/sign controls."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from zhenmode.model.solver.numerics.backend import jax, jnp
from zhenmode.model.solver.physics.air_sea import (
    AirState,
    ncar_coefficients10m,
    open_water_fluxes,
    saturation_specific_humidity,
)


def test_author_fortran_reference():
    support = Path(__file__).resolve().parents[1] / 'support'
    reference = json.loads((support / 'ncar_reference.json').read_text())
    assert hashlib.sha256((support / 'ncar_reference.f90').read_bytes()).hexdigest() == reference['driver_sha256']
    rows = reference['cases']
    inputs = [jnp.array([row[name] for row in rows]) for name in (
        'theta_air_k', 'humidity_air', 'temperature_surface_k', 'humidity_surface', 'scalar_wind')]
    for actual, key in zip(ncar_coefficients10m(*inputs), ('cd', 'ch', 'ce'), strict=True):
        # Upstream karman=0.4 is a default-real literal promoted to real(8):
        # 0.4000000059604645. Our SI constant is exact fp64 0.4. This accounts
        # for ~3e-8 relative coefficient differences; no formula is refitted.
        np.testing.assert_allclose(actual, [row[key] for row in rows], rtol=1e-7, atol=1e-14)
    np.testing.assert_allclose(saturation_specific_humidity(inputs[2], 101325),
                               inputs[3], rtol=1e-12, atol=1e-15)


def weather(**changes):
    values = dict(temperature_k=290., specific_humidity=.005, pressure_pa=101325.,
                  wind_u=6., wind_v=2., shortwave_down=200., longwave_down=320.,
                  rain=1e-5, snow=0., runoff=0., calving=0.)
    return AirState(**(values | changes))


def test_heat_signs_relative_wind_and_radiation():
    f = open_water_fluxes(weather(), 20., 1., .5)
    assert f.sensible < 0 and f.latent < 0 and f.evaporation > 0
    assert f.tau_x > 0 and f.tau_y > 0
    assert f.tau_x / f.tau_y == pytest.approx(5 / 1.5)
    assert f.shortwave == pytest.approx(186.8)
    assert f.longwave == pytest.approx(.98 * (320 - 5.670374419e-8 * 293.15**4))
    still = open_water_fluxes(weather(), 20., 6., 2.)
    assert still.tau_x == 0 and still.tau_y == 0
    humid = open_water_fluxes(weather(specific_humidity=.025), 20., 1., .5)
    assert humid.evaporation < 0 and humid.latent > 0  # condensation is preserved


def test_live_sst_feedback_and_jit():
    air = weather()
    compute = jax.jit(lambda t: open_water_fluxes(air, t, 0., 0.))
    cold, warm = compute(18.), compute(21.)
    assert warm.latent < cold.latent and warm.sensible < cold.sensible
    assert np.isfinite(np.array(warm)).all()
    derivative = jax.grad(lambda t: open_water_fluxes(air, t, 0., 0.).longwave)(20.)
    assert derivative == pytest.approx(-.98 * 4 * 5.670374419e-8 * 293.15**3)


def test_high_wind_control_distinguishes_older_drag_formula():
    # Equal temperature and equal humidity remove stability corrections.
    cd, _, _ = ncar_coefficients10m(293.15, .01, 293.15, .01, 40.)
    assert cd == pytest.approx(.00234)
    old = (2.7 / 40 + .142 + .0764 * 40) / 1000
    assert abs(float(cd) - old) > .0005
