import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    out = importlib.util.module_from_spec(spec)
    sys.modules[name] = out
    spec.loader.exec_module(out)
    return out


c = module("curvature", "research/experiments/curvature_pressure/diagnostic.py")
b = module("band_curve", "research/experiments/conservative_top_band/component.py")


@pytest.mark.parametrize("coef", list(c.PROFILES.values()))
def test_three_layer_shared_algorithm_equivalence(coef):
    z = b.target([0, 0])
    v = np.zeros((2, 3, 4))
    v[..., 0] = c.means(z[0], coef)
    v[..., 1] = 35
    s = b.State(z, -np.diff(z)[..., None] * v)
    mean, slope, _ = b.reconstruction(s)
    np.testing.assert_array_equal(c.reconstruct(z[0], mean[0, :, 0]), slope[0, :, 0])
    depth = np.linspace(-20, 0, 81)
    pressure, _ = c.pressure_bound(z[0], coef, depth)
    np.testing.assert_allclose(pressure, [b.pressure(s, x)[0] for x in depth], atol=1e-11)


def test_frozen_scan():
    result = c.run()
    assert len(result["rows"]) == 10 and result["qualification_passed"] is False


@pytest.mark.parametrize("coef", list(c.PROFILES.values()))
def test_stability_and_full_cell_integral(coef):
    depth = np.linspace(-20, 0, 401)
    derivative = np.polynomial.polynomial.polyder(coef)
    assert np.min(np.polynomial.polynomial.polyval(depth, derivative)) > 0
    z = c.grid(32, True)
    p, _ = c.pressure_bound(z, coef, z)
    anomaly = np.array(coef)
    anomaly[0] -= 20
    primitive = np.polynomial.polynomial.polyint(anomaly)
    exact = c.RHO * c.G * c.ALPHA * np.polynomial.polynomial.polyval(z, primitive)
    np.testing.assert_allclose(p, exact, atol=1e-11)
