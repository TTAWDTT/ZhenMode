import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "bridge", Path(__file__).parents[1] / "research/experiments/fd_static_bridge/bridge.py"
)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def fixture(n=9, eta=0, nonuniform=False):
    d = np.linspace(0, 40, n)
    if nonuniform:
        d = 40 * (d / 40) ** 1.2
    h = -np.diff(np.r_[0, -0.5 * (d[:-1] + d[1:]), -40])
    v = np.array([20 - 0.1 * d + 0.001 * d * d, np.full(n, 35), 0.01 * d, -0.02 * d]).T
    z = np.r_[eta, -0.5 * (d[:-1] + d[1:]), -40]
    target = z.copy()
    target[1] = eta - 0.25 * (eta - z[2])
    return dict(depth=d, h0=h, eta=eta, values=v, target=target, Tref=20, Sref=35)


@pytest.mark.parametrize("nonuniform", [False, True])
def test_inventory_and_explicit_geometry_difference(nonuniform):
    p = fixture(nonuniform=nonuniform)
    n, r = b.bridge(**p)
    np.testing.assert_allclose(n.sum(axis=0), r["original_material_T_S_reference_u_v"], atol=1e-12)
    assert (
        abs(r["original_material_T_S_reference_u_v"][0] - r["interpolation_geometry_inventory"][0])
        < 1e-11
    )  # eta0 trapezoid mass matches
    p["eta"] = -p["h0"][0] + 0.01
    p["target"][0] = p["eta"]
    p["target"][1] = -3.75
    n, r = b.bridge(**p)
    np.testing.assert_allclose(n.sum(axis=0), r["original_material_T_S_reference_u_v"], atol=1e-12)
    assert (
        abs(r["original_material_T_S_reference_u_v"][0] - r["interpolation_geometry_inventory"][0])
        > 0.001
    )


def test_bounds_active():
    p = fixture()
    p["values"][:, 0] = [45, 0, 45, 0, 45, 0, 45, 0, 45]
    p["values"][:, 1] = [0, 50, 0, 50, 0, 50, 0, 50, 0]
    out, r = b.bridge(**p)
    mean = out / (-np.diff(p["target"]))[:, None]
    assert np.all(mean[:, :2] >= b.LOW) and np.all(mean[:, :2] <= b.HIGH)
    np.testing.assert_allclose(
        out.sum(axis=0), r["original_material_T_S_reference_u_v"], atol=1e-12
    )


@pytest.mark.parametrize("failure", ["negative", "extrapolate", "mass", "nan", "bounds"])
def test_reject_preserves_inputs(failure):
    p = fixture()
    if failure == "negative":
        p["eta"] = -2.5
    if failure == "extrapolate":
        p["eta"] = 0.1
    if failure == "mass":
        p["h0"][0] += 1
    if failure == "nan":
        p["values"][0, 0] = np.nan
    if failure == "bounds":
        p["values"][0, 1] = 100
    snap = {k: v.copy() for k, v in p.items() if isinstance(v, np.ndarray)}
    with pytest.raises(ValueError):
        b.bridge(**p)
    for k, v in snap.items():
        assert p[k].tobytes() == v.tobytes()


@pytest.mark.parametrize("profile", ["quadratic", "cubic"])
def test_pressure_refinement(profile):
    errors = []
    for n in [9, 17, 33, 65]:
        p = fixture(n)
        if profile == "cubic":
            d = p["depth"]
            p["values"][:, 0] = 20 - 0.1 * d - 0.00001 * d**3
        _, r = b.bridge(**p)
        errors.append(r["max_original_pressure_difference_Pa"])
    assert all(a / bb > 3 for a, bb in zip(errors, errors[1:]))
