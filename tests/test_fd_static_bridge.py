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
    return dict(
        depth=d,
        h0=h,
        eta=eta,
        values=v,
        target=target,
        Tref=20,
        Sref=35,
        alpha=2e-4,
        beta=7.6e-4,
        rho0=1025.0,
        gravity=9.81,
    )


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


@pytest.mark.parametrize("key", ["eta", "Tref", "Sref", "alpha", "beta", "rho0", "gravity"])
def test_scalar_arrays_rejected(key):
    p = fixture()
    p[key] = np.array([p[key]])
    with pytest.raises(ValueError):
        b.bridge(**p)


def audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit",
        Path(__file__).parents[1] / "research/experiments/fd_static_bridge/discrete_audit.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discrete_fixture():
    return dict(
        depth=np.array([0.0, 5, 300, 500, 1000]),
        reference_weights=np.array([2.5, 147.5, 250, 350, 750]),
        wet_mask=np.array([1, 1, 1, 1, 0]),
        values=np.tile([15, 36, 0.2, -0.1], (5, 1)),
        eta=-2.49,
        Tref=15.0,
        Sref=35.0,
        alpha=2e-4,
        beta=7.6e-4,
        rho0=1025.0,
        gravity=9.81,
        terrain_depth=750.0,
        discrete_bottom=750.0,
        control_interfaces=np.array([]),
        source_sha="2" * 40,
        terrain_sha="a" * 64,
    )


def test_nonstandard_reference_bottom_inventory_not_truncated():
    p = discrete_fixture()
    r = audit_module().audit(**p)
    assert r["reference_column_m"] == 750 and r["wet_node_depth_m"] == 500
    assert r["unsampled_depth_to_discrete_bottom_m"] == 250
    assert r["continuous_bottom_profile_recovered"] is False
    assert r["T_S_reference_u_v_inventory"][2] == 150


def test_beta_matches_source_pressure():
    p = discrete_fixture()
    p.update(
        depth=np.array([0.0, 40]),
        reference_weights=np.array([20.0, 20.0]),
        wet_mask=np.ones(2),
        values=np.tile([15, 36, 0, 0], (2, 1)),
        eta=0.0,
        terrain_depth=40.0,
        discrete_bottom=40.0,
    )
    r = audit_module().audit(**p)
    assert abs(r["original_node_pressure_Pa"][-1] - 305.6796) < 1e-10


def test_cli_single_snapshot_identity(tmp_path, monkeypatch):
    import hashlib
    import io
    import json
    import runpy
    import sys

    p = fixture()
    p["source_sha"] = np.array("2" * 40)
    stream = io.BytesIO()
    np.savez(stream, **p)
    snapshot = stream.getvalue()
    calls = []

    def read(path):
        calls.append(str(path))
        assert len(calls) == 1
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", read)
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["bridge.py", "input.npz", str(output)])
    runpy.run_path(
        str(Path(__file__).parents[1] / "research/experiments/fd_static_bridge/bridge.py"),
        run_name="__main__",
    )
    with open(output) as f:
        r = json.load(f)
    assert r["input_sha256"] == hashlib.sha256(snapshot).hexdigest() and len(calls) == 1


def test_audit_rejects_shallow_discrete_bottom():
    p = discrete_fixture()
    p.update(discrete_bottom=100.0, terrain_depth=50.0)
    before = p["reference_weights"].copy()
    with pytest.raises(ValueError, match="shallower"):
        audit_module().audit(**p)
    np.testing.assert_array_equal(before, p["reference_weights"])


def test_audit_reports_weight_and_terrain_differences():
    p = discrete_fixture()
    p.update(discrete_bottom=700.0, terrain_depth=650.0)
    r = audit_module().audit(**p)
    assert r["reference_weight_minus_discrete_bottom_m"] == 50
    assert r["terrain_minus_discrete_bottom_m"] == -50
    p.update(discrete_bottom=750.0, terrain_depth=750.0)
    r = audit_module().audit(**p)
    assert r["reference_weight_minus_discrete_bottom_m"] == 0
    assert r["wet_node_depth_m"] == 500
