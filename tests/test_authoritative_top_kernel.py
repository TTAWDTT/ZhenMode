import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "inventory_kernel", ROOT / "research/experiments/authoritative_top_kernel/kernel.py"
)
k = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = k
spec.loader.exec_module(k)


def analytic(uniform=False):
    h = np.array([[2.5, 7.5, 12.5], [4.0, 6.0, 12.5]])
    eta = k.B + h.sum(axis=1)
    z = np.c_[eta, eta[:, None] - np.cumsum(h, axis=1)]
    x = 0.5 * (z[:, :-1] + z[:, 1:])
    c = np.zeros((2, 3, 4))
    c[:, :, 0] = 15 if uniform else 20 + 0.1 * x
    c[:, :, 1] = 35
    if uniform:
        c[:, :, 2] = 1025 * 0.2
        c[:, :, 3] = -1025 * 0.1
    n = h[:, :, None] * c
    eos = dict(rho0=1025.0, gravity=9.81, alpha=2e-4, beta=7.6e-4, Tref=15.0, Sref=35.0)
    deep = tuple(
        dict(
            nodes=np.array([30.0, 100.0, 400.0]),
            wet_mask=np.ones(3),
            pressure_increment=np.zeros(3),
            inventory=np.zeros((3, 4)),
        )
        for _ in range(2)
    )
    return k.State(
        h, n, eos, deep, dict(source_sha=["2" * 40] * 2, geometry_report_sha256="a" * 64)
    )


@pytest.mark.parametrize("uniform", [True, False])
def test_static_common_physical_profile(uniform):
    s = analytic(uniform)
    for depth in np.linspace(-22.5, 0, 101):
        assert abs(k.pressure(s, 0, depth) - k.pressure(s, 1, depth)) < 1e-10
    assert abs(k.pressure_power(s, [(-20.0, -1.0, 0.01)])) < 1e-10


def test_closed_water_inventory_gcl_and_uniform_tracer():
    s = analytic(True)
    before = s.copy()
    out, ok, r = k.advance(s, [(-20, -1, 0.01)], np.zeros_like(s.inventory))
    assert ok and out.eta[0] == pytest.approx(-0.01) and out.eta[1] == pytest.approx(0.01)
    np.testing.assert_allclose(
        out.inventory[:, :, :2] / out.h[:, :, None],
        np.broadcast_to([15, 35], (2, 3, 2)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        out.inventory.sum(axis=(0, 1)), s.inventory.sum(axis=(0, 1)), atol=1e-12
    )
    for j in range(2):
        assert out.deep[j]["inventory"].tobytes() == before.deep[j]["inventory"].tobytes()
    assert (
        s.inventory.tobytes() == before.inventory.tobytes() and r["qualification_passed"] is False
    )


@pytest.mark.parametrize("kind", ["pressure", "deep", "cfl", "badsource", "overflow"])
def test_fail_closed_rollback(kind):
    s = analytic(True)
    snap = s.copy()
    sources = np.zeros_like(s.inventory)
    seg = []
    kwargs = {}
    if kind == "pressure":
        kwargs["pressure_driven"] = True
    if kind == "deep":
        kwargs["deep_transport"] = True
    if kind == "cfl":
        seg = [(-20, -1, 100)]
    if kind == "badsource":
        sources[0, 0, 1] = -1000
    if kind == "overflow":
        sources[0, 0, 2] = 1e308
    out, ok, r = k.advance(s, seg, sources, **kwargs)
    assert not ok and r["rejection_reason"]
    assert out.inventory.tobytes() == snap.inventory.tobytes() == s.inventory.tobytes()
    assert out.h.tobytes() == snap.h.tobytes() == s.h.tobytes()


def test_nonzero_extensive_source():
    s = analytic(True)
    source = np.zeros_like(s.inventory)
    source[0, 1] = [0.1, 0.02, 0.03, -0.01]
    out, ok, _ = k.advance(s, [], source)
    assert ok
    np.testing.assert_allclose(
        out.inventory.sum(axis=(0, 1)) - s.inventory.sum(axis=(0, 1)),
        source.sum(axis=(0, 1)),
        atol=1e-12,
    )


def test_migration_before_actual_velocity_recovery():
    fixture = __import__("runpy").run_path(str(ROOT / "tests/test_local_top_bridge.py"))["fixture"]
    p = fixture(-2.4914792546513693)
    q = fixture(-0.1)
    s = k.migrate([p, q], "a" * 64)
    assert np.isfinite(k.velocity(s)).all() and s.migration["candidate_K_J_per_m2"] >= 0
    for j, col in enumerate([p, q]):
        oldh = col["reference_weights"][:3].copy()
        oldh[0] += col["eta"]
        old = oldh[:, None] * col["values"][:3]
        old[:, 2:] = 1025 * col["reference_weights"][:3, None] * col["values"][:3, 2:]
        np.testing.assert_allclose(s.inventory[j].sum(axis=0), old.sum(axis=0), atol=1e-12)
        assert s.deep[j]["values"].tobytes() == col["values"][3:].tobytes()


def test_pressure_join_anchors_band_bottom():
    fixture = __import__("runpy").run_path(str(ROOT / "tests/test_local_top_bridge.py"))["fixture"]
    s = k.migrate([fixture(-2.49), fixture(-0.1)], "a" * 64)
    base = k.pressure(s, 0, -22.5)
    for depth in [-22.500001, -23.0, -29.0]:
        expected = base + (abs(depth) - 22.5) / (30 - 22.5) * s.deep[0]["pressure_increment"][0]
        assert k.pressure(s, 0, depth) == pytest.approx(expected, abs=1e-10)


def test_thin_old_momentum_migrates_without_velocity_extrapolation():
    fixture = __import__("runpy").run_path(str(ROOT / "tests/test_local_top_bridge.py"))["fixture"]
    p = fixture(-2.5 + 1e-10)
    p["values"][0, 2] = 0.1
    s = k.migrate([p, fixture(-0.1)], "a" * 64)
    assert abs(k.velocity(s)).max() < 1
    assert s.migration["momentum_migration"].startswith("P0 overlap fractions")
