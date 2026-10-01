import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "hybrid", ROOT / "research/experiments/hybrid_dynamic_step/step.py"
)
k = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = k
spec.loader.exec_module(k)


def initial(eta=(0.0, 0.0), different=False, uniform=False):
    h = np.tile([2.5, 7.5, 12.5, 17.5, 10.0], (2, 1))
    h[:, 0] += eta
    if different:
        h[1, :3] = [4.0, 6.0, 12.5]
    z = np.c_[np.asarray(eta), np.asarray(eta)[:, None] - np.cumsum(h[:, :3], axis=1)]
    centers = 0.5 * (z[:, :-1] + z[:, 1:])
    v = np.zeros((2, 5, 4))
    v[:, :3, 0] = 20 + 0.1 * centers
    v[:, 3:, 0] = 20 - 0.1 * np.array([30.0, 50.0])
    v[:, :, 1] = 35
    if uniform:
        v[:, :, 0] = 15
    return k.State(h, h[:, :, None] * v, np.array([30.0, 50.0]), 1000.0, "2" * 40)


def test_static_same_profile_different_partition():
    s = initial(different=True)
    out, ok, r = k.advance(s, 1.0, np.zeros_like(s.n))
    assert ok
    assert abs(k.velocity(out)).max() < 1e-12
    assert abs(r["pressure_work_J_per_m2"]) < 1e-15


def test_velocity_generated_flux_deep_response_cross_band_gcl():
    s = initial((0.01, -0.01), uniform=True)
    out, ok, r = k.advance(s, 1.0, np.zeros_like(s.n))
    assert ok
    assert abs(k.velocity(out)[:, 3:, 0]).max() > 0
    assert max(abs(np.array(r["band_bottom_volume_flux_m_s"]))) > 0
    assert abs(r["water_residual_m"]) < 1e-12
    np.testing.assert_allclose(
        out.n[:, :, :2] / out.h[:, :, None], np.broadcast_to([15, 35], (2, 5, 2)), atol=1e-12
    )
    assert np.all(abs(np.array(r["inventory_residual"])) <= r["inventory_bound"])
    assert r["qualification_passed"] is False


def test_pressure_impulse_and_ke_exact_kick():
    s = initial((0.01, -0.01), uniform=True)
    out, work = k.kick(s, 1.0)
    assert k.ke(out) - k.ke(s) == pytest.approx(work, abs=1e-12)
    assert abs(k.velocity(out)).max() > 0


def test_restart_step_bitwise(tmp_path):
    s = initial((0.01, -0.01), uniform=True)
    path = tmp_path / "s.npz"
    k.save(s, path)
    loaded = k.load(path)
    a, ok, r = k.advance(s, 1.0, np.zeros_like(s.n))
    b, ok2, r2 = k.advance(loaded, 1.0, np.zeros_like(s.n))
    assert ok and ok2 and r == r2
    assert a.h.tobytes() == b.h.tobytes() and a.n.tobytes() == b.n.tobytes()


@pytest.mark.parametrize("dt", [1e7, -1, np.nan])
def test_reject_original_snapshot(dt):
    s = initial((0.01, -0.01))
    snap = s.copy()
    out, ok, r = k.advance(s, dt, np.zeros_like(s.n))
    assert not ok and r["rejection_reason"]
    assert out.n.tobytes() == s.n.tobytes() == snap.n.tobytes()


def test_dt300_flat_750m_near_original_top_crossing():
    h = np.tile([2.5, 7.5, 12.5, 42.5, 135.0, 200.0, 350.0], (2, 1))
    h[:, 0] += np.array([-2.4914792546513693, -2.4])
    v = np.zeros((2, 7, 4))
    v[:, :, 0] = 15
    v[:, :, 1] = 35
    s = k.State(h, h[:, :, None] * v, np.array([30.0, 100.0, 300.0, 500.0]), 100000.0, "2" * 40)
    out, ok, r = k.advance(s, 300.0, np.zeros_like(s.n))
    assert ok
    assert abs(k.velocity(out)[:, 3:, 0]).max() > 0
    assert max(abs(np.array(r["band_bottom_volume_flux_m_s"]))) > 0
    assert r["qualification_passed"] is False


def test_generated_flux_pressure_force_negative_adjoint():
    s = initial((0.01, -0.01))
    s.n[:, :, 2] = (
        1025 * s.h * np.array([[0.1, 0.2, -0.1, 0.05, 0.03], [-0.05, 0.1, 0.2, -0.02, 0.07]])
    )
    records, force = k.faces(s)
    p = k.pressures(s)
    expected = 0.0
    roots, weights = np.polynomial.legendre.leggauss(2)
    for a, b, q, lo, hi in records:
        if lo is None:
            delta = p[1, b] - p[0, a]
        else:
            points = 0.5 * (lo + hi) + 0.5 * (hi - lo) * roots
            delta = 0.5 * np.sum(
                weights
                * np.array([k.top_pressure(s, 1, d) - k.top_pressure(s, 0, d) for d in points])
            )
        expected -= q * delta
    actual = np.sum(k.velocity(s)[:, :, 0] * force)
    assert actual == pytest.approx(expected, rel=1e-13, abs=1e-12)


@pytest.mark.parametrize(
    "damage",
    ["string_dx", "array_dx", "float32_deep", "shape_deep", "nan_h", "source_dtype", "float_step"],
)
def test_strict_checkpoint_rejected(tmp_path, damage):
    s = initial()
    path = tmp_path / "bad.npz"
    packet = dict(
        h=s.h,
        n=s.n,
        deep_nodes=s.deep_nodes,
        dx=np.array(s.dx),
        source_sha=np.array(s.source_sha),
        step=np.array(0),
        version=np.array(1),
    )
    if damage == "string_dx":
        packet["dx"] = np.array("1000")
    if damage == "array_dx":
        packet["dx"] = np.array([1000.0])
    if damage == "float32_deep":
        packet["deep_nodes"] = s.deep_nodes.astype(np.float32)
    if damage == "shape_deep":
        packet["deep_nodes"] = s.deep_nodes[None, :]
    if damage == "nan_h":
        packet["h"] = s.h.copy()
        packet["h"][0, 0] = np.nan
    if damage == "source_dtype":
        packet["source_sha"] = np.array(123)
    if damage == "float_step":
        packet["step"] = np.array(1.5)
    np.savez(path, **packet)
    with pytest.raises(ValueError):
        k.load(path)


@pytest.mark.parametrize("damage", ["string_dx", "float32_deep", "shape_deep", "nonfinite"])
def test_invalid_state_failure_is_deep_snapshot(damage):
    s = initial()
    if damage == "string_dx":
        s.dx = "1000"
    if damage == "float32_deep":
        s.deep_nodes = s.deep_nodes.astype(np.float32)
    if damage == "shape_deep":
        s.deep_nodes = s.deep_nodes[None, :]
    if damage == "nonfinite":
        s.n[0, 0, 0] = np.nan
    snap = s.copy()
    out, ok, r = k.advance(s, 1.0, np.zeros_like(s.n))
    assert not ok and r["rejection_reason"]
    assert out.h.tobytes() == s.h.tobytes() == snap.h.tobytes()
    assert out.n.tobytes() == s.n.tobytes() == snap.n.tobytes()
    assert out.deep_nodes.tobytes() == s.deep_nodes.tobytes() == snap.deep_nodes.tobytes()
