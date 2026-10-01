import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

FILE = Path(__file__).parents[1] / "research/experiments/conservative_top_band/component.py"
spec = importlib.util.spec_from_file_location("band", FILE)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


def initial(eta=(0.0, 0.0), uniform=False):
    z = b.target(eta)
    x = 0.5 * (z[:, :-1] + z[:, 1:])
    c = np.zeros((2, 3, 4))
    c[..., 0] = 20 + 0.1 * x
    c[..., 1] = 35
    c[..., 2] = 0.02 * x
    c[..., 3] = -0.01 * x
    if uniform:
        c[:] = [15, 35, 0.2, -0.1]
    return b.State(z, -np.diff(z)[..., None] * c)


def test_fd_affine_integration():
    s = initial()
    nodes = np.array([-20.0, -10.0, -5.0, 0.0])
    c = np.array([20 + 0.1 * nodes, np.full(4, 35), 0.02 * nodes, -0.01 * nodes]).T
    np.testing.assert_allclose(b.fd_to_means(nodes, c, s.z[0]), s.n[0], atol=1e-13)
    with pytest.raises(ValueError):
        b.fd_to_means(nodes, c, b.target([1, 1])[0])


def test_general_point_mapping_not_invertible():
    nodes = np.linspace(-20, 0, 9)
    basis = np.eye(9)
    operator = np.column_stack(
        [
            b.fd_to_means(nodes, np.repeat(basis[:, k, None], 4, axis=1), b.target([0, 0])[0])[:, 0]
            for k in range(9)
        ]
    )
    assert np.linalg.matrix_rank(operator) == 3
    assert operator.shape[1] > np.linalg.matrix_rank(operator)


def test_same_physical_affine_pressure_different_grids():
    s = initial()
    z = s.z.copy()
    z[1, 1] = -4
    z[1, 2] = -12
    t = b.remap(s, z)
    for depth in np.linspace(-20, 0, 41):
        np.testing.assert_allclose(b.pressure(t, depth)[0], b.pressure(t, depth)[1], atol=1e-10)
    np.testing.assert_allclose(t.n.sum(axis=1), s.n.sum(axis=1), atol=1e-12)


def run_cycle(s):
    reports = []
    for direction in [1] * 12 + [-1] * 12:
        eta = s.z[:, 0] + np.array([-0.3, 0.3]) * direction
        lo, hi = -9.0, min(s.z[:, 0])
        s, ok, report = b.advance(s, [(lo, hi, 0.3 * direction)], eta, np.zeros_like(s.n))
        assert ok
        reports.append(report)
    return s, reports


@pytest.mark.parametrize("uniform", [False, True])
def test_crossing_cycle_conservation_and_accounts(uniform):
    s = initial(uniform=uniform)
    out, reports = run_cycle(s)
    np.testing.assert_allclose(out.n.sum(axis=(0, 1)), s.n.sum(axis=(0, 1)), atol=1e-11)
    assert abs(b.ledger(out)["water_m3"] - 40) < 1e-12
    assert len(reports) == 24
    assert all(np.isfinite(list(r.values())).all() for r in reports)
    if uniform:
        np.testing.assert_allclose(
            out.n / (-np.diff(out.z))[..., None], s.n / (-np.diff(s.z))[..., None], atol=1e-12
        )
    else:
        assert np.max(abs(out.n - s.n)) > 0  # cycle not reversible


def test_nonzero_source():
    s = initial()
    source = np.zeros_like(s.n)
    source[0, 1] = [0.2, 0.03, 0.01, -0.02]
    out, ok, _ = b.advance(s, [(-9.0, 0.0, 0.1)], [-0.1, 0.1], source)
    assert ok
    np.testing.assert_allclose(
        out.n.sum(axis=(0, 1)) - s.n.sum(axis=(0, 1)), source.sum(axis=(0, 1)), atol=1e-12
    )


def test_affine_remap_does_not_lose_profile():
    s = initial()
    z = s.z.copy()
    z[:, 1] = -4
    out = b.remap(s, z)
    center = 0.5 * (z[:, :-1] + z[:, 1:])
    np.testing.assert_allclose(out.n[..., 0] / (-np.diff(z)), 20 + 0.1 * center, atol=1e-13)
    # mean KE can change because resolved cell means change, not automatically dissipation
    assert abs(b.ledger(out)["reconstructed_ke_J"] - b.ledger(s)["reconstructed_ke_J"]) < 1e-10


@pytest.mark.parametrize(
    "segments,eta",
    [
        ([(-9, 0, 100)], [-100, 100]),
        ([(-9, 0, 0.1)], [0, 0]),
        ([(-21, 0, 0.1)], [-0.1, 0.1]),
        ([(-9, 0, 0.1), (-8, 0, 0.1)], [-0.2, 0.2]),
        ([], [-10, 10]),
    ],
)
def test_reject_deep_snapshot(segments, eta):
    s = initial()
    snapshot = s.copy()
    out, ok, _ = b.advance(s, segments, eta, np.zeros_like(s.n))
    assert not ok
    for a, c in ((s, snapshot), (out, snapshot)):
        assert (
            a.z.tobytes() == c.z.tobytes() and a.n.tobytes() == c.n.tobytes() and a.step == c.step
        )


def test_restart_before_and_after_crossing(tmp_path):
    s = initial()
    path = tmp_path / "s.npz"
    for k in range(12):
        b.save(s, path)
        recovered = b.load(path)
        eta = s.z[:, 0] + [-0.3, 0.3]
        seg = [(-9, min(s.z[:, 0]), 0.3)]
        a, ok, r = b.advance(s, seg, eta, np.zeros_like(s.n))
        c, ok2, r2 = b.advance(recovered, seg, eta, np.zeros_like(s.n))
        assert ok and ok2 and r == r2
        assert a.n.tobytes() == c.n.tobytes() and a.z.tobytes() == c.z.tobytes()
        s = a
    assert s.z[0, 0] < -2.5 and np.all(-np.diff(s.z) > 0)


@pytest.mark.parametrize("damage", ["step", "version", "geometry", "content"])
def test_corrupt_checkpoint(tmp_path, damage):
    s = initial()
    path = tmp_path / "bad.npz"
    packet = dict(z=s.z, n=s.n, step=np.array(0), version=np.array(b.VERSION))
    if damage == "step":
        packet["step"] = np.array(5.9)
    if damage == "version":
        packet["version"] = np.array(b.VERSION + 1)
    if damage == "geometry":
        packet["z"] = s.z.copy()
        packet["z"][0, 1] = 1
    if damage == "content":
        packet["n"] = s.n.copy()
        packet["n"][0, 0, 0] = np.nan
    np.savez(path, **packet)
    with pytest.raises(ValueError):
        b.load(path)


def test_opposed_face_bands_keep_material_exchange():
    s = initial(uniform=True)
    s.n[0, 0, 0] += 2.5
    s.n[1, 1, 0] -= 7.5
    before = s.n[1].sum(axis=0).copy()
    out, ok, _ = b.advance(s, [(-2.5, 0, 0.1), (-10, -2.5, -0.1)], [0, 0], np.zeros_like(s.n))
    assert ok
    assert abs(out.n[1, :, 0].sum() - before[0]) > 0.05
    np.testing.assert_allclose(out.n.sum(axis=(0, 1)), s.n.sum(axis=(0, 1)), atol=1e-12)


def test_single_column_target_change_remap():
    s = initial()
    z = s.z.copy()
    z[0, 1] = -4
    out = b.remap(s, z)
    assert out.n[1].tobytes() == s.n[1].tobytes()
    np.testing.assert_allclose(out.n.sum(axis=1), s.n.sum(axis=1), atol=1e-12)


def test_curvature_pressure_not_qualified():
    s = initial()
    z = s.z
    s.n[..., 0] = (-np.diff(z)) * (
        20 + 0.01 * (z[:, :-1] ** 2 + z[:, :-1] * z[:, 1:] + z[:, 1:] ** 2) / 3
    )
    depth = -4
    exact = b.RHO * b.G * (-2e-6) * (0 - depth**3) / 3
    assert abs(b.pressure(s, depth)[0] - exact) > 0.1


def salt_state(values):
    s = initial(uniform=True)
    s.n[..., 1] = -np.diff(s.z) * np.asarray(values)
    return s


@pytest.mark.parametrize(
    "values", [[[0, 1, 1], [0, 0, 0]], [[50, 49, 49], [50, 50, 50]], [[0, 50, 0], [50, 0, 50]]]
)
def test_shared_reconstruction_endpoint_limits(values):
    s = salt_state(values)
    c, m, _ = b.reconstruction(s)
    h = -np.diff(s.z)
    for sign in [-1, 1]:
        endpoint = c[..., :2] + sign * 0.5 * h[..., None] * m[..., :2]
        assert np.all(endpoint >= b.TRACER_LOWER) and np.all(endpoint <= b.TRACER_UPPER)
    np.testing.assert_array_equal(c * h[..., None], s.n)
    z = s.z.copy()
    z[:, 1] = -0.1
    z[:, 2] = -7
    out = b.remap(s, z)
    assert b.valid(out)
    np.testing.assert_allclose(out.n.sum(axis=1), s.n.sum(axis=1), atol=1e-12)


def test_negative_salt_transfer_counterexample():
    s = salt_state([[0, 1, 1], [0, 0, 0]])
    out, ok, _ = b.advance(s, [(-0.1, 0, 0.1)], [-0.1, 0.1], np.zeros_like(s.n))
    assert ok and b.valid(out)
    assert np.min(out.n[..., 1] / (-np.diff(out.z))) >= 0
    np.testing.assert_allclose(out.n.sum(axis=(0, 1)), s.n.sum(axis=(0, 1)), atol=1e-12)


@pytest.mark.parametrize("depth", [np.nan, np.inf, -np.inf])
def test_pressure_nonfinite_rejected(depth):
    with pytest.raises(ValueError):
        b.pressure(initial(), depth)


@pytest.mark.parametrize("bad_source,bad_input", [(True, False), (False, True)])
def test_infeasible_tracer_update_rollback(bad_source, bad_input):
    s = salt_state([[0, 0, 0], [0, 0, 0]])
    source = np.zeros_like(s.n)
    if bad_source:
        source[0, 0, 1] = -1
    if bad_input:
        s.n[0, 0, 1] = -1
    snapshot = s.copy()
    out, ok, reason = b.advance(s, [], [0, 0], source)
    assert not ok and reason["rejection_reason"]
    assert s.n.tobytes() == snapshot.n.tobytes() == out.n.tobytes()
    assert s.z.tobytes() == snapshot.z.tobytes() == out.z.tobytes()


def test_bounds_with_nonuniform_misaligned_face_and_restart(tmp_path):
    s = salt_state([[0, 1, 2], [2, 1, 0]])
    path = tmp_path / "bounds.npz"
    total = s.n.sum(axis=(0, 1)).copy()
    for direction in [1] * 12 + [-1] * 12:
        b.save(s, path)
        copy = b.load(path)
        eta = s.z[:, 0] + np.array([-0.3, 0.3]) * direction
        segment = [(-9, min(s.z[:, 0]), 0.3 * direction)]
        out, ok, report = b.advance(s, segment, eta, np.zeros_like(s.n))
        replay, ok2, report2 = b.advance(copy, segment, eta, np.zeros_like(s.n))
        assert ok and ok2 and b.valid(out) and report == report2
        assert out.n.tobytes() == replay.n.tobytes() and out.z.tobytes() == replay.z.tobytes()
        s = out
    np.testing.assert_allclose(s.n.sum(axis=(0, 1)), total, atol=1e-12)
