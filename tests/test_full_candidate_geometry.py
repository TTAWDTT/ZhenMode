"""Actual production-grid migration/metric tests; not complete-step qualification."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grid import GlobalOceanGrid  # noqa: E402
from jax_solver_global import JaxStateG, make_fd_params  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "full_candidate_geometry", ROOT / "research/experiments/full_candidate/step.py"
)
k = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = k
spec.loader.exec_module(k)


def production_grid():
    z = -np.array([0., 5., 15., 30., 100., 500.])
    lat = np.array([-30., 0., 30.])
    cosine = np.cos(np.deg2rad(lat))
    dx = np.broadcast_to(6371000 * cosine[None, :] * np.pi / 2, (4, 3)).copy()
    depth = np.full((4, 3), 750.)
    depth[1, 1], depth[2, 2] = 5., 0.
    wet2 = (depth > 0).astype(float)
    wet = (-z[None, None, :] <= depth[..., None]) * wet2[..., None]
    return GlobalOceanGrid(
        np.arange(4) * 90., lat, dx, 6371000 * np.pi / 6, cosine,
        np.zeros((4, 3)), z, -np.diff(z), 6, depth, wet2,
        wet2.astype(bool), ~wet2.astype(bool), wet, 4, 3,
    )


def entry(eta=-2.49):
    g = production_grid()
    fd = make_fd_params(g, column_geometry="nodal_dual_v1")
    p = SimpleNamespace(**fd._asdict(), column_geometry="nodal_dual_v1", T_ref=15., S_ref=35.)
    shape = (g.nx, g.ny, g.nz)
    s = JaxStateG(np.full(shape, .2), np.full(shape, -.1),
                  np.full(shape, 15.), np.full(shape, 35.),
                  np.full(shape[:2], eta), np.zeros(shape[:2]))
    out = k.migrate(s, p, source_sha="a" * 40, forcing_sha="b" * 64)
    return g, p, s, out


def test_production_shallow_bottom_no_invented_water_and_deep_unchanged():
    g, p, s, out = entry()
    ref = np.broadcast_to(np.asarray(p.dz_node), s.T.shape) * g.wet_mask_3d
    assert out.band_bottom[1, 1] == -10.
    assert out.h[1, 1].sum() == pytest.approx(7.51)
    np.testing.assert_array_equal(out.h[..., 3:], ref[..., 3:])
    np.testing.assert_array_equal(out.n[..., 3:, 0], ref[..., 3:] * s.T[..., 3:])
    np.testing.assert_array_equal(out.n[..., 3:, 2], 1025 * ref[..., 3:] * s.u[..., 3:])
    assert out.h[2, 2].sum() == 0.
    assert np.all(out.n[2, 2] == 0.)


@pytest.mark.parametrize("eta", [-2.4914792546513693, 0., .2])
def test_migration_per_column_water_tracer_reference_momentum(eta):
    g, p, s, out = entry(eta)
    ref = np.broadcast_to(np.asarray(p.dz_node), s.T.shape) * g.wet_mask_3d
    actual = ref.copy()
    actual[..., 0] += eta * g.wet_mask
    for i, values in enumerate((s.T, s.S, s.u, s.v)):
        old = (actual if i < 2 else ref * 1025) * values
        np.testing.assert_allclose(out.n[..., i].sum(axis=-1), old.sum(axis=-1),
                                   rtol=3e-15, atol=1e-12)
    np.testing.assert_allclose(out.h.sum(axis=-1), actual.sum(axis=-1), rtol=3e-15)


def test_spherical_face_volume_matches_original_metrics_and_walls():
    _, p, _, s = entry(0.)
    # Recover prescribed actual velocities; reference-momentum migration
    # itself need not preserve velocity on changed mass.
    s.n[..., 2] = 1025 * s.h * .2
    s.n[..., 3] = 1025 * s.h * -.1
    records, force = k.top_faces(s, p)
    for a, b, _, _, axis, q, lo, hi in records:
        assert s.h[a].sum() and s.h[b].sum()
        assert not (axis == 1 and a[1] == 2)
        width = p.dy if axis == 0 else (
            np.asarray(p.dx_2d)[a] / np.asarray(p.cos_lat)[a[1]]
            * .5 * (np.asarray(p.cos_lat)[a[1]] + np.asarray(p.cos_lat)[b[1]])
        )
        assert q == pytest.approx(width * (hi - lo) * (.2 if axis == 0 else -.1))
    assert np.max(abs(force)) < 1e-10


def test_spherical_area_negative_adjoint_pressure_not_unweighted_sum():
    _, p, _, s = entry(0.)
    s.eta[0, 1] = .1
    s.h[0, 1, :3] = (22.5 + .1) * k.FRACTIONS
    s.n[..., 0], s.n[..., 1] = 15 * s.h, 35 * s.h
    s.n[..., 2], s.n[..., 3] = 1025 * .2 * s.h, -1025 * .1 * s.h
    records, force = k.top_faces(s, p)
    expected = 0.
    for a, b, _, _, _, q, lo, hi in records:
        depth = .5 * (lo + hi)
        expected -= q * (k.pressure(s, b, depth) - k.pressure(s, a, depth))
    area = np.asarray(p.dx_2d) * p.dy
    u = np.zeros_like(force)
    np.divide(s.n[..., 2:], 1025 * s.h[..., None], out=u, where=s.h[..., None] > 0)
    actual = np.sum(area[..., None, None] * u * force)
    assert actual == pytest.approx(expected, abs=1e-5, rel=3e-14)


def test_deep_pressure_uniform_density_and_shallow_absence():
    g, p, _, s = entry()
    pp = k.deep_pressure(s, p, g, (0, 1))
    np.testing.assert_allclose(pp[3:], 1025 * 9.81 * s.eta[0, 1], atol=1e-10)
    np.testing.assert_array_equal(k.deep_pressure(s, p, g, (1, 1)), np.zeros(6))


def test_negative_attempt_cannot_be_migrated_as_accepted_entry():
    with pytest.raises(ValueError, match="nonpositive accepted entry"):
        entry(-2.5016904655664254)


def configured(*, source=False, mixing=False, wave=False):
    g, p, fd, _ = entry(0.)
    p.dt, p.mode_split, p.n_subcyc = 30., True, 3
    p.process_time_scheme = k.TIME_SCHEME
    p.kappa_h = p.kappa_v = p.nu_h = p.nu_v = p.kappa_conv = p.r_bot = 0.
    p.f = g.f.copy()
    p.lambda_bulk = 0.
    p.T_atm_3d = np.full((4, 3, 1), 15.)
    p.Q_heat_2d = np.full((4, 3), 10. if source else 0.)
    p.tau_x_2d = np.full((4, 3), .001 if source else 0.)
    p.tau_y_2d = np.zeros((4, 3))
    u, v = np.zeros(fd.T.shape), np.zeros(fd.T.shape)
    temperature = fd.T.copy()
    if mixing:
        p.kappa_h, p.kappa_v, p.nu_h, p.nu_v, p.kappa_conv = 1., 1e-5, 1., 1e-4, .01
        p.r_bot = .001
        temperature += .01 * np.asarray(g.z)[None, None, :]
        u[:, 1, 3:] = .01
    if wave:
        fd = fd._replace(eta=.01 * np.cos(np.arange(4)[:, None] * np.pi / 2)
                         * np.ones((4, 3)))
    fd = fd._replace(u=u, v=v, T=temperature)
    s = k.migrate(fd, p, source_sha="a" * 40, forcing_sha="b" * 64)
    return g, p, s


@pytest.mark.parametrize("source,mixing,wave", [(False, False, False), (True, False, False),
                                               (True, True, True), (False, False, True)])
def test_first_global_call_static_source_rotation_mixing_wave(source, mixing, wave):
    g, p, s = configured(source=source, mixing=mixing, wave=wave)
    p.f = np.broadcast_to(np.array([-.0001, 0., .0001])[None, :], (4, 3)).copy()
    s.identity["parameter_sha"] = k.parameter_digest(p)
    snap = s.copy()
    out, ok, report = k.advance(s, p, g, forcing_sha="b" * 64)
    assert ok, report
    assert out.step == 1 and not report["qualification_passed"]
    assert np.max(abs(np.asarray(report["inventory_residual"]))
                  - np.asarray(report["inventory_bound"])) <= 0
    assert s.n.tobytes() == snap.n.tobytes() and s.h.tobytes() == snap.h.tobytes()
    if source:
        area = np.asarray(p.dx_2d) * p.dy
        expected = (area * g.wet_mask * p.Q_heat_2d).sum() * p.dt / (1025 * k.C_P)
        assert report["source_inventory"][0] == pytest.approx(expected, rel=1e-14)
    if not source and not mixing and not wave:
        # Rest is preserved to floating arithmetic, not arbitrary bitwise
        # reversibility of a P1 integration/remap.
        np.testing.assert_allclose(out.n, s.n, rtol=16 * np.finfo(float).eps, atol=1e-17)
    if wave:
        assert not np.array_equal(out.eta, s.eta)


@pytest.mark.parametrize("field,value", [("nu_bi", 1e12), ("kappa_gm", 100.),
                                         ("dynamic_ice", True), ("restore_coef_S", .001),
                                         ("polar_cap_rows", 1)])
def test_original_enabled_unimplemented_process_blocks_and_rolls_back(field, value):
    g, p, s = configured(source=True)
    setattr(p, field, value)
    snap = s.copy()
    out, ok, report = k.advance(s, p, g, forcing_sha="b" * 64)
    assert not ok and field in report["rejection_reason"]
    for name in ("h", "n", "eta", "ice", "band_bottom"):
        assert getattr(out, name).tobytes() == getattr(snap, name).tobytes()
        assert getattr(s, name).tobytes() == getattr(snap, name).tobytes()


def test_completed_step_restart_next_decision_and_bytes(tmp_path):
    g, p, s = configured(source=True, mixing=True)
    out, ok, report = k.advance(s, p, g, forcing_sha="b" * 64)
    assert ok, report
    path = tmp_path / "candidate.npz"
    k.save(out, path)
    loaded = k.load(path, expected_identity=out.identity, p=p, grid=g)
    a, oka, ra = k.advance(out, p, g, forcing_sha="b" * 64)
    b, okb, rb = k.advance(loaded, p, g, forcing_sha="b" * 64)
    assert oka == okb and ra == rb
    for name in ("h", "n", "eta", "ice", "band_bottom"):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes()


def test_mutated_parameter_forcing_snapshot_is_rejected():
    g, p, s = configured(source=True)
    p.Q_heat_2d[0, 0] += 1.
    out, ok, report = k.advance(s, p, g, forcing_sha="b" * 64)
    assert not ok and "snapshot changed" in report["rejection_reason"]
    assert out.n.tobytes() == s.n.tobytes()


def test_static_affine_density_common_depth_across_shallow_and_deep_columns():
    g, p, s = configured()
    for cell in np.ndindex(s.eta.shape):
        if not np.any(s.h[cell][:3]):
            continue
        z = k.top_edges(s, cell)
        centers = np.r_[.5 * (z[:-1] + z[1:]), g.z[3:]]
        s.n[cell][:, 0] = s.h[cell] * (20 + .01 * centers)
    records, force = k.all_faces(s, p, g)
    assert records
    assert np.max(abs(force)) < 1e-12
    out, ok, report = k.advance(s, p, g, forcing_sha="b" * 64)
    assert ok, report
    velocity = k.means(out)[..., 2:] / 1025
    assert np.max(abs(velocity)) < 1e-12


@pytest.mark.parametrize("damage", ["float_step", "float32_h", "deep_mass", "band_bottom", "version"])
def test_damaged_restart_is_rejected(tmp_path, damage):
    g, p, s = configured()
    path = tmp_path / "bad.npz"
    k.save(s, path)
    with np.load(path, allow_pickle=False) as packet:
        data = {name: packet[name].copy() for name in packet.files}
    if damage == "float_step":
        data["step"] = np.array(5.9)
    elif damage == "float32_h":
        data["h"] = data["h"].astype(np.float32)
    elif damage == "deep_mass":
        data["h"][0, 0, 3] += 1
    elif damage == "band_bottom":
        data["band_bottom"][0, 0] -= 1
    else:
        data["version"] = np.array(2)
    np.savez(path, **data)
    with pytest.raises(ValueError):
        k.load(path, expected_identity=s.identity, p=p, grid=g)


def test_actual_deep_response_cross_band_uniform_tracer_gcl():
    g, p, s = configured()
    s.n[0, 1, 3:, 2] = 1025 * .03 * s.h[0, 1, 3:]
    out, ok, report = k.advance(s, p, g, forcing_sha="b" * 64)
    assert ok, report
    assert np.max(abs(np.asarray(report["cross_band_volume_m"]))) > 0
    assert not np.array_equal(out.n[..., 3:, 2:], s.n[..., 3:, 2:])
    c = k.means(out)
    np.testing.assert_allclose(c[..., :2][out.h > 0],
                               np.broadcast_to([15., 35.], c[..., :2][out.h > 0].shape),
                               rtol=2e-15, atol=2e-12)


@pytest.mark.parametrize("kind", ["outflow", "mixing", "identity"])
def test_failed_global_call_snapshot(kind):
    g, p, s = configured(source=True)
    if kind == "outflow":
        p.dt = 1e12
        s.n[0, 1, 3:, 2] = 1025 * .03 * s.h[0, 1, 3:]
    elif kind == "mixing":
        p.nu_h = 1e20
    s.identity["parameter_sha"] = k.parameter_digest(p)
    snap = s.copy()
    out, ok, report = k.advance(s, p, g, forcing_sha=("c" * 64 if kind == "identity" else "b" * 64))
    assert not ok and report["rejection_reason"]
    if kind != "identity":
        assert "CFL" in report["rejection_reason"]
    for name in ("h", "n", "eta", "ice", "band_bottom"):
        assert getattr(out, name).tobytes() == getattr(snap, name).tobytes()
        assert getattr(s, name).tobytes() == getattr(snap, name).tobytes()


def test_original_stage_clock_not_silently_replaced():
    g, p, s = configured(source=True)
    p.process_time_scheme = "symmetric_fast_v3"
    out, ok, report = k.advance(s, p, g, forcing_sha="b" * 64)
    assert not ok and "original stage schedule" in report["rejection_reason"]
    assert out.n.tobytes() == s.n.tobytes()
