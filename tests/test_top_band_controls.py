"""Isolated FV falsifications, not original FD or real one-degree acceptance."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "research/experiments/top_band_controls/component.py"
SPEC = importlib.util.spec_from_file_location("top_band_component", PATH)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


def same_bytes(a, b):
    return (a.scheme == b.scheme and a.step == b.step and a.h.tobytes() == b.h.tobytes()
            and a.n.tobytes() == b.n.tobytes())


def assert_roundoff(a, b, scale):
    assert np.all(np.abs(np.asarray(a) - b) <= 64 * np.finfo(float).eps * (1 + scale))


def run(scheme, state=None, count=12):
    start = m.initial() if state is None else state
    state, ok, reason = m.remap(start, scheme)
    assert ok, reason
    decisions = []
    for index in range(count):
        q = np.array([0.25, 0.35, 0.]) * (1 if state.step < 6 else -1)
        state, ok, reason = m.advance(state, q, np.zeros((2, 3, 4)))
        decisions.append((ok, reason))
        assert ok
    return state, decisions


@pytest.mark.parametrize("scheme", ["merge", "moving"])
def test_conversion_water_heat_salt_momentum_and_bounds(scheme):
    original = m.initial()
    result, ok, _ = m.remap(original, scheme)
    assert ok
    assert_roundoff(result.h.sum(axis=1), original.h.sum(axis=1), 20)
    assert_roundoff(result.n.sum(axis=1), original.n.sum(axis=1), np.abs(original.n).sum())
    c = m.means(result)[result.h > 0]
    lo, hi = m.means(original).min(axis=(0, 1)), m.means(original).max(axis=(0, 1))
    assert np.all(c >= lo) and np.all(c <= hi)


def test_merge_exact_energy_loss_and_mixing():
    start = m.initial()
    merged, ok, _ = m.remap(start, "merge")
    assert ok
    mass = m.RHO0 * start.h[:, :2]
    shear = m.means(start)[:, 0, 2:] - m.means(start)[:, 1, 2:]
    expected = np.sum(mass[:, 0] * mass[:, 1] / (2 * mass.sum(axis=1)) * np.sum(shear**2, axis=1))
    assert_roundoff(m.energy(start) - m.energy(merged), expected, m.energy(start))
    assert m.variance(merged, 0) < m.variance(start, 0)
    assert m.variance(merged, 1) < m.variance(start, 1)


@pytest.mark.parametrize("scheme", ["merge", "moving"])
def test_scalar_donor_oracle_and_nonzero_extensive_sources(scheme):
    state, _, _ = m.remap(m.initial(), scheme)
    q = np.array([0.05, -0.03, 0.])
    sources = np.arange(24, dtype=float).reshape(2, 3, 4) * 1e-4
    effective = q.copy()
    rates = sources.copy()
    if scheme == "merge":
        effective = np.array([0., q[:2].sum(), 0.])
        rates[:, 1] += rates[:, 0]
        rates[:, 0] = 0
    expected_h, expected_n = state.h.copy(), state.n.copy()
    for cell, amount in enumerate(effective):
        donor = 0 if amount >= 0 else 1
        for field in range(4):
            transported = amount * state.n[donor, cell, field] / (state.h[donor, cell] or 1)
            expected_n[0, cell, field] -= transported
            expected_n[1, cell, field] += transported
        expected_h[0, cell] -= amount
        expected_h[1, cell] += amount
    expected_n += rates
    # Compare intermediate totals independently of target-interface remapping.
    actual, ok, _ = m.advance(state, q, sources)
    assert ok
    if scheme == "merge":
        assert_roundoff(actual.n, expected_n, np.abs(expected_n).sum())
    else:
        # Exact overlap amounts for this named fixture, independent of remap
        # loops: left new top gains .045 of old middle; right loses .045 top.
        mapped = expected_n.copy()
        fraction_left = .045 / expected_h[0, 1]
        mapped[0, 0] = expected_n[0, 0] + fraction_left * expected_n[0, 1]
        mapped[0, 1] = (1 - fraction_left) * expected_n[0, 1]
        fraction_right = .045 / expected_h[1, 0]
        mapped[1, 0] = (1 - fraction_right) * expected_n[1, 0]
        mapped[1, 1] = expected_n[1, 1] + fraction_right * expected_n[1, 0]
        assert_roundoff(actual.n, mapped, np.abs(mapped).sum())
    assert_roundoff(actual.h.sum(axis=1), expected_h.sum(axis=1), 20)
    assert_roundoff(actual.n.sum(axis=1), expected_n.sum(axis=1), np.abs(expected_n).sum())
    assert_roundoff(actual.n.sum(axis=(0, 1)), state.n.sum(axis=(0, 1)) + sources.sum(axis=(0, 1)),
                    np.abs(state.n).sum())


@pytest.mark.parametrize("scheme", ["merge", "moving"])
def test_uniform_tracer_gcl_shared_eta_and_crossing(scheme):
    state, _, _ = m.remap(m.initial(uniform=True), scheme)
    for index in range(6):
        state, ok, _ = m.advance(state, [0.25, 0.35, 0.], np.zeros((2, 3, 4)))
        assert ok
        assert_roundoff(state.eta, [-0.6 * (index + 1), 0.6 * (index + 1)], 20)
        assert_roundoff(m.means(state)[state.h > 0], [15., 35., .25, -.1], 35)
    assert state.eta[0] < -2.5 and m.valid(state)
    before = state.copy()
    fixed, ok, reason = m.remap(state, "fixed")
    assert not ok and reason == "nonpositive_target"
    assert same_bytes(fixed, before) and same_bytes(state, before)


@pytest.mark.parametrize("scheme", ["merge", "moving"])
def test_stable_density_shear_and_cycle_loss_are_reported(scheme):
    start = m.initial()
    converted, _, _ = m.remap(start, scheme)
    c = m.means(converted)
    density = m.RHO0 * (1 - 2e-4 * (c[..., 0] - 20) + 8e-4 * (c[..., 1] - 35))
    for col in range(2):
        assert np.all(np.diff(density[col, converted.h[col] > 0]) >= 0)
    final, _ = run(scheme)
    final_c = m.means(final)
    final_density = m.RHO0 * (1 - 2e-4 * (final_c[..., 0] - 20) + 8e-4 * (final_c[..., 1] - 35))
    for col in range(2):
        assert np.all(np.diff(final_density[col, final.h[col] > 0]) >= 0)
    active = final_c[final.h > 0]
    start_c = m.means(start)
    assert np.all(active >= start_c.min(axis=(0, 1)) - 1e-13)
    assert np.all(active <= start_c.max(axis=(0, 1)) + 1e-13)
    assert_roundoff(final.h.sum(), start.h.sum(), 40)
    assert_roundoff(final.n.sum(axis=(0, 1)), start.n.sum(axis=(0, 1)), np.abs(start.n).sum())
    assert np.max(np.abs(m.means(final) - m.means(converted))) > 0
    assert m.energy(final) < m.energy(converted)
    assert m.variance(final, 0) < m.variance(converted, 0)


@pytest.mark.parametrize("scheme", ["merge", "moving"])
@pytest.mark.parametrize("restart_at", [-1, 0, 5, 7])
def test_restart_before_after_conversion_descent_and_return(scheme, restart_at, tmp_path):
    reference, reference_decisions = run(scheme)
    state = m.initial()
    if restart_at == -1:
        m.save(state, tmp_path / "restart.npz")
        state = m.load(tmp_path / "restart.npz")
    state, ok, _ = m.remap(state, scheme)
    assert ok
    decisions = []
    for index in range(12):
        if restart_at == index:
            m.save(state, tmp_path / "restart.npz")
            state = m.load(tmp_path / "restart.npz")
        q = np.array([.25, .35, 0]) * (1 if state.step < 6 else -1)
        state, ok, reason = m.advance(state, q, np.zeros((2, 3, 4)))
        decisions.append((ok, reason))
    assert decisions == reference_decisions and same_bytes(state, reference)


@pytest.mark.parametrize("scheme", ["merge", "moving"])
def test_one_sided_and_invalid_entry_neighbor_exhaustion_rollback(scheme):
    original = m.initial()
    before = original.copy()
    out, ok, _ = m.remap(original, scheme, conversion=(True, False))
    assert not ok and same_bytes(out, before) and same_bytes(original, before)
    absent = original.copy()
    absent.h[:, 1] = 0
    absent.n[:, 1] = 0
    assert not m.valid(absent)  # This is invalid ENTRY, not loss during advance.
    before = absent.copy()
    out, ok, _ = m.remap(absent, scheme)
    assert not ok and same_bytes(out, before) and same_bytes(absent, before)
    exhausted = original.copy()
    exhausted.h[:, :2] = -1
    assert not m.valid(exhausted)
    before = exhausted.copy()
    out, ok, _ = m.remap(exhausted, scheme)
    assert not ok and same_bytes(out, before) and same_bytes(exhausted, before)


@pytest.mark.parametrize("scheme", ["merge", "moving"])
def test_valid_entry_exhaustion_is_blocked_first_by_cfl_and_snapshot(scheme):
    original = m.initial()
    state, _, _ = m.remap(original, scheme)
    assert m.valid(state)
    before = state.copy()
    out, ok, reason = m.advance(state, [20, 20, 0], np.zeros((2, 3, 4)))
    assert not ok and reason == "outflow_cfl"
    assert same_bytes(out, before) and same_bytes(state, before)
    # A valid thin band likewise cannot disappear under an accepted donor
    # step: the declared outflow CFL rejects BEFORE exhaustion/neighbor loss.
    thin = m.State(m.target_h(np.full(2, -9.6), scheme), np.zeros((2, 3, 4)), scheme)
    thin.n = thin.h[..., None] * [15., 35., .1, 0.]
    assert m.valid(thin)
    before = thin.copy()
    out, ok, reason = m.advance(thin, [.1, .4, 0], np.zeros((2, 3, 4)))
    assert not ok and reason == "outflow_cfl"
    assert same_bytes(out, before) and same_bytes(thin, before)


def test_pressure_same_physical_profile_fails_mixed_representation():
    residual = m.pressure_counterexample()
    assert all(abs(value) > 1e-8 for value in residual.values())
    # Analytic rho anomaly is -rho0*1e-4*z. Its integral is common in both
    # columns: exact horizontal acceleration is zero on any projection grid.
    h = m.target_h(np.full(2, -2.0), "moving")
    equal = m.stationary_profile(h)
    assert m.pressure_at(equal, -4.)[0] == m.pressure_at(equal, -4.)[1]
    expected = m.RHO0 * m.GRAVITY * -2.0 + m.GRAVITY * m.RHO0 * 1e-4 * (16 - 4) / 2
    # P0 integral on a full cell is exact for linear density.
    assert_roundoff(m.pressure_at(equal, -4.)[0], expected, abs(expected))


def test_common_depth_scan_and_linear_matched_control():
    rows = m.pressure_scan()
    at_four = next(row for row in rows if row["common_depth_m"] == -4)
    for scheme, expected in (("fixed", 4.5248625), ("merge", 6.03315), ("moving", 0.)):
        assert_roundoff(at_four["p0_error_Pa"][scheme], expected, abs(at_four["analytic_pressure_Pa"]))
    for row in rows:
        for scheme in ("fixed", "merge", "moving"):
            assert_roundoff(row["linear_error_Pa"][scheme], 0, abs(row["analytic_pressure_Pa"]))
            assert abs(row["linear_mixed_acceleration_m_per_s2"][scheme]) < 1e-14
    # At -4 the moving P0 projection is exact; mixing it with the fixed P0
    # projection creates a difference, not a universal moving-scheme failure.
    assert abs(at_four["p0_error_Pa"]["moving"]) < 1e-10


def test_linear_control_rejects_nonaffine_density():
    state = m.initial()
    with pytest.raises(ValueError, match="non-affine"):
        m.pressure_linear_at(state, -4)


def test_counterflow_aggregation_erases_resolved_tracer_exchange():
    result = m.counterflow()
    assert result["column_q_m3_per_s"] == 0
    assert_roundoff(result["resolved_T_S_u_v_exchange_per_s"][0], .6, 2)
    assert_roundoff(result["merged_T_S_u_v_exchange_per_s"], 0, 100)
    # Fine donor transport conserves global totals but moves 0.6 m^3 K into
    # the right column. Coarse zero volume transport has no such exchange.
    state = m.initial()
    before = state.copy()
    fine_exchange = .1 * m.means(state)[0, 0] - .1 * m.means(state)[1, 1]
    fine_n = state.n.copy()
    fine_n[0, 0] -= .1 * m.means(state)[0, 0]
    fine_n[1, 0] += .1 * m.means(state)[0, 0]
    fine_n[0, 1] += .1 * m.means(state)[1, 1]
    fine_n[1, 1] -= .1 * m.means(state)[1, 1]
    assert_roundoff(fine_n.sum(axis=(0, 1)), before.n.sum(axis=(0, 1)), np.abs(before.n).sum())
    assert_roundoff(fine_n[1].sum(axis=0) - before.n[1].sum(axis=0), fine_exchange, 1000)
    assert same_bytes(state, before)


@pytest.mark.parametrize("corruption", ["fractional_step", "bool_step", "negative_step", "step_array",
                                      "version", "fractional_version", "missing", "scheme_array",
                                      "unknown_scheme", "shape", "nonfinite", "geometry", "moving_geometry",
                                      "inactive_content", "truncated"])
def test_checkpoint_strict_version_state_and_geometry(corruption, tmp_path):
    state = m.initial()
    data = {"h": state.h.copy(), "n": state.n.copy(), "scheme": np.array(state.scheme),
            "step": np.array(5), "schema_version": np.array(m.CHECKPOINT_VERSION)}
    if corruption == "fractional_step":
        data["step"] = np.array(5.9)
    elif corruption == "bool_step":
        data["step"] = np.array(True)
    elif corruption == "negative_step":
        data["step"] = np.array(-1)
    elif corruption == "step_array":
        data["step"] = np.array([5])
    elif corruption == "version":
        data["schema_version"] = np.array(2)
    elif corruption == "fractional_version":
        data["schema_version"] = np.array(1.)
    elif corruption == "missing":
        del data["schema_version"]
    elif corruption == "scheme_array":
        data["scheme"] = np.array(["fixed"])
    elif corruption == "unknown_scheme":
        data["scheme"] = np.array("unknown")
    elif corruption == "shape":
        data["n"] = np.zeros((1, 3, 4))
    elif corruption == "nonfinite":
        data["n"][0, 0, 0] = np.nan
    elif corruption == "geometry":
        data["h"][0, 2] = 9.
    elif corruption == "moving_geometry":
        data["scheme"] = np.array("moving")
        data["h"][0, 0] = 1.
    elif corruption == "inactive_content":
        data["scheme"] = np.array("merge")
        data["h"] = m.target_h(np.zeros(2), "merge")
    path = tmp_path / "damaged.npz"
    if corruption == "truncated":
        path.write_bytes(b"not an npz checkpoint")
    else:
        np.savez(path, **data)
    before = state.copy()
    with pytest.raises(ValueError):
        m.load(path)
    assert same_bytes(state, before)
