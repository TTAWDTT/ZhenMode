"""
Tests for the IMEX time integrator.

Run: python tests/test_integrator.py
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataclasses import replace
from config import DEFAULT_CONFIG
from grid import make_grid
from state import initialize_state
from integrator import step, integrate


# ── Test helpers ────────────────────────────────────────────────────

def get_grid():
    """Load grid once (cached via module-level lazy init)."""
    if not hasattr(get_grid, '_cache'):
        get_grid._cache = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    return get_grid._cache


def assert_close(a, b, tol=1e-6, msg=""):
    assert abs(a - b) < tol, f"{msg}: {a} != {b} (tol={tol})"


def assert_allclose(a, b, tol=1e-6, msg=""):
    diff = np.max(np.abs(np.asarray(a) - np.asarray(b)))
    assert diff < tol, f"{msg}: max diff = {diff} (tol={tol})"


def make_state():
    grid = get_grid()
    return initialize_state(grid, DEFAULT_CONFIG.physics)


# ── Tests ───────────────────────────────────────────────────────────

def test_rest_state_stays_at_rest():
    """Zero velocity, uniform T/S, no forcing -> remains at rest."""
    grid = get_grid()
    physics = DEFAULT_CONFIG.physics
    state = make_state()

    step(state, grid, physics, DEFAULT_CONFIG.time, dt=600.0)

    assert_allclose(state.u, 0.0, tol=1e-12, msg="u should stay zero")
    assert_allclose(state.v, 0.0, tol=1e-12, msg="v should stay zero")
    assert_allclose(state.T, physics.T_ref, tol=1e-12, msg="T should stay uniform")
    assert_allclose(state.S, physics.S_ref, tol=1e-12, msg="S should stay uniform")
    assert state.diagnostics_filled(), "diagnostics should be filled"
    print("PASS: test_rest_state_stays_at_rest")


def test_coriolis_rotation_helper():
    """Direct sign check of the f-plane rotation helper."""
    from integrator import _coriolis_rotation

    f0 = 1.0e-4
    dt = np.pi / (2.0 * f0)  # quarter inertial period

    # Pure northward -> eastward after quarter period in NH (f0 > 0)
    u0 = np.zeros((4, 4))
    v0 = np.ones((4, 4))
    u1, v1 = _coriolis_rotation(u0, v0, f0, dt)

    assert_allclose(u1, 1.0, tol=1e-12, msg="northward should rotate to eastward")
    assert_allclose(v1, 0.0, tol=1e-12, msg="v should be 0 after quarter period")

    # Pure eastward -> southward after another quarter period
    u2, v2 = _coriolis_rotation(u1, v1, f0, dt)
    assert_allclose(u2, 0.0, tol=1e-12, msg="eastward should rotate to southward")
    assert_allclose(v2, -1.0, tol=1e-12, msg="southward should be -1")

    # Full period returns to start
    u3, v3 = _coriolis_rotation(u2, v2, f0, 2.0 * dt)
    assert_allclose(u3, 0.0, tol=1e-12, msg="u should return to 0 after full period")
    assert_allclose(v3, 1.0, tol=1e-12, msg="v should return to 1 after full period")
    print("PASS: test_coriolis_rotation_helper")


def test_coriolis_rotation_integration():
    """A weak uniform northward jet begins rotating eastward under Coriolis."""
    grid = get_grid()
    physics = DEFAULT_CONFIG.physics
    state = make_state()

    # Weak uniform northward jet (constant -> advection terms vanish, stable)
    state.v[:, :, 0] = 1.0e-4

    # Quarter inertial period, many tiny steps to keep explicit part stable
    f0 = grid.f0
    T_inertial = 2.0 * np.pi / f0
    n_steps = 400
    dt = (T_inertial / 4.0) / n_steps

    for _ in range(n_steps):
        step(state, grid, physics, DEFAULT_CONFIG.time, dt=dt)

    # After a quarter inertial period, northward flow should have rotated
    # mostly eastward in the Northern Hemisphere (f0 > 0).
    mean_u = np.mean(state.u[:, :, 0])
    mean_v = np.mean(state.v[:, :, 0])

    assert mean_u > 0.0, f"u should become eastward, got {mean_u:.5e}"
    assert mean_v > 0.0, f"v should still be positive (quarter period), got {mean_v:.5e}"
    assert mean_u > mean_v, f"eastward component should dominate: u={mean_u:.5e}, v={mean_v:.5e}"
    print(f"PASS: test_coriolis_rotation_integration  (mean u={mean_u:.5e}, mean v={mean_v:.5e})")


def test_wind_stress_spins_up_surface_current():
    """Constant zonal wind stress accelerates surface u."""
    grid = get_grid()
    physics = replace(DEFAULT_CONFIG.physics, tau_x=0.1)
    state = make_state()

    dt = 600.0
    for _ in range(10):
        step(state, grid, physics, DEFAULT_CONFIG.time, dt=dt)

    # Surface layer should have positive zonal velocity
    assert np.mean(state.u[:, :, 0]) > 0.01, f"surface u too small: {np.mean(state.u[:, :, 0]):.5e}"
    print(f"PASS: test_wind_stress_spins_up_surface_current  (mean surface u={np.mean(state.u[:, :, 0]):.5e})")


def test_heat_flux_warms_surface():
    """Constant surface heat flux increases surface temperature."""
    grid = get_grid()
    physics = replace(DEFAULT_CONFIG.physics, Q_heat=200.0)
    state = make_state()

    dt = 600.0
    for _ in range(10):
        step(state, grid, physics, DEFAULT_CONFIG.time, dt=dt)

    surface_temp = np.mean(state.T[:, :, 0])
    assert surface_temp > physics.T_ref, f"surface T did not warm: {surface_temp:.5f}"
    print(f"PASS: test_heat_flux_warms_surface  (surface T={surface_temp:.5f})")


def test_integrate_runs_and_callback():
    """integrate() runs without crashing and callback receives expected times."""
    grid = get_grid()
    physics = DEFAULT_CONFIG.physics
    state = make_state()

    # Short run: 1 hour total, output every 20 min
    time_cfg = replace(DEFAULT_CONFIG.time, dt=600.0, t_total=3600.0, dt_output=1200.0)
    times = []

    def callback(t, s):
        times.append(t)
        assert s.diagnostics_filled(), "state diagnostics should be filled at callback"

    integrate(state, grid, physics, time_cfg, callback=callback)

    # Expected callback times: 0, 1200, 2400, 3600
    expected = [0.0, 1200.0, 2400.0, 3600.0]
    assert_allclose(times, expected, tol=1e-9, msg="callback times mismatch")
    print(f"PASS: test_integrate_runs_and_callback  (times={times})")


def test_step_updates_diagnostics():
    """step() fills w, rho, p diagnostics."""
    grid = get_grid()
    physics = DEFAULT_CONFIG.physics
    state = make_state()

    assert not state.diagnostics_filled(), "initial state should not have diagnostics"

    step(state, grid, physics, DEFAULT_CONFIG.time, dt=600.0)

    assert state.w is not None, "w should be computed"
    assert state.rho is not None, "rho should be computed"
    assert state.p is not None, "p should be computed"
    assert state.w.shape == state.u.shape, "w shape mismatch"
    assert state.rho.shape == state.u.shape, "rho shape mismatch"
    assert state.p.shape == state.u.shape, "p shape mismatch"
    print("PASS: test_step_updates_diagnostics")


def test_eta_unchanged():
    """eta is not evolved in v0.1 (rigid lid)."""
    grid = get_grid()
    physics = DEFAULT_CONFIG.physics
    state = make_state()

    state.eta[:, :] = 0.5
    eta_before = state.eta.copy()

    step(state, grid, physics, DEFAULT_CONFIG.time, dt=600.0)

    assert_allclose(state.eta, eta_before, tol=1e-15, msg="eta should not change")
    print("PASS: test_eta_unchanged")


# ── Run all tests ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Ocean Integrator Tests")
    print("=" * 60)
    print()

    tests = [
        test_rest_state_stays_at_rest,
        test_coriolis_rotation_helper,
        test_coriolis_rotation_integration,
        test_wind_stress_spins_up_surface_current,
        test_heat_flux_warms_surface,
        test_integrate_runs_and_callback,
        test_step_updates_diagnostics,
        test_eta_unchanged,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    print("=" * 60)
