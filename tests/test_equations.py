"""
Tests for ocean equation modules (EOS, pressure, momentum, tracers).

Run: python tests/test_equations.py
"""
import sys
import os
import numpy as np
from dataclasses import replace

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import (
    DEFAULT_CONFIG, RHO_0, G_EARTH, ALPHA_T, BETA_S, C_P,
)
from grid import make_grid
from state import ModelState, initialize_state
from eos import compute_density, compute_buoyancy, density_anomaly
from pressure import (
    compute_hydrostatic_pressure, compute_pressure_gradient,
)
from momentum import compute_momentum_tendency, compute_vertical_velocity
from tracers import compute_tracer_tendency


# ── Test helpers ────────────────────────────────────────────────────

def get_grid():
    """Load grid once (cached via module-level lazy init)."""
    if not hasattr(get_grid, '_cache'):
        get_grid._cache = make_grid(
            DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file
        )
    return get_grid._cache


def assert_close(a, b, tol=1e-6, msg=""):
    assert abs(a - b) < tol, f"{msg}: {a} != {b} (tol={tol})"


# ── Rest state tests ────────────────────────────────────────────────

def test_rest_state_zero_momentum_tendency():
    """Rest state with no forcing -> zero momentum tendency."""
    g = get_grid()
    p = DEFAULT_CONFIG.physics
    state = initialize_state(g, p)
    dudt, dvdt = compute_momentum_tendency(state, g, p)
    max_dudt = np.max(np.abs(dudt))
    max_dvdt = np.max(np.abs(dvdt))
    assert max_dudt < 1e-12, f"dudt not zero: max={max_dudt}"
    assert max_dvdt < 1e-12, f"dvdt not zero: max={max_dvdt}"
    print(f"PASS: test_rest_state_zero_momentum_tendency"
          f"  (max|dudt|={max_dudt:.2e}, max|dvdt|={max_dvdt:.2e})")


def test_rest_state_zero_tracer_tendency():
    """Rest state with no forcing -> zero tracer tendency."""
    g = get_grid()
    p = DEFAULT_CONFIG.physics
    state = initialize_state(g, p)
    dTdt, dSdt = compute_tracer_tendency(state, g, p)
    max_dTdt = np.max(np.abs(dTdt))
    max_dSdt = np.max(np.abs(dSdt))
    assert max_dTdt < 1e-12, f"dTdt not zero: max={max_dTdt}"
    assert max_dSdt < 1e-12, f"dSdt not zero: max={max_dSdt}"
    print(f"PASS: test_rest_state_zero_tracer_tendency"
          f"  (max|dTdt|={max_dTdt:.2e}, max|dSdt|={max_dSdt:.2e})")


# ── Shape tests ─────────────────────────────────────────────────────

def test_tendency_shapes():
    """All tendency arrays have shape (nx, ny, nz)."""
    g = get_grid()
    p = DEFAULT_CONFIG.physics
    state = initialize_state(g, p)
    dudt, dvdt = compute_momentum_tendency(state, g, p)
    dTdt, dSdt = compute_tracer_tendency(state, g, p)
    expected = (g.nx, g.ny, g.nz)
    assert dudt.shape == expected, f"dudt shape: {dudt.shape} != {expected}"
    assert dvdt.shape == expected, f"dvdt shape: {dvdt.shape} != {expected}"
    assert dTdt.shape == expected, f"dTdt shape: {dTdt.shape} != {expected}"
    assert dSdt.shape == expected, f"dSdt shape: {dSdt.shape} != {expected}"
    print(f"PASS: test_tendency_shapes  (all {expected})")


# ── EOS tests ───────────────────────────────────────────────────────

def test_density_warm_lighter():
    """Warmer water -> lower density; saltier water -> higher density."""
    p = DEFAULT_CONFIG.physics
    rho_ref = compute_density(
        np.array(p.T_ref), np.array(p.S_ref), p
    )
    assert_close(rho_ref, RHO_0, tol=1e-10, msg="rho at T_ref, S_ref")
    rho_warm = compute_density(
        np.array(p.T_ref + 1.0), np.array(p.S_ref), p
    )
    assert rho_warm < rho_ref, \
        f"Warm water should be lighter: {rho_warm} vs {rho_ref}"
    rho_salty = compute_density(
        np.array(p.T_ref), np.array(p.S_ref + 1.0), p
    )
    assert rho_salty > rho_ref, \
        f"Salty water should be denser: {rho_salty} vs {rho_ref}"
    print(f"PASS: test_density_warm_lighter"
          f"  (rho_ref={rho_ref:.2f}, rho_warm={rho_warm:.2f},"
          f" rho_salty={rho_salty:.2f})")


def test_uniform_TS_zero_density_anomaly():
    """Uniform T=T_ref, S=S_ref -> zero density anomaly."""
    g = get_grid()
    p = DEFAULT_CONFIG.physics
    state = initialize_state(g, p)
    rho_prime = density_anomaly(state.T, state.S, p)
    max_rho = np.max(np.abs(rho_prime))
    assert max_rho < 1e-12, f"density anomaly not zero: max={max_rho}"
    print(f"PASS: test_uniform_TS_zero_density_anomaly"
          f"  (max|rho'|={max_rho:.2e})")


# ── Pressure tests ──────────────────────────────────────────────────

def test_pressure_monotonic_with_depth():
    """Cold anomaly (uniform) -> pressure increases monotonically with depth."""
    g = get_grid()
    p = DEFAULT_CONFIG.physics
    state = initialize_state(g, p)
    # Make T 1 degree colder everywhere -> positive density anomaly
    state.T = np.full_like(state.T, p.T_ref - 1.0)

    pressure = compute_hydrostatic_pressure(state, g, p)

    # Surface pressure should be 0 (eta=0, p_bc[0]=0)
    max_surf = np.max(np.abs(pressure[:, :, 0]))
    assert max_surf < 1e-10, f"Surface pressure not zero: max={max_surf}"

    # Pressure should increase monotonically with depth index
    for k in range(g.nz - 1):
        diff = pressure[:, :, k + 1] - pressure[:, :, k]
        assert np.all(diff > 0), \
            f"Pressure not increasing at k={k}: min diff={diff.min()}"

    # Check exact value at k=1: p_bc[1] = G * rho' * dz[0]
    rho_prime = density_anomaly(state.T, state.S, p)
    expected_p1 = G_EARTH * rho_prime[0, 0, 0] * g.dz[0]
    assert_close(
        pressure[0, 0, 1], expected_p1, tol=1e-8,
        msg="pressure at k=1"
    )
    print(f"PASS: test_pressure_monotonic_with_depth"
          f"  (p_surf={max_surf:.2e}, p_bottom={pressure[0, 0, -1]:.2f} Pa)")


def test_pressure_gradient_from_eta():
    """Nonzero eta -> barotropic pressure gradient."""
    g = get_grid()
    p = DEFAULT_CONFIG.physics
    state = initialize_state(g, p)

    # Sinusoidal eta in x (exactly one period, periodic)
    nx = g.nx
    A = 0.1  # 10 cm SSH anomaly
    eta_pattern = A * np.sin(2 * np.pi * np.arange(nx) / nx)
    state.eta = np.tile(
        eta_pattern[:, np.newaxis], (1, g.ny)
    )

    pgf_x, pgf_y = compute_pressure_gradient(state, g, p)

    # p = RHO_0 * G * eta (barotropic only, rho'=0)
    # pgf_x = -dp/dx / RHO_0 = -G * d(eta)/dx
    # d(eta)/dx[j] = A * (2*pi/(nx*dx)) * cos(2*pi*j/nx)  (exact for spectral)
    expected_pgf = -G_EARTH * A * (2 * np.pi / (nx * g.dx))
    actual_pgf = pgf_x[0, 64, 0]  # at j=0, center lat, surface

    assert_close(
        actual_pgf, expected_pgf, tol=1e-6,
        msg="barotropic pgf_x at j=0"
    )

    # pgf_y should be ~0 (eta doesn't vary in y)
    max_pgf_y = np.max(np.abs(pgf_y))
    assert max_pgf_y < 1e-10, \
        f"pgf_y should be zero, max={max_pgf_y}"

    print(f"PASS: test_pressure_gradient_from_eta"
          f"  (pgf_x={actual_pgf:.5e}, expected={expected_pgf:.5e})")


def test_baroclinic_pressure_gradient():
    """Warm anomaly creates baroclinic pgf pointing toward warm region."""
    g = get_grid()
    p = DEFAULT_CONFIG.physics
    state = initialize_state(g, p)

    # Warm anomaly: T = T_ref + A*cos(2*pi*x/Lx), uniform in y and z
    nx = g.nx
    A = 1.0  # 1C amplitude
    pattern = A * np.cos(2 * np.pi * np.arange(nx) / nx)
    state.T = p.T_ref + np.tile(
        pattern[:, np.newaxis, np.newaxis], (1, g.ny, g.nz)
    )

    pgf_x, pgf_y = compute_pressure_gradient(state, g, p)

    # pgf_y should be ~0 (no y variation)
    max_pgf_y = np.max(np.abs(pgf_y))
    assert max_pgf_y < 1e-10, \
        f"pgf_y should be zero, max={max_pgf_y}"

    # At surface (k=0), p_bc=0 so pgf_x should be ~0
    max_pgf_surf = np.max(np.abs(pgf_x[:, :, 0]))
    assert max_pgf_surf < 1e-10, \
        f"Surface pgf_x should be zero, max={max_pgf_surf}"

    k_test = 5  # z = -50m

    # At j=nx/4: T at reference, transitioning warm->cool
    # p increasing in +x (warm low -> cool high) -> dp/dx > 0 -> pgf_x < 0
    j1 = nx // 4
    assert pgf_x[j1, 64, k_test] < 0, \
        f"pgf_x at warm->cool transition should be < 0, " \
        f"got {pgf_x[j1, 64, k_test]}"

    # At j=3*nx/4: T at reference, transitioning cool->warm
    # p decreasing in +x -> dp/dx < 0 -> pgf_x > 0
    j2 = 3 * nx // 4
    assert pgf_x[j2, 64, k_test] > 0, \
        f"pgf_x at cool->warm transition should be > 0, " \
        f"got {pgf_x[j2, 64, k_test]}"

    # Magnitude should increase with depth (more warm column integrated)
    mag_shallow = abs(pgf_x[j1, 64, 2])
    mag_deep = abs(pgf_x[j1, 64, 8])
    assert mag_deep > mag_shallow, \
        f"Baroclinic pgf should increase with depth: " \
        f"shallow={mag_shallow}, deep={mag_deep}"

    print(f"PASS: test_baroclinic_pressure_gradient"
          f"  (pgf at j={j1}, k={k_test}: {pgf_x[j1, 64, k_test]:.5e})")


# ── Surface forcing tests ───────────────────────────────────────────

def test_wind_stress_surface_layer():
    """Wind stress accelerates only the surface layer."""
    g = get_grid()
    p = replace(DEFAULT_CONFIG.physics, tau_x=0.1)  # 0.1 N/m^2
    state = initialize_state(g, DEFAULT_CONFIG.physics)
    dudt, dvdt = compute_momentum_tendency(state, g, p)

    dz_surface = abs(g.z[0] - g.z[1])  # 5.0 m
    expected = p.tau_x / (RHO_0 * dz_surface)

    # Surface layer: positive acceleration, uniform magnitude
    assert np.all(dudt[:, :, 0] > 0), \
        f"Surface dudt should be positive, mean={dudt[:, :, 0].mean()}"
    assert_close(
        np.mean(dudt[:, :, 0]), expected, tol=1e-10,
        msg="wind stress magnitude"
    )

    # All other layers: zero (rest state, no other forcing)
    max_below = np.max(np.abs(dudt[:, :, 1:]))
    assert max_below < 1e-12, \
        f"dudt below surface should be zero, max={max_below}"

    # dvdt should be zero everywhere (tau_y = 0)
    max_dvdt = np.max(np.abs(dvdt))
    assert max_dvdt < 1e-12, \
        f"dvdt should be zero, max={max_dvdt}"

    print(f"PASS: test_wind_stress_surface_layer"
          f"  (dudt_surf={expected:.5e} m/s^2)")


def test_heat_flux_surface_layer():
    """Surface heat flux warms only the top layer."""
    g = get_grid()
    p = replace(DEFAULT_CONFIG.physics, Q_heat=100.0)  # 100 W/m^2
    state = initialize_state(g, DEFAULT_CONFIG.physics)
    dTdt, dSdt = compute_tracer_tendency(state, g, p)

    dz_surface = abs(g.z[0] - g.z[1])  # 5.0 m
    expected = p.Q_heat / (RHO_0 * C_P * dz_surface)

    # Surface layer: positive warming
    assert np.all(dTdt[:, :, 0] > 0), \
        f"Surface dTdt should be positive"
    assert_close(
        np.mean(dTdt[:, :, 0]), expected, tol=1e-10,
        msg="heat flux magnitude"
    )

    # All other layers: zero
    max_below = np.max(np.abs(dTdt[:, :, 1:]))
    assert max_below < 1e-12, \
        f"dTdt below surface should be zero, max={max_below}"

    # dSdt should be zero (no salinity flux in v0.1)
    max_dSdt = np.max(np.abs(dSdt))
    assert max_dSdt < 1e-12, \
        f"dSdt should be zero, max={max_dSdt}"

    print(f"PASS: test_heat_flux_surface_layer"
          f"  (dTdt_surf={expected:.5e} C/s)")


# ── Coriolis test ───────────────────────────────────────────────────

def test_coriolis_eastward_deflection():
    """Eastward velocity -> southward Coriolis deflection in NH."""
    g = get_grid()
    p = DEFAULT_CONFIG.physics
    state = initialize_state(g, p)
    state.u = np.ones_like(state.u)  # 1 m/s eastward everywhere

    dudt, dvdt = compute_momentum_tendency(state, g, p)

    # Coriolis: dvdt = -f*u (negative in NH = southward)
    f0 = DEFAULT_CONFIG.grid.f0
    k_mid = g.nz // 2  # avoid bottom friction layer

    # dvdt at center should be -f0 (u=1)
    dvdt_center = dvdt[64, 64, k_mid]
    expected = -f0 * 1.0
    assert_close(
        dvdt_center, expected, tol=1e-7,
        msg="Coriolis dvdt at center"
    )

    # dudt should be ~0 at non-bottom levels
    # (Coriolis f*v=0, advection=0 for uniform u, diffusion=0)
    dudt_surface = dudt[64, 64, 0]
    assert abs(dudt_surface) < 1e-10, \
        f"dudt at surface should be ~0, got {dudt_surface}"

    # Bottom friction: dudt at deepest level should be -r_bot * u = -r_bot
    r_bot = p.r_bot
    dudt_bottom = dudt[64, 64, -1]
    expected_bottom = -r_bot * 1.0
    assert_close(
        dudt_bottom, expected_bottom, tol=1e-7,
        msg="bottom friction dudt"
    )

    print(f"PASS: test_coriolis_eastward_deflection"
          f"  (dvdt={dvdt_center:.5e}, expected={expected:.5e})")


# ── Vertical velocity tests ─────────────────────────────────────────

def test_vertical_velocity_bottom_bc():
    """Vertical velocity is zero at the bottom."""
    g = get_grid()
    p = DEFAULT_CONFIG.physics
    state = initialize_state(g, p)

    # Rest state -> w = 0 everywhere
    w = compute_vertical_velocity(state, g)
    max_w = np.max(np.abs(w))
    assert max_w < 1e-12, f"w not zero for rest state: max={max_w}"

    # Bottom BC: w[..., -1] = 0 exactly
    max_w_bottom = np.max(np.abs(w[:, :, -1]))
    assert max_w_bottom < 1e-15, \
        f"w at bottom should be exactly 0, got {max_w_bottom}"

    print(f"PASS: test_vertical_velocity_bottom_bc"
          f"  (max|w|={max_w:.2e})")


def test_vertical_velocity_nonzero_divergence():
    """Nonzero horizontal divergence -> nonzero interior w."""
    g = get_grid()
    p = DEFAULT_CONFIG.physics
    state = initialize_state(g, p)

    # Divergent velocity: u = sin(2*pi*x/Lx), v = 0
    nx = g.nx
    u_pattern = np.sin(2 * np.pi * np.arange(nx) / nx)
    state.u = np.tile(
        u_pattern[:, np.newaxis, np.newaxis], (1, g.ny, g.nz)
    )

    w = compute_vertical_velocity(state, g)

    # Shape check
    assert w.shape == (g.nx, g.ny, g.nz), f"w shape: {w.shape}"

    # Bottom BC
    max_w_bottom = np.max(np.abs(w[:, :, -1]))
    assert max_w_bottom < 1e-15, \
        f"w at bottom should be 0, got {max_w_bottom}"

    # Interior should be nonzero (divergence is nonzero)
    max_w_interior = np.max(np.abs(w[:, :, :-1]))
    assert max_w_interior > 1e-6, \
        f"Interior w should be nonzero, max={max_w_interior}"

    print(f"PASS: test_vertical_velocity_nonzero_divergence"
          f"  (max|w|={max_w_interior:.5e})")


# ── End-to-end smoke test ───────────────────────────────────────────

def test_end_to_end_smoke():
    """Full pipeline: grid -> state -> diagnostics -> tendencies."""
    g = get_grid()
    p = DEFAULT_CONFIG.physics
    state = initialize_state(g, p)

    # Fill all diagnostics
    state.rho = compute_density(state.T, state.S, p)
    state.p = compute_hydrostatic_pressure(state, g, p)
    state.w = compute_vertical_velocity(state, g)
    assert state.diagnostics_filled(), "Diagnostics should be filled"

    # Compute tendencies
    dudt, dvdt = compute_momentum_tendency(state, g, p)
    dTdt, dSdt = compute_tracer_tendency(state, g, p)

    # Verify shapes
    expected = (g.nx, g.ny, g.nz)
    for arr, name in [
        (dudt, "dudt"), (dvdt, "dvdt"),
        (dTdt, "dTdt"), (dSdt, "dSdt"),
    ]:
        assert arr.shape == expected, f"{name} shape: {arr.shape}"

    # Rest state -> all tendencies zero
    for arr, name in [
        (dudt, "dudt"), (dvdt, "dvdt"),
        (dTdt, "dTdt"), (dSdt, "dSdt"),
    ]:
        max_val = np.max(np.abs(arr))
        assert max_val < 1e-12, f"{name} not zero: max={max_val}"

    print(f"PASS: test_end_to_end_smoke"
          f"  (shapes={expected}, all tendencies ~0)")


# ── Run all tests ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Ocean Equation Tests")
    print("=" * 60)
    print()

    tests = [
        test_rest_state_zero_momentum_tendency,
        test_rest_state_zero_tracer_tendency,
        test_tendency_shapes,
        test_density_warm_lighter,
        test_uniform_TS_zero_density_anomaly,
        test_pressure_monotonic_with_depth,
        test_pressure_gradient_from_eta,
        test_baroclinic_pressure_gradient,
        test_wind_stress_surface_layer,
        test_heat_flux_surface_layer,
        test_coriolis_eastward_deflection,
        test_vertical_velocity_bottom_bc,
        test_vertical_velocity_nonzero_divergence,
        test_end_to_end_smoke,
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
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)
