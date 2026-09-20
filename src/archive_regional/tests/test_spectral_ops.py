"""
Test Suite for Spectral Operators
Validates every operator against analytical solutions.

Run: python -m pytest tests/test_spectral_ops.py -v
Or:  python tests/test_spectral_ops.py
"""
import sys
import os
import numpy as np
from numpy import pi, sin, cos, exp

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectral_ops import (
    wavenumbers, wavenumber_grid,
    d_dx, d_dy, d2_dx2, d2_dy2, laplacian_h,
    divergence_h, solve_poisson, pressure_projection,
    linear_step_diffusion, linear_step_coriolis,
    dealias_2_3, spectral_truncate,
    d_dz, d2_dz2,
)


# ── Test helpers ────────────────────────────────────────────────────

def max_rel_error(computed, analytical, tol=1e-10):
    """Max relative error, normalized by global max to avoid zero-crossing blowup."""
    scale = np.max(np.abs(analytical))
    if scale < tol:
        scale = tol
    return np.max(np.abs(computed - analytical)) / scale


def report(name, error, threshold=1e-10):
    status = "PASS" if error < threshold else "FAIL"
    print(f"  [{status}] {name}: max_rel_error = {error:.2e}")
    return error < threshold


# ── Tests ───────────────────────────────────────────────────────────

def test_wavenumbers():
    """Wavenumber array should have correct shape and ordering."""
    print("\n=== test_wavenumbers ===")
    k = wavenumbers(128, 1000.0)
    assert k.shape == (128,), f"Expected (128,), got {k.shape}"
    # k[0] should be 0 (mean mode), k[1] should be positive
    assert abs(k[0]) < 1e-15, f"k[0] should be 0, got {k[0]}"
    assert k[1] > 0, f"k[1] should be positive, got {k[1]}"
    # Nyquist frequency at n/2
    assert abs(k[64]) > 0, f"Nyquist should be nonzero"
    print(f"  k[0]={k[0]:.4e}, k[1]={k[1]:.4e}, k[64]={k[64]:.4e}")
    print("  [PASS] wavenumbers basic structure")
    return True


def test_first_derivative_sin():
    """
    d/dx of sin(2*pi*x/L) = (2*pi/L) * cos(2*pi*x/L)
    """
    print("\n=== test_first_derivative_sin ===")
    nx, ny = 128, 64
    Lx, Ly = 100000.0, 50000.0
    dx, dy = Lx / nx, Ly / ny

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    k = 2 * pi / Lx  # one full wave
    u = sin(k * X)

    du_dx = d_dx(u, dx)
    analytical = k * cos(k * X)

    err = max_rel_error(du_dx, analytical)
    return report("d/dx of sin(kx)", err)


def test_first_derivative_y():
    """
    d/dy of cos(2*pi*y/Ly) = -(2*pi/Ly) * sin(2*pi*y/Ly)
    """
    print("\n=== test_first_derivative_y ===")
    nx, ny = 64, 128
    Lx, Ly = 50000.0, 100000.0
    dx, dy = Lx / nx, Ly / ny

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    k = 2 * pi / Ly
    u = cos(k * Y)

    du_dy = d_dy(u, dy)
    analytical = -k * sin(k * Y)

    err = max_rel_error(du_dy, analytical)
    return report("d/dy of cos(ky)", err)


def test_second_derivative():
    """
    d2/dx2 of sin(kx) = -k^2 * sin(kx)
    """
    print("\n=== test_second_derivative ===")
    nx, ny = 128, 64
    Lx = 100000.0
    dx = Lx / nx
    dy = 50000.0 / ny

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, dy * ny, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    k = 2 * pi / Lx
    u = sin(k * X)

    d2u = d2_dx2(u, dx)
    analytical = -k**2 * sin(k * X)

    err = max_rel_error(d2u, analytical)
    return report("d2/dx2 of sin(kx)", err)


def test_laplacian():
    """
    Laplacian of sin(kx)*sin(ky) = -(kx^2 + ky^2) * sin(kx)*sin(ky)
    """
    print("\n=== test_laplacian ===")
    nx, ny = 128, 128
    Lx, Ly = 100000.0, 100000.0
    dx, dy = Lx / nx, Ly / ny

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    kx = 2 * pi / Lx
    ky = 2 * pi / Ly
    u = sin(kx * X) * sin(ky * Y)

    lap = laplacian_h(u, dx, dy)
    analytical = -(kx**2 + ky**2) * sin(kx * X) * sin(ky * Y)

    err = max_rel_error(lap, analytical)
    return report("Laplacian of sin(kx)*sin(ky)", err)


def test_divergence_free():
    """
    Divergence of a divergence-free field should be ~0.
    u = -y, v = x  =>  div = 0
    But on a periodic domain, u = sin(kx)*cos(ky), v = -cos(kx)*sin(ky) is div-free.
    """
    print("\n=== test_divergence_free ===")
    nx, ny = 128, 128
    Lx, Ly = 100000.0, 100000.0
    dx, dy = Lx / nx, Ly / ny

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    kx = 2 * pi / Lx
    ky = 2 * pi / Ly
    # Stream function: psi = sin(kx*X)*sin(ky*Y)
    # u = -dpsi/dy = -ky*sin(kx*X)*cos(ky*Y)
    # v =  dpsi/dx =  kx*cos(kx*X)*sin(ky*Y)
    # div = du/dx + dv/dy = -ky*kx*cos(kx*X)*cos(ky*Y) + kx*ky*cos(kx*X)*cos(ky*Y) = 0
    u = -ky * sin(kx * X) * cos(ky * Y)
    v =  kx * cos(kx * X) * sin(ky * Y)

    div = divergence_h(u, v, dx, dy)
    err = np.max(np.abs(div))
    print(f"  max|div| = {err:.2e}")
    return report("Divergence-free field", err, threshold=1e-10)


def test_poisson_solver():
    """
    Solve nabla^2 p = rhs where rhs = -k^2 * sin(kx)*sin(ky)
    Expected: p = sin(kx)*sin(ky) (up to constant)
    """
    print("\n=== test_poisson_solver ===")
    nx, ny = 128, 128
    Lx, Ly = 100000.0, 100000.0
    dx, dy = Lx / nx, Ly / ny

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    kx = 2 * pi / Lx
    ky = 2 * pi / Ly
    k2 = kx**2 + ky**2

    p_exact = sin(kx * X) * sin(ky * Y)
    rhs = -k2 * p_exact

    p = solve_poisson(rhs, dx, dy)

    # p and p_exact may differ by a constant; subtract means
    p_shifted = p - np.mean(p)
    p_exact_shifted = p_exact - np.mean(p_exact)

    err = max_rel_error(p_shifted, p_exact_shifted)
    return report("Poisson solver: nabla^2 p = rhs", err)


def test_pressure_projection():
    """
    After pressure projection, the velocity field should be divergence-free.
    """
    print("\n=== test_pressure_projection ===")
    nx, ny = 128, 128
    Lx, Ly = 100000.0, 100000.0
    dx, dy = Lx / nx, Ly / ny

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Start with a divergent velocity field
    k = 2 * pi / Lx
    u = sin(k * X) * cos(k * Y)   # not divergence-free
    v = cos(k * X) * sin(k * Y)   # du/dx + dv/dy != 0

    u_divfree, v_divfree, p = pressure_projection(u, v, dx, dy)

    div = divergence_h(u_divfree, v_divfree, dx, dy)
    err = np.max(np.abs(div))
    print(f"  max|div| after projection = {err:.2e}")
    return report("Pressure projection -> div-free", err, threshold=1e-10)


def test_diffusion_exact():
    """
    Exact solution of du/dt = nu * nabla^2 u with u(x,0) = sin(kx):
      u(x, t) = sin(kx) * exp(-nu * k^2 * t)

    The matrix exponential should reproduce this exactly.
    """
    print("\n=== test_diffusion_exact ===")
    nx, ny = 128, 64
    Lx = 100000.0
    dx = Lx / nx
    dy = 50000.0 / ny

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, dy * ny, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    k = 2 * pi / Lx
    nu = 100.0   # m^2/s
    dt = 600.0   # s

    u0 = sin(k * X)
    u_exact = sin(k * X) * exp(-nu * k**2 * dt)

    u_computed = linear_step_diffusion(u0, nu, dx, dy, dt)

    err = max_rel_error(u_computed, u_exact)
    return report("Diffusion matrix exponential (1 mode)", err)


def test_diffusion_stability_large_dt():
    """
    Demonstrate unconditional stability: a very large dt should still be stable.
    Explicit Euler would blow up at dt > dx^2/(4*nu).
    """
    print("\n=== test_diffusion_stability_large_dt ===")
    nx, ny = 64, 64
    Lx = 50000.0
    dx = Lx / nx
    dy = Lx / ny

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Lx, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    k = 2 * pi / Lx
    nu = 100.0
    # Explicit CFL limit: dt < dx^2 / (4*nu) ~ (781)^2 / 400 ~ 1525 s
    # Use dt = 100000 s (65x beyond explicit limit)
    dt = 100000.0

    u0 = sin(k * X)
    u_result = linear_step_diffusion(u0, nu, dx, dy, dt)

    # Should be stable and decayed (not NaN, not blown up)
    max_val = np.max(np.abs(u_result))
    is_stable = np.isfinite(max_val) and max_val <= 1.0
    print(f"  dt = {dt:.0f}s (explicit limit ~ {dx**2/(4*nu):.0f}s)")
    print(f"  max|u| = {max_val:.6f} (should be <= 1.0, decayed)")
    err = 0.0 if is_stable else 1.0
    return report("Unconditional stability (dt >> CFL)", err, threshold=0.5)


def test_coriolis_rotation():
    """
    Coriolis rotates (u, v) by angle f0*dt.
    After time 2*pi/f0 (inertial period), u should return to original.
    """
    print("\n=== test_coriolis_rotation ===")
    f0 = 8.36e-5  # ~35N
    T_inertial = 2 * pi / f0  # inertial period

    u0 = np.array([1.0, 0.0, 0.5])
    v0 = np.array([0.0, 1.0, -0.3])

    # One full inertial period: should return to original
    u_new, v_new = linear_step_coriolis(u0, v0, f0, T_inertial)

    err = max(np.max(np.abs(u_new - u0)), np.max(np.abs(v_new - v0)))
    return report("Coriolis full inertial period", err)


def test_coriolis_quarter_period():
    """
    After T/4, u -> -v, v -> u (90-degree rotation).
    """
    print("\n=== test_coriolis_quarter_period ===")
    f0 = 8.36e-5
    T_inertial = 2 * pi / f0

    u0 = np.array([1.0, 0.5])
    v0 = np.array([0.0, -0.3])

    u_new, v_new = linear_step_coriolis(u0, v0, f0, T_inertial / 4)

    # After 90deg rotation: u_new ~ -v0, v_new ~ u0
    err_u = np.max(np.abs(u_new - (-v0)))
    err_v = np.max(np.abs(v_new - u0))
    err = max(err_u, err_v)
    return report("Coriolis quarter period (90deg rotation)", err)


def test_spectral_truncate():
    """
    Truncating to lowest 1/2 modes then back should reduce high-freq content.
    """
    print("\n=== test_spectral_truncate ===")
    nx, ny = 128, 128
    Lx = 100000.0
    dx = Lx / nx
    dy = Lx / ny

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Lx, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Two waves: low-freq (mode 1) and high-freq (mode 30)
    k1 = 2 * pi / Lx        # 1 wave
    k2 = 2 * pi * 30 / Lx   # 30 waves -- will be cut by keep_fraction=0.1 (keeps 12)
    u = sin(k1 * X) + 0.5 * sin(k2 * X)

    u_filtered = spectral_truncate(u, keep_fraction=0.1)

    # After filtering, high-freq should be mostly removed
    # Check by comparing to just the low-freq component
    u_low_only = sin(k1 * X)
    err = max_rel_error(u_filtered, u_low_only, tol=0.01)
    return report("Spectral truncation removes high-freq", err, threshold=0.01)


def test_vertical_derivative():
    """
    Test d/dz on a uniform vertical grid.
    d/dz of exp(z) = exp(z)
    """
    print("\n=== test_vertical_derivative ===")
    z = np.linspace(0, -100, 21)  # 21 levels, 5m spacing
    u = np.exp(z / 100.0)  # u = exp(z/100), du/dz = (1/100)*exp(z/100)

    du_dz = d_dz(u, z)
    analytical = (1.0 / 100.0) * np.exp(z / 100.0)

    # Skip boundaries (one-sided less accurate)
    err = max_rel_error(du_dz[1:-1], analytical[1:-1])
    return report("d/dz of exp(z/100)", err, threshold=1e-3)


def test_vertical_second_derivative():
    """
    d2/dz2 of z^2 = 2
    """
    print("\n=== test_vertical_second_derivative ===")
    z = np.linspace(0, -100, 21)
    u = z**2  # d2u/dz2 = 2

    d2u_dz2 = d2_dz2(u, z)

    # Skip boundaries
    err = np.max(np.abs(d2u_dz2[1:-1] - 2.0))
    print(f"  max|d2u/dz2 - 2| (interior) = {err:.2e}")
    return report("d2/dz2 of z^2 = 2", err, threshold=1e-2)


def test_3d_laplacian():
    """
    Test that horizontal Laplacian works on 3D arrays (nx, ny, nz).
    """
    print("\n=== test_3d_laplacian ===")
    nx, ny, nz = 64, 64, 5
    Lx, Ly = 50000.0, 50000.0
    dx, dy = Lx / nx, Ly / ny

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    kx = 2 * pi / Lx
    ky = 2 * pi / Ly

    # Each vertical level gets the same field
    u_2d = sin(kx * X) * sin(ky * Y)
    u_3d = np.stack([u_2d] * nz, axis=2)

    lap_3d = laplacian_h(u_3d, dx, dy)
    analytical_2d = -(kx**2 + ky**2) * u_2d
    analytical_3d = np.stack([analytical_2d] * nz, axis=2)

    err = max_rel_error(lap_3d, analytical_3d)
    return report("3D Laplacian (broadcast over z)", err)


# ── Run all tests ───────────────────────────────────────────────────

def run_all():
    print("=" * 60)
    print("SPECTRAL OPERATORS TEST SUITE")
    print("=" * 60)

    tests = [
        test_wavenumbers,
        test_first_derivative_sin,
        test_first_derivative_y,
        test_second_derivative,
        test_laplacian,
        test_divergence_free,
        test_poisson_solver,
        test_pressure_projection,
        test_diffusion_exact,
        test_diffusion_stability_large_dt,
        test_coriolis_rotation,
        test_coriolis_quarter_period,
        test_spectral_truncate,
        test_vertical_derivative,
        test_vertical_second_derivative,
        test_3d_laplacian,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} passed")
    if passed == total:
        print("ALL TESTS PASSED")
    else:
        print(f"{total - passed} TESTS FAILED")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
