"""
Physics validation: wind forcing produces correct Ekman transport
direction and Sverdrup interior response.

Tests three aspects of the 2D wind stress forcing:
  1. Ekman transport direction -- uniform eastward wind on an f-plane
     (Northern Hemisphere) should drive southward surface flow
     (transport to the right of the wind).
  2. Sverdrup response -- Stommel gyre wind curl should produce
     downward Ekman pumping and equatorward barotropic transport.
  3. Forcing field sanity -- wind_stress_gyre produces the correct
     antisymmetric gyre profile with tau_y = 0, peaked at tau0 in the
     interior, and tapered to zero at both y-edges so it is continuous
     across the solver's periodic meridional seam.

Uses the numpy solver only (no JAX) for direct state inspection.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import DEFAULT_CONFIG, RHO_0
from grid import make_grid
from state import initialize_state
from integrator import step
from forcing import Forcing, wind_stress_gyre, ekman_pumping


def _barotropic_velocity(state, grid):
    """Depth-averaged horizontal velocity (mirrors integrator definition)."""
    dz = np.array(grid.dz)
    H_sw = float(np.sum(dz))
    dz_norm = dz.reshape(1, 1, -1) / H_sw
    u_avg = 0.5 * (state.u[..., :-1] + state.u[..., 1:])
    v_avg = 0.5 * (state.v[..., :-1] + state.v[..., 1:])
    ubt = np.sum(u_avg * dz_norm, axis=-1)
    vbt = np.sum(v_avg * dz_norm, axis=-1)
    return ubt, vbt


def _interior_mean(field, ocean_mask, margin=5):
    """Mean of a 2D field over interior ocean points, excluding a margin."""
    s = (slice(margin, -margin), slice(margin, -margin))
    sub = field[s]
    mask = ocean_mask[s]
    if mask.any():
        return float(np.mean(sub[mask]))
    return float(np.mean(sub))


def test_forcing_field_sanity(grid, tau0=0.1):
    """Verify wind_stress_gyre produces a continuous, tapered Stommel
    profile that is safe under the solver's spectral (periodic) FFT.
    """
    tau_x, tau_y = wind_stress_gyre(grid, tau0=tau0)
    ny = grid.ny

    tau_y_max = float(np.max(np.abs(tau_y)))
    tau_x_max = float(np.max(np.abs(tau_x)))
    # Antisymmetry about y-center: tau_x[:, j] = -tau_x[:, ny-1-j]
    antisym_err = float(np.max(np.abs(tau_x + tau_x[:, ::-1])))
    # Boundary continuity: forcing must be ~0 at both edges so the periodic
    # FFT derivatives do not see a step discontinuity at the meridional seam.
    south_edge = float(np.mean(tau_x[:, 0]))
    north_edge = float(np.mean(tau_x[:, -1]))
    # Interior gyre sign: easterly in the south, westerly in the north,
    # evaluated away from the taper zone.
    margin = 2 * 8  # twice the default taper width
    south_int = float(np.mean(tau_x[:, margin]))
    north_int = float(np.mean(tau_x[:, ny - 1 - margin]))

    print("--- Test 3: Forcing Field Sanity ---")
    print(f"  tau_y max:          {tau_y_max:.2e} (expect ~0)")
    print(f"  max|tau_x|:         {tau_x_max:.6f} (expect ~{tau0})")
    print(f"  antisymmetry err:   {antisym_err:.2e} (expect ~0)")
    print(f"  south edge tau_x:   {south_edge:.3e} (expect ~0, seam-continuous)")
    print(f"  north edge tau_x:   {north_edge:.3e} (expect ~0, seam-continuous)")
    print(f"  south interior:     {south_int:.6f} (expect < 0, easterly)")
    print(f"  north interior:     {north_int:.6f} (expect > 0, westerly)")

    ok = (
        tau_y_max < 1e-15
        and abs(tau_x_max - tau0) < 1e-2
        and antisym_err < 1e-10
        and abs(south_edge) < 1e-12
        and abs(north_edge) < 1e-12
        and south_int < 0
        and north_int > 0
    )
    print(f"  [{'PASS' if ok else 'FAIL'}] forcing field sanity")
    print()
    return ok


def test_ekman_transport(grid, physics, tau0=0.1, n_steps=20):
    """Uniform eastward wind on f-plane should drive southward surface flow.

    For f0 > 0 (NH), Coriolis turns wind-driven u to the right,
    producing negative v (southward) in the surface layer.
    """
    nx, ny = grid.nx, grid.ny
    f0 = grid.f0

    tau_x_uniform = np.full((nx, ny), tau0)
    forcing = Forcing(tau_x=tau_x_uniform)

    state = initialize_state(grid, physics)
    dt = DEFAULT_CONFIG.time.dt

    t0 = time.perf_counter()
    for _ in range(n_steps):
        step(state, grid, physics, DEFAULT_CONFIG.time, forcing=forcing)
    elapsed = time.perf_counter() - t0

    v_surf = state.v[:, :, 0]
    mask = grid.ocean_mask
    v_surf_mean = (
        float(np.mean(v_surf[mask])) if mask.any() else float(np.mean(v_surf))
    )
    # Theoretical Ekman transport: M_y = -tau_x / (rho_0 * f0)  [m^2/s]
    M_y_theory = -tau0 / (RHO_0 * f0)

    print("--- Test 1: Ekman Transport Direction ---")
    print(f"  wind: tau_x = {tau0} N/m^2 (eastward, uniform)")
    print(f"  f0 = {f0:.5e} /s (NH, f0 > 0)")
    print(f"  steps: {n_steps} (dt={dt}s, total={n_steps * dt / 3600:.1f}h)")
    print(f"  wall time: {elapsed:.1f}s")
    print(f"  v_surface mean: {v_surf_mean:.6e} m/s (expect < 0)")
    print(f"  Ekman M_y:      {M_y_theory:.6e} m^2/s (theory)")

    ok = v_surf_mean < 0
    print(f"  [{'PASS' if ok else 'FAIL'}] surface v is southward (right of wind)")
    print()
    return ok


def test_sverdrup_response(grid, physics, tau0=0.1, n_steps=10):
    """Stommel gyre wind curl should drive downward Ekman pumping and
    equatorward barotropic transport.

    Sverdrup balance: beta * V = curl(tau) / rho_0
    With curl(tau) < 0 and beta > 0, V < 0 (southward).

    A short integration (10 steps = 100 min) is sufficient to detect the
    correct sign of the barotropic response while staying numerically
    stable and well within the 30s budget.
    """
    beta = grid.beta
    f0 = grid.f0

    tau_x, tau_y = wind_stress_gyre(grid, tau0=tau0)
    forcing = Forcing(tau_x=tau_x, tau_y=tau_y)

    # Ekman pumping (analytical, no integration needed)
    w_ek = ekman_pumping(grid, tau_x, tau_y)
    w_ek_mean = _interior_mean(w_ek, grid.ocean_mask)

    # Integrate model with Stommel gyre wind
    state = initialize_state(grid, physics)
    dt = DEFAULT_CONFIG.time.dt

    t0 = time.perf_counter()
    for _ in range(n_steps):
        step(state, grid, physics, DEFAULT_CONFIG.time, forcing=forcing)
    elapsed = time.perf_counter() - t0

    _, vbt = _barotropic_velocity(state, grid)
    vbt_mean = _interior_mean(vbt, grid.ocean_mask)
    has_nan = bool(np.any(np.isnan(vbt)))

    print("--- Test 2: Sverdrup Response ---")
    print(f"  wind: Stommel gyre, tau0 = {tau0} N/m^2")
    print(f"  beta = {beta:.5e} /(m*s),  f0 = {f0:.5e} /s")
    print(f"  steps: {n_steps} (dt={dt}s, total={n_steps * dt / 3600:.1f}h)")
    print(f"  wall time: {elapsed:.1f}s")
    print(f"  w_ek interior: {w_ek_mean:.6e} m/s (expect < 0, downwelling)")
    print(f"  vbt interior:  {vbt_mean:.6e} m/s (expect < 0, southward)")
    if has_nan:
        print(f"  WARNING: NaN detected in barotropic velocity")

    ok_w = w_ek_mean < 0
    ok_v = vbt_mean < 0 and not has_nan
    print(f"  [{'PASS' if ok_w else 'FAIL'}] Ekman pumping downward")
    print(f"  [{'PASS' if ok_v else 'FAIL'}] barotropic transport southward")
    print()
    return ok_w and ok_v


def main():
    print("=== Wind Forcing Physics Validation ===")
    print()

    t0 = time.perf_counter()
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    t_grid = time.perf_counter() - t0

    print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}  (created in {t_grid:.1f}s)")
    print(f"Lat: {grid.lat[0]:.1f} - {grid.lat[-1]:.1f} N")
    print(f"f0 = {grid.f0:.5e} /s,  beta = {grid.beta:.5e} /(m*s)")
    n_ocean = int(grid.ocean_mask.sum())
    print(f"Ocean: {n_ocean} / {grid.nx * grid.ny} points")
    print()

    results = [
        ("Forcing field sanity", test_forcing_field_sanity(grid)),
        ("Ekman transport direction", test_ekman_transport(grid, physics)),
        ("Sverdrup response", test_sverdrup_response(grid, physics)),
    ]

    print("=" * 55)
    print("Summary:")
    all_pass = True
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_pass = all_pass and ok
    print()
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")


if __name__ == "__main__":
    main()
