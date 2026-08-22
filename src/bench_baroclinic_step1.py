"""Step 1: Activate baroclinic stratified initial T/S from WOA climatology.

Goal (roadmap Step 1): verify that passing WOA-derived T_init/S_init into
init_state() produces a NON-ZERO density anomaly (rho' != 0), a stable
stratified water column, and measurable mesoscale SSH variance — instead of
the degenerate barotropic (rho'==0) behaviour seen in bench_t3_realdata.

This is a standalone diagnostic, NOT the real-data SLA comparison (Step 5).
It modifies physics only via dataclasses.replace() at test level; the
production DEFAULT_CONFIG is never mutated.

Usage:
    python src/bench_baroclinic_step1.py --spinup-days 5 --nu-bi 3e11
    # --nu-bi 3e11  -> use biharmonic viscosity 3e11 (calibrated default is 1e12)
"""
import sys, os, time, argparse
import numpy as np
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp

from config import DEFAULT_CONFIG
from grid import make_grid
from jax_solver import make_solver
from forcing import wind_stress_gyre, heat_flux_meridional
from woa_data import get_initial_fields
from eos import density_anomaly

DT = 300.0          # s (matches bench_t3_realdata)
LOOKAHEAD_STEPS = 4  # how many dt to advance when probing stability


def summary_rho(rho, ocean):
    """rho: (nx,ny,nz) density anomaly; ocean: (nx,ny) land mask. Return stats."""
    m = ocean[..., None]                  # (nx,ny,1) broadcast over nz
    masked = np.where(m, rho, np.nan)      # NaN on land
    vals = masked[np.isfinite(masked)]
    if vals.size == 0:
        return dict(nonzero=0.0, rms=float("nan"), min=float("nan"), max=float("nan"))
    return dict(
        nonzero=float(np.mean(np.abs(vals) > 1e-6)),
        rms=float(np.sqrt(np.mean(vals ** 2))),
        min=float(vals.min()),
        max=float(vals.max()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spinup-days", type=float, default=5.0)
    ap.add_argument("--nu-bi", type=float, default=None,
                    help="override biharmonic viscosity (default: physics default 1e12)")
    ap.add_argument("--kappa-bi", type=float, default=None)
    ap.add_argument("--probe", action="store_true",
                    help="advance LOOKAHEAD_STEPS to probe numerical stability")
    ap.add_argument("--no-stratification", action="store_true",
                    help="diagnostic: run with uniform T/S (should show rho'~0)")
    args = ap.parse_args()

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    lon = np.asarray(grid.lon)
    lat = np.asarray(grid.lat)
    dx_deg = float(lon[1] - lon[0])
    print("=" * 64)
    print("STEP 1 — Baroclinic activation diagnostic (stratified T/S)")
    print("=" * 64)
    print(f"grid {grid.nx}x{grid.ny}x{grid.nz}, {dx_deg:.2f} deg, "
          f"z-levels: {grid.nz}")

    # ── Physics override (test-level only; production config untouched) ──
    physics = DEFAULT_CONFIG.physics
    if args.nu_bi is not None or args.kappa_bi is not None:
        physics = replace(physics,
                          nu_bi=args.nu_bi if args.nu_bi is not None else physics.nu_bi,
                          kappa_bi=args.kappa_bi if args.kappa_bi is not None else physics.kappa_bi)
        print(f"  [override] nu_bi={physics.nu_bi:g}  kappa_bi={physics.kappa_bi:g}")
    else:
        print(f"  [default ] nu_bi={physics.nu_bi:g}  kappa_bi={physics.kappa_bi:g}")

    # ── Initial T/S ──
    if args.no_stratification:
        print("\nno-stratification diagnostic: T_ref / S_ref uniform field")
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        T_init = np.full((nx, ny, nz), physics.T_ref, dtype=np.float64)
        S_init = np.full((nx, ny, nz), physics.S_ref, dtype=np.float64)
    else:
        print("\nloading WOA2023 annual-mean stratified T/S ...")
        T_init, S_init = get_initial_fields(grid)
        print(f"  T_init shape {T_init.shape}, range [{np.nanmin(T_init):.2f}, "
              f"{np.nanmax(T_init):.2f}] degC, NaN={np.isnan(T_init).sum()}")
        print(f"  S_init shape {S_init.shape}, range [{np.nanmin(S_init):.2f}, "
              f"{np.nanmax(S_init):.2f}] PSU, NaN={np.isnan(S_init).sum()}")

    # ── Density anomaly check (this is THE activation signal) ──
    ocean_mask = np.asarray(grid.ocean_mask, dtype=bool)
    rho = density_anomaly(jnp.array(T_init), jnp.array(S_init), physics,
                          eos_type=physics.eos_type)
    st = summary_rho(np.asarray(rho), ocean_mask)
    print("\n  density anomaly rho' = rho - rho0  (activation signal)")
    print(f"    fraction of ocean with |rho'|>1e-6 : {st['nonzero']:.3f}")
    print(f"    rms / min / max                  : {st['rms']:.5f} / "
          f"{st['min']:.3f} / {st['max']:.3f} kg/m^3")
    if (not args.no_stratification) and st["rms"] < 1e-3:
        print("  !! WARNING: stratification appears negligible — check WOA data / eos_type")
    if args.no_stratification:
        print("  (expected ~0 for uniform T/S — sanity that we can detect rho')")

    # ── Wind + heat forcing (same as bench_t3_realdata) ──
    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)
    step, init_state, _ = make_solver(grid, physics, DT,
                                      forcing=(tau_x, tau_y, Q_heat))
    state = init_state(T_init=jnp.array(T_init), S_init=jnp.array(S_init))

    # ── Short integration ──
    n_spin = int(round(args.spinup_days * 86400.0 / DT))
    print(f"\nintegrating {args.spinup_days:.0f} d ({n_spin} steps, dt={DT:.0f}s) ...")
    t0 = time.time()
    every = max(1, n_spin // 5)
    for n in range(n_spin):
        state = step(state)
        if (n + 1) % every == 0:
            eta_rms = float(jnp.sqrt(jnp.mean(state.eta ** 2)))
            etastd = float(jnp.std(state.eta[np.asarray(grid.ocean_mask)]))
            T_top = float(jnp.nanmean(np.asarray(state.T[:, :, 0])))
            print(f"  step {n+1:7d}/{n_spin}  eta_rms={eta_rms:.5f} m  "
                  f"eta_ocean_std={etastd:.5f} m  T_surf={T_top:.2f} C  "
                  f"({time.time()-t0:.0f}s)")

    # ── Diagnostic: SSH variance vs barotropic baseline ──
    eta = np.asarray(state.eta)
    eta_full = eta if np.isnan(eta[ocean_mask]).sum() == 0 else np.nan_to_num(eta)
    eta_std = float(np.std(eta_full[ocean_mask]))
    T_top = float(np.nanmean(np.asarray(state.T[:, :, 0])))
    T_surf_std = float(np.nanstd(np.asarray(state.T[:, :, 0])[ocean_mask]))
    rho_final = density_anomaly(state.T, state.S, physics, eos_type=physics.eos_type)
    st_f = summary_rho(np.asarray(rho_final), ocean_mask)

    print("\n────────────────────────────────────────────────────")
    print("STEP 1 RESULT")
    print(f"  SSH ocean std           : {eta_std:.5f} m   "
          f"(barotropic baseline was ~0.004 m)")
    print(f"  surface T mean / std    : {T_top:.2f} C / {T_surf_std:.3f} C")
    print(f"  rho' rms at end         : {st_f['rms']:.5f} kg/m^3  "
          f"(nonzero={st_f['nonzero']:.3f})")
    sig_strat = "ACTIVATED (rho' >> 0)" if st_f["rms"] > 1e-3 else "DEGENERATE (rho' ~ 0)"
    print(f"  stratification status   : {sig_strat}")

    # ── Optional numerical-stability probe ──
    if args.probe:
        print(f"\nprobing stability {LOOKAHEAD_STEPS} additional steps ...")
        max_eta = 0.0
        for n in range(LOOKAHEAD_STEPS):
            state = step(state)
            e = float(jnp.max(jnp.abs(np.nan_to_num(state.eta))))
            max_eta = max(max_eta, e)
        print(f"  max|eta| after probe    : {max_eta:.5f} m"
              f"  {'(stable)' if np.isfinite(max_eta) and max_eta < 1.0 else '(UNSTABLE: check nu_bi)'}")

    print("\ndone.")


if __name__ == "__main__":
    main()
