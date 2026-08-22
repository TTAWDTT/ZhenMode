"""Step 2: Recalibrate biharmonic viscosity for stratified flow.

Roadmap Step 2. With baroclinic T/S active (Step 1), nu_bi=1e12 (calibrated
against UNSTRATIFIED flow to suppress eddies) may over-damp mesoscale eddies,
so SSH variance never reaches an eddy equilibrium and instead stays as
initial geostrophic-imbalance gravity-wave oscillation.

This script scans several nu_bi values over a longer integration and records
the SSH-variance time series to see whether eta_std:
  - decays to a STABLE eddy-equilibrium plateau (good),
  - stays as large-amplitude gravity-wave oscillation (over/under-damped),
  - blows up (unstable).

It overrides physics at test level only via dataclasses.replace(); the
production DEFAULT_CONFIG is never mutated.

Usage:
    python src/bench_baroclinic_step2.py --spinup-days 20 --nu-bi 1e11,3e11,1e12
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

DT = 300.0                 # s
SNAP_STEPS = 48            # record every 48 steps (4 h at dt=300s)
SAVE = "results/step2_scan.npz"


def run_case(nu_bi, kappa_bi, grid, physics0, T_init, S_init,
             tau_x, tau_y, Q_heat, spinup_days, ocean_mask,
             tau_restore_days=0.0):
    # Optional Haney surface-T restoring (Step 3) to suppress thermal drift.
    T_sst = T_init[:, :, 0] if tau_restore_days > 0 else None
    physics = replace(physics0, nu_bi=nu_bi, kappa_bi=kappa_bi)
    step, init_state, _ = make_solver(grid, physics, DT,
                                      forcing=(tau_x, tau_y, Q_heat),
                                      T_sst=T_sst,
                                      tau_restore_days=tau_restore_days)
    state = init_state(T_init=jnp.array(T_init), S_init=jnp.array(S_init))
    n_spin = int(round(spinup_days * 86400.0 / DT))
    n_snap = n_spin // SNAP_STEPS + 1
    t_days = np.empty(n_snap)
    eta_std = np.empty(n_snap)
    T_surf = np.empty(n_snap)
    i = 0
    t_days[i], eta_std[i], T_surf[i] = 0.0, 0.0015, float(np.nanmean(T_init[:, :, 0]))
    i += 1
    t0 = time.time()
    om = ocean_mask
    for n in range(1, n_spin + 1):
        state = step(state)
        if n % SNAP_STEPS == 0 or n == n_spin:
            e = np.asarray(state.eta)
            t_days[i] = n * DT / 86400.0
            eta_std[i] = float(np.std(e[om]))
            T_surf[i] = float(np.nanmean(np.asarray(state.T[:, :, 0])))
            i += 1
    return (n_spin, time.time() - t0), t_days[:i], eta_std[:i], T_surf[:i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spinup-days", type=float, default=20.0)
    ap.add_argument("--nu-bi", default="1e11,3e11,1e12",
                    help="comma-separated nu_bi scan values")
    ap.add_argument("--tau-restore-days", type=float, default=0.0,
                    help="Haney surface-T restoring timescale (0 disables)")
    args = ap.parse_args()
    nu_bis = [float(x) for x in args.nu_bi.split(",")]

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    ocean_mask = np.asarray(grid.ocean_mask, dtype=bool)
    physics0 = DEFAULT_CONFIG.physics
    print("=" * 64)
    print("STEP 2 — biharmonic viscosity scan (stratified, longer integration)")
    print("=" * 64)
    print(f"grid {grid.nx}x{grid.ny}x{grid.nz}  spinup={args.spinup_days:.0f}d  "
          f"dt={DT:.0f}s  nu_bi scan = {args.nu_bi}")

    T_init, S_init = get_initial_fields(grid)
    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)

    print("\n  running cases ...\n")
    results = {}
    for nu_bi in nu_bis:
        kappa_bi = nu_bi  # keep kappa_bi tied to nu_bi for this scan
        label = f"{nu_bi:.0e}"
        print(f"  === nu_bi = {label}  (restore={args.tau_restore_days:g}d) ===")
        dur, t_d, es, ts = run_case(nu_bi, kappa_bi, grid, physics0, T_init, S_init,
                                    tau_x, tau_y, Q_heat,
                                    args.spinup_days, ocean_mask,
                                    tau_restore_days=args.tau_restore_days)
        n_spin, wall = dur
        # late-window stats (last 2 days) to see if it reaches a plateau
        late = es[-16:]
        print(f"      ran {n_spin} steps in {wall:.0f}s")
        print(f"      eta_std: start={es[0]:.4f}  mid={es[len(es)//2]:.4f}  "
              f"end={es[-1]:.4f} m")
        print(f"      late-2d: mean={np.mean(late):.4f}  std={np.std(late):.4f} m  "
              f"({'stable plateau' if np.std(late) < 0.05*max(np.abs(es)) or np.std(late)<0.02 else 'still varying'})")
        print(f"      T_surf end={ts[-1]:.2f} C\n")
        results[label] = dict(t_days=t_d, eta_std=es, T_surf=ts)

    os.makedirs("results", exist_ok=True)
    np.savez(SAVE, nu_bis=np.array(nu_bis),
             **{f"{k}__eta_std": v["eta_std"] for k, v in results.items()},
             **{f"{k}__t_days": v["t_days"] for k, v in results.items()})
    print(f"saved -> {SAVE}")

    # ── winner pick: lowest late-window eta_std std (most settled) ──
    best = min(results, key=lambda k: np.std(results[k]["eta_std"][-16:]))
    print("\n────────────────────────────────────────────────────")
    print(f"RECOMMENDED nu_bi = {best}  (most settled late-window SSH variance)")


if __name__ == "__main__":
    main()
