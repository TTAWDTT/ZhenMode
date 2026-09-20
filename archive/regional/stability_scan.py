"""
CFL / Stability-domain scan for the JAX ocean solver.

Goal: find the maximum stable time step under REAL ocean stratification
(WOA2023 climatology), where baroclinic (internal) wave modes drive the
explicit forward-backward RK2 to its stability limit.

Approach:
  For each dt in a sweep (e.g. 10..1000 s):
    - initialize with real WOA T/S (strong vertical stratification,
      ~22 C temperature contrast -> strong baroclinic PGF)
    - no surface forcing (isolates the baroclinic adjustment instability)
    - run a fixed number of steps, tracking max|u|, max|T| growth
    - classify as STABLE / DEGRADED / BLOWUP and record the blowup step

Output: prints a table + writes results/stability_scan.png
(log-linear plot of max|u| vs simulated time for each dt).

Usage:
  python src/stability_scan.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import DEFAULT_CONFIG, PhysicsConfig
from grid import make_grid
from jax_solver import make_solver, JaxState
from woa_data import get_initial_fields


# ── Scan configuration ──
DT_SWEEP = [10.0, 20.0, 30.0, 40.0, 45.0, 50.0, 60.0, 90.0,
            120.0, 150.0, 200.0, 300.0, 450.0, 600.0, 900.0]
SIM_TIME = 7200.0           # 2 h simulated (enough for blowup to appear)
MAX_WALL_PER_DT = 90.0      # hard wall-clock guard per dt config (s)
OUT_PATH = 'results/stability_scan.png'

# 3-tier classification thresholds by physical max|u| (m/s):
#   STABLE  <= 10 m/s   (realistic ocean speeds)
#   DEGRADED<= 1e4 m/s  (unphysical but finite; instability onset)
#   BLOWUP  >  1e4 m/s or NaN
MU_STABLE = 10.0
MU_DEGRADED = 1.0e4
MU_BLOWUP = 1.0e6


def classify(mu):
    """3-tier classification by max|u|."""
    if jnp.isnan(mu):
        return "BLOWUP"
    if mu > MU_BLOWUP:
        return "BLOWUP"
    if mu > MU_DEGRADED:
        return "DEGRADED"
    if mu > MU_STABLE:
        return "DEGRADED"
    return "STABLE"


def load_real_init(grid):
    """Real WOA2023 initial T/S on the solver grid (no forcing)."""
    print("Loading WOA2023 climatology...")
    t0 = time.perf_counter()
    T_init, S_init = get_initial_fields(grid)
    print(f"  T_init range: [{T_init.min():.2f}, {T_init.max():.2f}] C "
          f"({time.perf_counter()-t0:.1f}s)")
    print(f"  S_init range: [{S_init.min():.2f}, {S_init.max():.2f}] PSU")
    return T_init, S_init


def run_dt(grid, physics, dt, T_init, S_init, sim_time):
    """Run one dt config; return (tag: str, blowup_step: int|None,
    maxu_at_blowup: float, times[], maxu[])."""
    n_steps = max(1, int(sim_time / dt))

    step_fn, init_state, _ = make_solver(grid, physics, dt, forcing=None)
    state = init_state(T_init=T_init, S_init=S_init)

    # JIT compile + first step
    state = step_fn(state)
    jax.block_until_ready(state.u)

    # Track growth of max|u| (internal wave energy proxy)
    times = [0.0]
    maxu = [float(jnp.max(jnp.abs(state.u)))] if np.any(state.u) else [0.0]

    tag = "STABLE"
    blowup_step = None
    maxu_at_blowup = float('nan')

    t_start = time.perf_counter()
    for i in range(1, n_steps + 1):
        state = step_fn(state)
        t = i * dt

        # sample every few steps to keep array small
        if i % max(1, n_steps // 40) == 0 or i == n_steps:
            jax.block_until_ready(state.u)
            mu = float(jnp.max(jnp.abs(state.u)))
            times.append(t)
            maxu.append(mu)

            if classify(mu) == "BLOWUP":
                tag = "BLOWUP"
                blowup_step = i
                maxu_at_blowup = mu
                break

        # wall-clock guard
        if time.perf_counter() - t_start > MAX_WALL_PER_DT:
            print(f"  [dt={dt:.0f}s] wall-clock guard hit at step {i}")
            break

    # if no BLOWUP triggered, tag by end value (DEGRADED vs STABLE)
    if tag != "BLOWUP":
        end_mu = max(maxu) if maxu else 0.0
        if end_mu > 10.0:
            tag = "DEGRADED"

    return tag, blowup_step, maxu_at_blowup, times, maxu


def main():
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics

    print("=" * 72)
    print("  CFL / Stability-Domain Scan (real WOA stratification)")
    print("=" * 72)
    print(f"  Grid: {grid.nx}x{grid.ny}x{grid.nz}")
    print(f"  dx={grid.dx:.0f}m  dy={grid.dy:.0f}m  f0={grid.f0:.2e}")
    print(f"  Simulated time: {SIM_TIME/3600:.1f} h")
    print(f"  Forcing: none (isolates baroclinic adjustment)")
    print()

    # Analytical CFL estimate for the external (surface gravity) mode
    H_sw = float(np.sum(grid.dz))
    c_ext = np.sqrt(9.81 * H_sw)

    print(f"  H_sw={H_sw:.0f}m -> external wave speed "
          f"c_ext=sqrt(gH)={np.sqrt(9.81*H_sw):.0f} m/s")
    print(f"  External-mode CFL: dx/c_ext = {grid.dx/np.sqrt(9.81*H_sw):.0f} s")
    print()

    T_init, S_init = load_real_init(grid)

    print()
    print(f"  {'dt(s)':>7} {'n_steps':>8} {'result':>10} {'blowup@step':>12} "
          f"{'max|u|@end/blow':>16}")
    print(f"  {'-'*60}")

    results = []
    for dt in DT_SWEEP:
        tag, blowup_step, maxu_blow, times, maxu = run_dt(
            grid, physics, dt, T_init, S_init, SIM_TIME)

        blow_txt = f"~{blowup_step}" if blowup_step else "-"
        end_maxu = max(maxu)
        results.append({
            'dt': dt, 'tag': tag, 'blowup_step': blowup_step,
            'times': times, 'maxu': maxu,
            'maxu_end': end_maxu,
        })
        print(f"  {dt:>7.0f} {int(SIM_TIME/dt):>8} {tag:>10} {blow_txt:>12} "
              f"{end_maxu:>16.3e}")
        sys.stdout.flush()

    # Find the largest physically-stable dt (STABLE only)
    stable_dts = [r['dt'] for r in results if r['tag'] == 'STABLE']
    largest_stable = max(stable_dts) if stable_dts else None
    first_nonstable = next(
        (r['dt'] for r in results if r['tag'] != 'STABLE'), None)
    print()
    print(f"  Largest STABLE dt   : {largest_stable if largest_stable else 'NONE'}")
    print(f"  First non-STABLE dt : {first_nonstable if first_nonstable else '>max swept'}")

    # ── Plot ──
    plot_stability(results, OUT_PATH)
    print(f"\n  Saved plot: {OUT_PATH}")


def plot_stability(results, out_path):
    """Log-linear plot of max|u| vs simulated time for each dt."""
    colors = {'STABLE': '#55A868', 'DEGRADED': '#E2B93B', 'BLOWUP': '#C44E52'}
    styles = {'STABLE': '-o', 'DEGRADED': '--s', 'BLOWUP': '--x'}
    fig, ax = plt.subplots(figsize=(10, 7))
    for r in results:
        c = colors.get(r['tag'], '#888888')
        s = styles.get(r['tag'], '-o')
        ax.loglog(r['times'], np.maximum(r['maxu'], 1e-12),
                  s, color=c, lw=1.8, ms=4,
                  label=f"dt={r['dt']:.0f}s ({r['tag']})")
    ax.axhline(MU_STABLE, color='green', ls=':', lw=1, alpha=0.6)
    ax.axhline(MU_DEGRADED, color='orange', ls=':', lw=1, alpha=0.5)
    ax.axhline(MU_BLOWUP, color='red', ls=':', lw=1, alpha=0.5)
    ax.text(0.99, 0.03, 'STABLE | DEGRADED | BLOWUP thresholds',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, alpha=0.6)
    ax.set_xlabel('Simulated time (s)')
    ax.set_ylabel('max |u| (m/s)')
    ax.set_title('Stability-Domain Scan under WOA Stratification\n'
                 '(no forcing; log-log growth of max|u|)')
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == "__main__":
    main()
