"""
JAX vs PyTorch — numerical equivalence + performance benchmark.

Runs both solvers from identical WOA-seeded initial conditions for the
same number of steps, then compares u, v, T, S, eta field-by-field and
times each backend.

Run:  python src/bench_jax_torch.py  (> logs/bench_jax_torch.log 2>&1)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import DEFAULT_CONFIG
from grid import make_grid
from woa_data import get_initial_fields
from forcing import wind_stress_gyre, heat_flux_meridional


def main():
    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    dt = 300.0
    N_STEPS = 8

    tau_x, tau_y = wind_stress_gyre(grid, tau0=0.1)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)
    forcing = (tau_x, tau_y, Q_heat)

    T_init, S_init = get_initial_fields(grid)
    print(f"Grid {grid.nx}x{grid.ny}x{grid.nz}, dt={dt}s, steps={N_STEPS}")
    print(f"physics: nu_h={physics.nu_h} nu_bi={physics.nu_bi} "
          f"kappa_bi={physics.kappa_bi}")

    # ── JAX ──
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax_solver as js
    jax_step, jax_init, jax_diag = js.make_solver(grid, physics, dt,
                                                  forcing=forcing)
    jst = jax_init(T_init=T_init, S_init=S_init)
    t0 = time.perf_counter()
    jst = jax_step(jst)  # compile warmup
    jax.block_until_ready(jst.u)
    t_comp = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(N_STEPS - 1):
        jst = jax_step(jst)
    jax.block_until_ready(jst.u)
    t_jax = (time.perf_counter() - t0) / (N_STEPS - 1) * 1000

    # ── PyTorch ──
    import torch
    import torch_solver as ts
    pt_step, pt_init, pt_diag = ts.make_solver(grid, physics, dt,
                                               forcing=forcing, compile=False)
    pst = pt_init(T_init=T_init, S_init=S_init)
    t0 = time.perf_counter()
    pst = pt_step(pst)   # first step (torch eager, includes any init)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(N_STEPS - 1):
        pst = pt_step(pst)
    t_pt = (time.perf_counter() - t0) / (N_STEPS - 1) * 1000

    # ── Compare after N_STEPS (both advanced the same number of full steps)
    def _cmp(name, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        diff = np.abs(a - b)
        denom = np.maximum(np.abs(a), np.abs(b))
        # avoid div-by-zero on zero fields
        denom = np.where(denom == 0, 1.0, denom)
        rel = diff / denom
        print(f"  {name:>5}: max_abs={diff.max():.6e}  "
              f"mean_rel={rel.mean():.3e}  "
              f"max_rel={rel.max():.3e}")

    print(f"\nResults after {N_STEPS} steps:")
    _cmp('u', jst.u, pst.u)
    _cmp('v', jst.v, pst.v)
    _cmp('T', jst.T, pst.T)
    _cmp('S', jst.S, pst.S)
    _cmp('eta', jst.eta, pst.eta)

    # Global T/|u| envelope agreement
    print(f"\n  JAX: T=[{float(np.min(np.asarray(jst.T))):.4f}, "
          f"{float(np.max(np.asarray(jst.T))):.4f}]  "
          f"|u|max={float(np.max(np.abs(np.asarray(jst.u)))):.4f}")
    print(f"  PT : T=[{float(np.min(np.asarray(pst.T))):.4f}, "
          f"{float(np.max(np.asarray(pst.T))):.4f}]  "
          f"|u|max={float(np.max(np.abs(np.asarray(pst.u)))):.4f}")

    # ── Timing summary ──
    print(f"\nTiming (CPU, {grid.nx}x{grid.ny}x{grid.nz}):")
    print(f"  JAX : {t_jax:8.2f} ms/step  (compile {t_comp:.1f}s)")
    print(f"  PT  : {t_pt:8.2f} ms/step  (first step {t_first:.2f}s)")
    if t_pt > 0:
        print(f"  ratio JAX/PT = {t_jax / t_pt:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
