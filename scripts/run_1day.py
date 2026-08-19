"""
Smoke test: integrate the ocean model for one day with default config.

Run: python scripts/run_1day.py
"""
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from config import DEFAULT_CONFIG
from grid import make_grid
from state import initialize_state
from integrator import integrate


def main():
    print("=" * 60)
    print("Ocean Solver — 1-day Smoke Test")
    print("=" * 60)

    cfg = DEFAULT_CONFIG
    print("Building grid...")
    grid = make_grid(cfg.grid, cfg.bathymetry_file)
    print(f"  Grid: {grid.nx} x {grid.ny} x {grid.nz}")
    print(f"  dx = {grid.dx:.1f} m, dy = {grid.dy:.1f} m")
    print(f"  f0 = {grid.f0:.5e} /s")
    print()

    print("Initializing state...")
    state = initialize_state(grid, cfg.physics)
    print(f"  T_ref = {cfg.physics.T_ref} C, S_ref = {cfg.physics.S_ref} psu")
    print()

    print(f"Integrating for {cfg.time.t_total/3600:.0f} h...")
    n_steps = int(round(cfg.time.t_total / cfg.time.dt))
    print(f"  dt = {cfg.time.dt} s, n_steps = {n_steps}")

    wall_t0 = time.time()

    def callback(t, s):
        hours = t / 3600.0
        u_surf = np.mean(s.u[:, :, 0])
        v_surf = np.mean(s.v[:, :, 0])
        T_surf = np.mean(s.T[:, :, 0])
        print(f"  t = {hours:6.2f} h  |  u_surf = {u_surf: .5e}  v_surf = {v_surf: .5e}  T_surf = {T_surf:.5f}")

    integrate(state, grid, cfg.physics, cfg.time, callback=callback)

    wall_t1 = time.time()
    elapsed = wall_t1 - wall_t0
    print()
    print(f"Wall time: {elapsed:.2f} s")
    print(f"Seconds per step: {elapsed / n_steps:.4f}")
    print()

    # Final statistics
    print("Final state statistics:")
    print(f"  max |u|: {np.max(np.abs(state.u)):.5e} m/s")
    print(f"  max |v|: {np.max(np.abs(state.v)):.5e} m/s")
    print(f"  mean T surface: {np.mean(state.T[:, :, 0]):.5f} C")
    print(f"  min/max T: {np.min(state.T):.5f} / {np.max(state.T):.5f} C")
    print(f"  mean S surface: {np.mean(state.S[:, :, 0]):.5f} psu")
    print(f"  min/max rho: {np.min(state.rho):.5f} / {np.max(state.rho):.5f} kg/m^3")
    print(f"  max |w|: {np.max(np.abs(state.w)):.5e} m/s")
    print()
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
