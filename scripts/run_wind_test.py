"""
Wind-driven experiment: apply constant zonal wind stress for 1 day.

Compares rest state vs wind-forced state to demonstrate the model produces
physically meaningful ocean response (Ekman transport, Coriolis deflection,
surface mixing).

Run: python scripts/run_wind_test.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from config import Config, GridConfig, PhysicsConfig, TimeConfig, DEFAULT_CONFIG
from grid import make_grid
from state import initialize_state
from integrator import integrate


def run_experiment(label, physics_override=None, time_override=None):
    """Run a 1-day integration with optional physics overrides."""
    cfg = DEFAULT_CONFIG

    # Apply overrides
    if physics_override:
        physics = physics_override
    else:
        physics = cfg.physics

    if time_override:
        time_cfg = time_override
    else:
        time_cfg = cfg.time

    print(f"\n{'='*60}")
    print(f"Experiment: {label}")
    print(f"{'='*60}")
    print(f"  tau_x={physics.tau_x:.3f} N/m^2, tau_y={physics.tau_y:.3f} N/m^2")
    print(f"  Q_heat={physics.Q_heat:.1f} W/m^2")

    print("Building grid...")
    grid = make_grid(cfg.grid, cfg.bathymetry_file)
    print(f"  Grid: {grid.nx} x {grid.ny} x {grid.nz}")
    print(f"  dx = {grid.dx:.1f} m, dy = {grid.dy:.1f} m")
    print(f"  f0 = {grid.f0:.5e} /s (lat={cfg.grid.lat_center}N)")
    print(f"  Ocean points: {grid.ocean_mask.sum()} / {grid.nx*grid.ny}")

    print("Initializing state...")
    state = initialize_state(grid, physics)
    print(f"  T_ref = {physics.T_ref} C, S_ref = {physics.S_ref} psu")

    n_steps = int(round(time_cfg.t_total / time_cfg.dt))
    print(f"Integrating {time_cfg.t_total/3600:.0f} h ({n_steps} steps, dt={time_cfg.dt}s)...")

    # Collect time series
    times = []
    series = {'u_surf': [], 'v_surf': [], 'T_surf': [], 'w_max': [], 'ke': []}

    def callback(t, s):
        hours = t / 3600.0
        u_surf = np.mean(s.u[:, :, 0])
        v_surf = np.mean(s.v[:, :, 0])
        T_surf = np.mean(s.T[:, :, 0])
        w_max = np.max(np.abs(s.w)) if s.w is not None else 0.0
        ke = 0.5 * np.mean(s.u**2 + s.v**2)  # kinetic energy per unit mass

        times.append(hours)
        series['u_surf'].append(u_surf)
        series['v_surf'].append(v_surf)
        series['T_surf'].append(T_surf)
        series['w_max'].append(w_max)
        series['ke'].append(ke)

        if len(times) % 6 == 0:  # print every 6 hours
            print(f"  t={hours:6.1f}h  u={u_surf:+.5e}  v={v_surf:+.5e}  "
                  f"T={T_surf:.5f}  |w|max={w_max:.3e}  KE={ke:.3e}")

    t0 = time.time()
    integrate(state, grid, physics, time_cfg, callback=callback)
    wall = time.time() - t0

    # Final statistics
    print(f"\n  Wall time: {wall:.1f}s ({wall/n_steps:.4f}s/step)")
    print(f"\n  --- Final State Statistics ---")
    print(f"  max |u|:      {np.max(np.abs(state.u)):.5e} m/s")
    print(f"  max |v|:      {np.max(np.abs(state.v)):.5e} m/s")
    print(f"  max |w|:      {np.max(np.abs(state.w)):.5e} m/s")
    print(f"  mean u_surf:  {np.mean(state.u[:,:,0]):.5e} m/s")
    print(f"  mean v_surf:  {np.mean(state.v[:,:,0]):.5e} m/s")
    print(f"  mean u_bot:   {np.mean(state.u[:,:,-1]):.5e} m/s")
    print(f"  mean v_bot:   {np.mean(state.v[:,:,-1]):.5e} m/s")
    print(f"  T range:      {np.min(state.T):.5f} / {np.max(state.T):.5f} C")
    print(f"  S range:      {np.min(state.S):.5f} / {np.max(state.S):.5f} psu")
    print(f"  rho range:    {np.min(state.rho):.5f} / {np.max(state.rho):.5f} kg/m^3")
    print(f"  KE (mean):    {0.5*np.mean(state.u**2 + state.v**2):.5e} m^2/s^2")

    # Vertical profile (domain-averaged)
    print(f"\n  --- Vertical Profile (domain mean) ---")
    print(f"  {'z(m)':>8} {'u(m/s)':>14} {'v(m/s)':>14} {'T(C)':>10} {'rho':>10}")
    for k in range(grid.nz):
        z_k = grid.z[k]
        u_k = np.mean(state.u[:,:,k])
        v_k = np.mean(state.v[:,:,k])
        T_k = np.mean(state.T[:,:,k])
        r_k = np.mean(state.rho[:,:,k])
        print(f"  {z_k:8.0f} {u_k:+14.5e} {v_k:+14.5e} {T_k:10.5f} {r_k:10.5f}")

    return {
        'label': label,
        'times': np.array(times),
        'series': {k: np.array(v) for k, v in series.items()},
        'final_state': state,
        'grid': grid,
    }


def main():
    print("="*60)
    print("Ocean Solver — Wind-Driven Experiment")
    print("Comparing: rest state vs zonal wind forcing")
    print("="*60)

    # --- Experiment 1: Rest state (no forcing) ---
    # Use dt=150s for stability (dt=600s diverges with wind forcing)
    stable_time = TimeConfig(dt=150.0, dt_output=3600.0, t_total=12*3600.0)

    physics_rest = PhysicsConfig()  # defaults: all forcing = 0

    # --- Experiment 2: Constant westward wind stress (typical Trades) ---
    # tau_x = -0.1 N/m^2 ~ moderate trade wind stress
    physics_wind = PhysicsConfig(
        tau_x=-0.1,    # N/m^2, westward (easterly wind)
        tau_y=0.0,
        Q_heat=0.0,
    )

    # --- Experiment 3: Westward wind + surface cooling ---
    physics_wind_cool = PhysicsConfig(
        tau_x=-0.1,
        tau_y=0.0,
        Q_heat=-100.0,  # W/m^2, cooling
    )

    results = []
    results.append(run_experiment("Rest (no forcing)", physics_override=physics_rest, time_override=stable_time))
    results.append(run_experiment("Westward wind (tau_x=-0.1)", physics_override=physics_wind, time_override=stable_time))
    results.append(run_experiment("Wind + cooling", physics_override=physics_wind_cool, time_override=stable_time))

    # --- Comparison table ---
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    header = f"{'Metric':<25} {'Rest':>16} {'Wind':>16} {'Wind+Cool':>16}"
    print(header)
    print("-" * len(header))

    for key, fmt in [
        ('u_surf', '{:+.5e}'),
        ('v_surf', '{:+.5e}'),
        ('T_surf', '{:.5f}'),
        ('w_max',  '{:.3e}'),
        ('ke',     '{:.3e}'),
    ]:
        vals = []
        for r in results:
            v = r['series'][key][-1]  # final value
            vals.append(fmt.format(v))
        print(f"  {key:<23} {vals[0]:>16} {vals[1]:>16} {vals[2]:>16}")

    # Also print max|u|, max|w|, T range
    print()
    for label, func in [
        ("max|u| (m/s)",     lambda s: np.max(np.abs(s.u))),
        ("max|v| (m/s)",     lambda s: np.max(np.abs(s.v))),
        ("max|w| (m/s)",     lambda s: np.max(np.abs(s.w))),
        ("T min (C)",        lambda s: np.min(s.T)),
        ("T max (C)",        lambda s: np.max(s.T)),
        ("rho min (kg/m^3)", lambda s: np.min(s.rho)),
        ("rho max (kg/m^3)", lambda s: np.max(s.rho)),
    ]:
        vals = [f"{func(r['final_state']):.5e}" for r in results]
        print(f"  {label:<23} {vals[0]:>16} {vals[1]:>16} {vals[2]:>16}")

    print("\nDone.")


if __name__ == "__main__":
    main()
