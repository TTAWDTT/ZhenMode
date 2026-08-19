"""
Diagnostic: find when/why the wind-forced simulation diverges.
Runs with frequent output to pinpoint the onset of instability.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from config import DEFAULT_CONFIG, PhysicsConfig, TimeConfig
from grid import make_grid
from state import initialize_state
from integrator import step, _update_diagnostics

cfg = DEFAULT_CONFIG
grid = make_grid(cfg.grid, cfg.bathymetry_file)
physics = PhysicsConfig(tau_x=-0.1, tau_y=0.0, Q_heat=0.0)

# Try different timesteps
for dt_test in [600, 300, 150, 60]:
    time_cfg = TimeConfig(dt=dt_test, dt_output=dt_test, t_total=12*3600)
    n_steps = int(round(time_cfg.t_total / dt_test))

    state = initialize_state(grid, physics)
    _update_diagnostics(state, grid, physics)

    print(f"\n--- dt={dt_test}s, {n_steps} steps, 12h ---")
    diverged = False
    for n in range(n_steps):
        step(state, grid, physics, time_cfg, dt_test)

        if (n + 1) % max(1, n_steps // 12) == 0 or n < 5:
            u_max = np.max(np.abs(state.u))
            v_max = np.max(np.abs(state.v))
            w_max = np.max(np.abs(state.w))
            ke = 0.5 * np.mean(state.u**2 + state.v**2)
            t_h = (n + 1) * dt_test / 3600.0
            if np.isnan(u_max) or u_max > 1e3:
                print(f"  t={t_h:5.1f}h  DIVERGED  max|u|={u_max:.3e}")
                diverged = True
                break
            print(f"  t={t_h:5.1f}h  max|u|={u_max:.4e}  max|v|={v_max:.4e}  max|w|={w_max:.4e}  KE={ke:.4e}")

    if not diverged:
        print(f"  -> STABLE for 12h, final max|u|={np.max(np.abs(state.u)):.4e}")
