"""
Localize the source of the ~day-84 velocity explosion in the 90-day
real-wind verification run.

The 30-day run (step b) PASSED, but the 90-day run blew up at day ~80-84:
max|u| climbed 4.2 -> 10.85 -> -inf. Temperature stayed healthy (25-29C)
until the velocity explosion dragged it down.  This diagnostic re-runs the
SAME config forward and records, at every checkpoint:

  - max|u| value and its EXACT (i, j, k) grid location (land-masked)
  - whether that cell is ocean or land
  - a per-z-level kinetic-energy profile  sum(u^2 + v^2) over each layer
  - the horizontal KE-rich column (i, j) to see spatial localization
  - dominant z-levels by KE share

It aborts early (default once max|u| > 4 m/s, far before the -inf blowup)
so the growing location is caught while the field is still finite.

Config is identical to the failing run:
  --real-wind, --restore-days 5, convective kappa_conv default.

Usage:
  python src/diag_localize_blowup.py [--days 90] [--abort-u 4.0]
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from config import DEFAULT_CONFIG
from grid import make_grid
from jax_solver import make_solver
from woa_data import get_initial_fields
from forcing import heat_flux_meridional
from wind_reanalysis import real_wind_forcing

DT = 300.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=90.0)
    ap.add_argument("--restore-days", type=float, default=5.0)
    ap.add_argument("--abort-u", type=float, default=4.0,
                    help="abort once max|u| exceeds this (m/s)")
    args = ap.parse_args()

    n_steps = int(round(args.days * 86400.0 / DT))

    grid = make_grid(DEFAULT_CONFIG.grid, DEFAULT_CONFIG.bathymetry_file)
    physics = DEFAULT_CONFIG.physics
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    ocean_mask = jnp.array(grid.ocean_mask)          # (nx, ny) bool
    ocean_3d = ocean_mask[:, :, None]                # (nx, ny, 1) broadcast
    lon, lat = grid.lon, grid.lat

    print("=" * 64)
    print(f"LOCALIZE BLOWUP - real wind + restore{args.restore_days:g}d "
          f"({args.days:.0f} days)")
    print("=" * 64)
    print(f"Grid: {nx}x{ny}x{nz}  dt={DT}s  steps={n_steps}")
    print(f"Landmask applied to KE diagnostics. abort when max|u|>{args.abort_u}")
    print()

    # ── forcing (identical to failing run) ──
    tau_x, tau_y = real_wind_forcing(grid=grid)
    Q_heat = heat_flux_meridional(grid, Q0=50.0)
    forcing = (tau_x, tau_y, Q_heat)

    T_init, S_init = get_initial_fields(grid)
    T_sst = T_init[:, :, 0]
    step_fn, init_state, diag_fn = make_solver(
        grid, physics, DT, forcing=forcing,
        T_sst=T_sst, tau_restore_days=args.restore_days)
    state = init_state(T_init=T_init, S_init=S_init)

    t0 = time.perf_counter()
    state = step_fn(state)
    jax.block_until_ready(state.u)
    print(f"JIT compile + first step: {time.perf_counter()-t0:.2f}s\n")

    check_interval = max(100, n_steps // 60)
    aborted = False

    # Last few KE horizontal maps retained for spatial inspection.
    ke_maps = []          # list of (day, KE_horizontal_2d map)
    last_zprof = None     # most recent per-z KE profile

    for i in range(2, n_steps + 1):
        state = step_fn(state)
        if i % check_interval != 0 and i != n_steps:
            continue
        jax.block_until_ready(state.u)

        u = state.u                              # (nx,ny,nz)
        v = state.v
        u_flat = jnp.abs(u) * ocean_3d
        v_flat = v * ocean_3d

        # global max|u| over ocean and its (i,j,k)
        umax = float(jnp.max(u_flat))
        flat_idx = jnp.argmax(u_flat.reshape(-1))
        ii, jj, kk = np.unravel_index(int(flat_idx), (nx, ny, nz))
        is_land = bool(~ocean_mask[ii, jj])

        # per-z KE profile
        ke_layer = jnp.sum(u_flat ** 2 + v_flat ** 2, axis=(0, 1))   # (nz,)
        ke_total = float(jnp.sum(ke_layer))
        ke_frac = np.asarray(ke_layer) / (ke_total + 1e-30)
        last_zprof = ke_frac

        # KE-rich horizontal column (2D KE argmax)
        ke_2d = jnp.sum(u_flat ** 2 + v_flat ** 2, axis=2)           # (nx,ny)
        ke_2d_masked = ke_2d * ocean_mask
        ke2d_arg = jnp.argmax(ke_2d_masked.reshape(-1))
        cx, cy = np.unravel_index(int(ke2d_arg), (nx, ny))

        # top-layer (k=0) share of KE
        top_ke_share = float(ke_layer[0]) / (ke_total + 1e-30)

        # dominant z-levels (top 3 by KE share)
        top3_z = np.argsort(ke_frac)[::-1][:3]
        zprof = " ".join(f"k{k}:{ke_frac[k]:.2f}" for k in top3_z)

        day = i * DT / 86400.0
        print(f"day {day:6.2f}  max|u|={umax:7.3f}  @(i={ii},j={jj},z={kk}) "
              f"land={is_land}  KEtop_share={top_ke_share:.3f}  "
              f"KEcol=(i={cx},j={cy}) lon={lon[cx]:.1f}E lat={lat[cy]:.1f}N  "
              f"topZ[{zprof}]")

        ke_maps.append((day, np.asarray(ke_2d_masked)))
        if len(ke_maps) > 6:
            ke_maps.pop(0)

        if umax > args.abort_u:
            print(f"\n>>> ABORT: max|u|={umax:.3f} > {args.abort_u} "
                  f"(day {day:.1f})")
            aborted = True
            break

    # ── Spatial report ──
    print("\n" + "=" * 64)
    print("KE horizontal map at last checkpoints:")
    print("=" * 64)
    for day, kmap in ke_maps:
        km = np.asarray(kmap) * np.asarray(ocean_mask)
        argmax_ = np.unravel_index(np.argmax(km), (nx, ny))
        print(f"  day {day:6.2f}: KE max={float(km.max()):.3e} at "
              f"(i={argmax_[0]},j={argmax_[1]}) "
              f"lon={lon[argmax_[0]]:.1f}E lat={lat[argmax_[1]]:.1f}N")
        # fraction of KE in top-5% cells (spatial concentration)
        flat = np.sort(km.ravel())
        n = flat.size
        thresh = flat[int(0.95 * n)]
        frac = km[km >= thresh].sum() / (km.sum() + 1e-30)
        print(f"           top-5% cells hold {frac*100:.1f}% of KE")

    print("\n" + "=" * 64)
    print("Per-z KE profile at last checkpoint:")
    if last_zprof is not None:
        zl = [f"k{k}:{float(last_zprof[k]):.3f}" for k in range(nz)]
        print("  " + "  ".join(zl))
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
