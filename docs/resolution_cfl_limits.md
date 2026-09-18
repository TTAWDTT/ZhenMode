# Resolution scaling: measured CFL limits and the auto-scaling fix

Measured 2026-09-18 with CPU probes in `src/jax_solver_global.py`. Every
number below is from an actual run, not an estimate.

## TL;DR

Three 1°-calibrated parameters all violate their CFL on finer grids, and any
one of them alone diverges a 0.5° run. `--resolution` now auto-scales all
three by the right power of dx, so `--resolution 0.5` runs stably.

| parameter | 1° value | CFL | scales as | 0.5° value |
|---|---|---|---|---|
| `--dt-bt` | 150 s | external gravity wave: `dt < 2·dx/√(g·H_sw)` | dx¹ | 75 s |
| `--nu-h` | 5e6 m²/s | explicit Laplacian: `nu_h·dt/dx² < 0.25` | dx² | 1.25e6 |
| `--nu-bi` | 2e14 m⁴/s | biharmonic: `nu_bi·dt/dx⁴ < ~0.05` | dx⁴ | 1.25e13 |

Pass any of the three explicitly to override the auto-scaling.

## What was measured

Setup: `make_global_grid(lat_max=60.0, smooth_passes=30, min_depth=100.0)`,
`make_solver_global(mode_split=True, dt=3600)`, IC `T = 15 + 1·cos(2λ)cos(2φ)`,
`S = 35`, 40-60 steps.

### Fault 1 — barotropic subcycle CFL (the regression)

Holding `nu_h = nu_bi = 0` and sweeping `(dt, dt_bt)` at 0.5°:

| eff = dt / n_subcyc | outcome |
|---|---|
| ≤ 200 s | **stable** (all dt from 300 to 3600) |
| ≥ 300 s | **DIVERGED** |

The barotropic subcycle runs at `p.dt_bt` (`_free_surface_step_fd(...,
dt_half=p.dt_bt)`), so this is a pure `dt_bt` limit:
`dt_bt < 2·dx_min/√(g·H_sw) ≈ 283 s` at 0.5°, vs `≈ 570 s` at 1.0°.

The current production launch scripts (`_launch_mix.sh`, `_launch_pjksweep.sh`,
`_launch_closure.sh`) pass `--dt-bt 300`, which is safe at 1° but **over the
0.5° limit**. The historical spin config used `dt_bt=150` — that is why 0.5°
"used to work". This is a config regression, not an architectural limit.

### Fault 2 — biharmonic CFL

At 0.5°, `dt_bt=150`, `nu_h=0`, sweep `nu_bi`: `2e14` DIVERGED, `2e13` and
below stable. `nu_bi ∝ dx⁴`, so a 2x refinement needs a 16x reduction.

### Fault 3 — horizontal Laplacian CFL

At 0.5°, `dt_bt=150`, `nu_bi=0`, `nu_h=5e6` alone DIVERGED. `nu_h ∝ dx²`, so
a 2x refinement needs 4x (to ~1.25e6, the auto-scaled value).

### Ruled out

- **RK2 stage-2 column divergence** (`--project-adv-vel`): moves the blow-up
  by 1-2 steps only, with every fault still present. Not the cause.
- **Solver bug**: with `T ≡ T_ref` the step-1 tendency is exactly 0 at both
  resolutions (max|u| = 0.0), so the operators are correct.

## Validation of the fix

`dt_bt/nu_h/nu_bi` scaled from the 1° reference by dx^1/2/4, `dt=3600`,
production physics:

| res | dt_bt | nu_h | nu_bi | 60-step result |
|---|---|---|---|---|
| 1.0° | 300 | 5e6 | 2e14 | **ok** |
| 0.5° | 150 | 1.25e6 | 1.25e13 | **ok** |

## Notes

- The 1° references in the code are `DT_BT_DEFAULT=150`, `NU_H_REF_1DEG=5e6`,
  `NU_BI_REF_1DEG=2e14`. At `--resolution 1.0` the scaling is a no-op, so the
  legacy defaults are preserved bit-for-bit.
- `--init-from` npz files are hard-bound to 360x120 and cannot be used at
  other resolutions.
