# Resolution scaling: measured CFL limits and the auto-scaling fix

Measured 2026-09-18 with CPU probes in `src/ocean_solver/numerics/stability.py`. Every
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
  other resolutions. Omitting `--init-from` is resolution-agnostic: the WOA
  interpolation in `get_initial_fields(grid)` follows `grid.lon/lat/z`.

## Cluster validation (2026-09-18)

Three 30-day runs on the L20X cluster, production physics
(`--dt 3600 --mode-split --use-scan --dtype float32`, kappa_gm=1000,
kappa_v=1e-5, kappa_conv=0.05), each with `--resolution` auto-scaling:

| res | grid | 30-day result | max\|u\| | wall |
|---|---|---|---|---|
| 1.0° | 360x120x14 | **PASS** | 0.91 | 0.9 min |
| 0.5° | 720x240x14 | **PASS** | 1.25 | 3.7 min |
| 0.4° | 900x300x14 | FAIL_DRIFT | 1.39 | 4.9 min |

1.0° and 0.5° integrate cleanly: max|u| and max|eta| both settle, max|T|
*falls* (29.7 -> 28.0 / 28.5) as the seasonal cycle spins up.

0.4° is **numerically stable but drifts in one spot**: max|T| climbs
29.7 -> 32.6 while max|u| and eta converge, and the excess is a single
isolated point (122.6°E 10.2°N, the Sulu Sea). The same point starts at the
same 28.4 C at all three resolutions and *cools* at 1.0°/0.5° (to 26.5 /
27.7) but *warms* 2.6 C at 0.4°. That is a broken-coastline artifact — at
0.4° the archipelago resolves into one-cell straits and isolated shallow
columns — not a CFL violation. The auto-scaling fix does its job at 0.4°
(the run no longer blows up); the remaining drift is a separate grid-quality
issue at the fine end.

## Resolution sweep: 1.5° -> 0.3°, one run per resolution (2026-09-18)

Same production config, 30 days each, eight resolutions fanned across the
cluster's eight GPUs. 0.1° is the ETOPO source spacing, so the legal ladder
is its integer multiples (0.25° floors to the same step as 0.2°).

| res | nx x ny | dt_bt | nu_h | nu_bi | max\|u\| | max\|T\| | verdict |
|---|---|---|---|---|---|---|---|
| 1.5° | 240 x 80 | 225 s | 1.125e7 | 1.012e15 | 0.752 | 27.94 | PASS |
| 1.2° | 300 x 100 | 180 s | 7.2e6 | 4.147e14 | 0.836 | 28.08 | PASS |
| 1.0° | 360 x 120 | 150 s | 5e6 | 2e14 | 0.912 | 28.02 | PASS |
| 0.8° | 450 x 151 | 120 s | 3.2e6 | 8.192e13 | 1.007 | 28.06 | PASS |
| 0.6° | 600 x 200 | 90 s | 1.8e6 | 2.592e13 | 1.147 | **32.24** | FAIL_DRIFT |
| 0.5° | 720 x 240 | 75 s | 1.25e6 | 1.25e13 | 1.248 | 28.48 | PASS |
| 0.4° | 900 x 300 | 60 s | 8e5 | 5.12e12 | 1.394 | **32.65** | FAIL_DRIFT |
| 0.3° | 1200 x 400 | 45 s | 4.5e5 | 1.62e12 | 1.920 | **52.61** | FAIL_BLOWUP (d20) |

max|u| rises monotonically with resolution (0.75 -> 1.92) exactly as the
auto-scaled dt_bt shrinks — the CFL fix holds at every rung, and no run
diverges on the momentum/eta side. (0.3° NaN'd only by day 20, after its
tracer field had already run away; max|u| was still 1.9 and eta 1.8.)

The FAIL_DRIFT/FAIL_BLOWUP rows are **all the same Sulu Sea point**
(122-123°E, 8-11°N), and the verdict is not monotone in resolution. The
coastline pattern there decides it:

| res | pattern at the hotspot | SST d0 -> d30 |
|---|---|---|
| 0.8° | connected shallow sea | 28.4 -> 27.1 (cools) |
| 0.5° | peninsula / bay | 28.4 -> 27.7 (cools) |
| 0.6° | one-cell strait against land | 28.4 -> **32.2** |
| 0.4° | narrow bay, 3 sides land | 28.4 -> **32.7** |
| 0.3° | as 0.4°, finer | 28.4 -> **52.6** by d10 |

Where the archipelago resolves into an isolated or near-enclosed shallow
column, horizontal exchange with the open sea collapses and the surface heat
flux piles up in place. 0.5° happens to keep that water connected, so it
passes; 0.6° and 0.4° cut it off, so they don't.

### Root cause of the fails, and the fix (measured 2026-09-18, later the same day)

The "coastline artifact, not a smaller dt" reading above is **wrong** — it was
a hypothesis, and the measurements below killed it. What actually separates
the passing rungs from the failing ones:

**1. The 0.6°/0.4° fails are the column heat leak, cured by `--project-adv-vel`.**
The leak is the flux-form advection's rigid-lid surface term `Fz_top =
Fz[0]*T[0]`, `Fz[0] = -col_div_h(u,v)`. `--project-adv-vel` drives the
stage-2 predictor's column divergence to zero. Measured (30 d each):

| res | no pav | with pav |
|---|---|---|
| 0.6° | FAIL_DRIFT 32.24 | **PASS 30.43** |
| 0.4° | FAIL_DRIFT 32.65 | **PASS 28.54** |

**2. The 0.3° fail is NOT the column projection** — it survives pav at every
CG iteration count:

| 0.3° config | result |
|---|---|
| pav, n_iter=150 | FAIL_BLOWUP |
| pav, n_iter=1600 | FAIL_BLOWUP |

The CG-convergence hypothesis for pav was checked directly with a
cross-resolution probe of `_project_column_divergence` on the stage-2
predictor. Residual column divergence after n_iter=150: 0.50% at 1.0°, 0.83%
at 0.6°, 2.46% at 0.4°, 3.02% at 0.3° — CG does degrade with dx, but raising
n_iter to 1600 fixes 0.6°/0.4° outcomes not at all and 0.3° not at all, so
iteration count is not the lever. `n_iter` is now overridable via
`OCEAN_PAV_NITER` for probes but the default stays 150.

**3. The 0.3° lever is `nu_h` (horizontal viscosity).** Sweeping nu_h at 0.3°
with pav on, everywhere else production:

| nu_h | × the dx²-scaled value | max\|u\|_peak | verdict |
|---|---|---|---|
| 4.5e5 | 1.0× | 2.007 | FAIL_BLOWUP |
| 1.0e6 | 2.2× | 1.449 | **PASS** |
| 2.0e6 | 4.4× | 1.153 | **PASS** |
| 3.0e6 | 6.7× | 1.048 | **PASS** |
| 4.5e6 | 10× | 0.951 | **PASS** |

At 0.4°/0.5°/0.6° the dx²-scaled value is already sufficient (all PASS), so
only the finest rung needs a floor. The reason is not the explicit Laplacian
CFL — `nu_h*dt/(n_nu*dx²)` is *better* at 0.3° (0.018) than at 1° (0.061),
because n_subcyc scales too. It is that the horizontal advective Courant
`|u|·dt/dx` rises 5× from 1° to 0.3° (0.06 -> 0.31): centered-difference
advection at Courant 0.3 barely damps grid-scale noise, so the finer grid
needs more nu_h as a physical noise sink, not as a CFL fix.

### What was ruled out (0.3°, all 30 d, all with pav)

| probe | result |
|---|---|
| `--min-depth` 200 / 300 / 500 | all still FAIL_BLOWUP |
| `--smooth-passes` 60 | still FAIL_BLOWUP |
| `--dt` 1800 / 900 / 600 | all FAIL_DRIFT (dt is not the lever; smaller dt made day-20 max\|T\| *worse*, 71-75 C) |
| `--nu-bi` 1.62e13 / 1.62e14 | FAIL_BLOWUP (10× more biharmonic diverged *earlier*, d10) |
| `--freeze-adv-vel` | FAIL_BLOWUP, max\|u\| 3.87 |
| final-state `col_div_h` | same order at 0.4° (1.5e-3) and 0.3° (1.3e-3) — not the discriminator |
| surface-layer (k=0) divergence | ~1e-6 at all rungs, unchanged by the projection |
| vertical w (from continuity) | 3.6e-3 m/s global max; w·dt/dz ≈ 0 at the hotspot |
| hotspot `T_init` gradient | 0.01 K (WOA) — model-generated, not init |
| connectivity to the main ocean | connected at every resolution |
| hotspot `adv` term | the only positive term, growing to +2.9e-4 K/s at 0.3° |

