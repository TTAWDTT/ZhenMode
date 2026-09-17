# Resolution scaling: measured CFL limits

Measured on 2026-09-18 with the CPU checkerboard / smooth-anomaly probes
(`src/jax_solver_global.py` at commit `190e70c`). Every number below is
from an actual run, not an estimate.

## TL;DR

**Do not run `--resolution` finer than 1.0° with the production dt.** The
solver has a real, dx-dependent time-step limit that is NOT the
biharmonic, NOT the horizontal Laplacian, and NOT the RK2 stage-2 column
divergence. At 0.5° it forces `--dt <= 300 s` (measured, with `--mode-split`
and `--dt-bt 300`). There is no flag that removes it.

## What was measured

Setup: `make_global_grid(..., lat_max=60.0, smooth_passes=30,
min_depth=100.0)`, `make_solver_global(..., mode_split=True, dt_bt=300)`,
initial state `T = 15 C + 1 C * cos(2*lon) * cos(2*lat)`, `S = 35`,
`nu_h = nu_bi = kappa_gm = kappa_redi = kappa_conv = 0` (physics fully
off, so only the advection / Coriolis / rigid-lid pressure solve acts).
30-40 steps, `|u|` tracked.

| resolution | dt | `--project-adv-vel` | outcome |
|---|---|---|---|
| 1.0° | 3600 | off | **stable** (saturates ~1e-1 m/s) |
| 1.0° | 900 | off | **stable** (saturates ~6e-2) |
| 0.5° | 3600 | off/on | **DIVERGED** @ 7 steps |
| 0.5° | 900 | off/on | **DIVERGED** @ 17-18 steps |
| 0.5° | 600 | off | **DIVERGED** @ 27 steps |
| 0.5° | **300** | off/on | **stable** (saturates ~4e-2) |

The 0.5° / dt=300 trajectory is essentially identical to the 1.0° / dt=900
one, so the stable branch is the physical one; dt=600 and up are genuinely
unstable, not slow transients.

## Rules ruled OUT

- **Biharmonic (`nu_bi`)**: setting `nu_bi = 0` does not fix 0.5°. The
  instability persists with the biharmonic term fully removed.
- **Horizontal Laplacian (`nu_h`)**: same — `nu_h = 0` does not fix it.
  (Note the production `nu_h = 5e6` *is* itself over its CFL at 0.5°:
  `5e6 * 300 / 28009^2 = 1.9`, vs the 0.25 limit in
  `nu_nsub='cfl'`. That is a *separate*, second failure that must also be
  addressed, via `--nu-nsub cfl` or a smaller `nu_h`.)
- **RK2 stage-2 column divergence** (the Defect-6 / `--project-adv-vel`
  target): turning it on moved the blow-up by 1-2 steps only. Not the cause.

## Why the earlier analytical CFL table was wrong

The budget table from the previous session used `dt_bt = 300` and predicted
the biharmonic term was the binding constraint at 0.5° (CFL ~0.58 vs a 0.05
threshold). That table is superseded: with the biharmonic removed entirely
the solver still diverges, so the constraint is elsewhere and cannot be
budgeted from `nu_bi`.

## Practical guidance

- `--resolution 1.0` and coarser: production settings are fine.
- `--resolution 0.5`: **`--dt` must be <= 300 s** (10x more steps per
  simulated year), and `nu_h` must be reduced or `--nu-nsub cfl` used
  (`5e6` is ~7.6x over the Laplacian CFL there).
- The cause of the residual dx-dependent limit is **not yet identified**.
  It is the main blocker before any sub-1° scientific run; the obvious
  next steps are (a) instrument which tendency term first exceeds its
  neighbours at 0.5°, (b) check whether the 2/3-rule dealias mask
  (`dealias_lon_mask`, sized from `nx`) and the `adv_nsub` vertical-advection
  subcycle count — both of which silently depend on grid size — are the
  culprits.
