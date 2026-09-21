# Monthly 2-m Air Temperature Forcing Protocol

Status: locked
Date: 2026-09-21
Baseline: annual-mean NCEP R1 2-m air temperature, 365-day FCT/TVD run

## Question

Does replacing the annual-mean NCEP 2m air target with twelve monthly-mean
NCEP R1 2m air fields improve the SST climate score?

## Motivation

The annual-mean real-air target was reproducible:

- first run: A1 RMSE `1.466 C`, A2 RMSE `1.883 C`, PASS
- repeat: A1 RMSE `1.469 C`, A2 RMSE `1.886 C`, PASS

The remaining seasonal phase is still missing. Real high-latitude cooling,
warm subtropics, and mixed-layer variability need a monthly atmospheric cycle.

## Implementation

Add an opt-in runner flag:

```text
--real-air-temp-monthly
```

The runner will load twelve calendar-month NCEP R1 2m air fields, cache them
locally, interpolate each to the ocean grid, and blend adjacent months with
the same 5-day scheme used for seasonal wind. This flag is independent of the
existing annual-mean opt-in and does not change the default zonal target.

## Run

Use one 365-day run with the current baseline physics and:

- `--real-air-temp-monthly`
- `bulk_lambda_mult = 1.0`
- all other settings unchanged

Compare against the existing annual-mean real-air runs.

## Decision rule

1. Stability must remain PASS.
2. Heat and salt drifts must remain small.
3. If monthly forcing improves global A2 RMSE by at least `2%` relative to the
   annual-mean repeat (`1.886 C`), treat it as the better real-air baseline.
4. If A2 does not improve by at least `2%`, keep annual-mean as the simpler
   baseline and investigate regional/seasonal error before another forcing
   change.
5. Do not change the default until the result is analyzed.
