# Real 2-m Air Temperature Forcing Protocol

Status: locked
Date: 2026-09-21
Baseline: 365-day FCT/TVD run with zonal WOA SST bulk target

## Question

Can replacing the zonally uniform WOA-SST-derived bulk target with observed,
spatially varying NCEP R1 2-m air temperature improve the global SST climate
without making A2 circular?

## Motivation

The scalar lambda sweep showed that stronger restoring improved A1 but gave
only a small A2 improvement:

- baseline lambda 1: A1 RMSE `1.013 C`, A2 RMSE `2.115 C`
- lambda 0.25: A1 RMSE `3.104 C`, A2 RMSE `3.387 C`
- lambda 0.5:  A1 RMSE `1.835 C`, A2 RMSE `2.482 C`
- lambda 2.0:  A1 RMSE `0.504 C`, A2 RMSE `2.031 C`

The pre-registered 5% A2 improvement threshold was not reached. This supports
the conclusion that scalar restoring is not the limiting factor. The missing
piece is likely the horizontal structure of atmospheric forcing.

## Implementation

Add an opt-in runner flag:

```text
--real-air-temp
```

For now it uses the calendar-year annual mean of NCEP R1 2-m air temperature.
The original zonal WOA SST target remains the default. The new target is not a
WOA SST restoring field and therefore does not make the A2 score circular.

## Run

Use one 365-day run with the baseline physics and:

- `--real-air-temp`
- `bulk_lambda_mult = 1.0`
- all other settings unchanged from the FCT baseline

## Pass / continue criteria

1. Stability must remain PASS.
2. Global heat and salt drifts must remain small and comparable to baseline.
3. If A2 RMSE improves by at least `5%`, repeat once and consider making
   monthly-varying 2-m air temperature the next experiment.
4. If A2 RMSE does not improve by at least `5%`, do not make real-air target
   default immediately; instead compare its regional error structure and move
   to vertical mixing or a more complete atmospheric state.

## Relation to mature models

MOM6, NEMO, ROMS, HYCOM and MPAS-Ocean normally use spatially varying
atmospheric states. A single meridional target can be useful for controlled
numerical testing, but it is not sufficient for a climate-skill benchmark.
