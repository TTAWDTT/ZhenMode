# Surface Forcing Sensitivity Protocol

Status: locked
Date: 2026-09-21
Baseline: 365-day FCT/TVD run with `lambda_bulk = 40 W m^-2 K^-1`

## Question

Does the current zonally uniform bulk heat restoring control the failed A2
SST climate score?

The polar/boundary attribution found:

- closed-wall cells contribute only about `5.6%` of global A2 SSE;
- coastal cells contribute about `31.6%`;
- deep open-ocean cells contribute about `62.6%`;
- the regression
  `model - WOA ~ T_atm - WOA`
  has global correlation `0.834`, slope `0.886`, and `R^2 = 0.695`.

This suggests that the surface target is at least as important as the polar
cap. The A2 test remains non-circular because `T_atm` is only the ocean-only
zonal mean of WOA SST. Zonal SST structure must still be predicted by the
ocean dynamics.

## Hypothesis

The bulk restoring is either too strong or too weak for the current one-degree
configuration. Reducing or increasing it should reveal whether surface
restoring is the limiting factor.

## Sensitivity runs

Use the baseline FCT/TVD configuration and vary only:

- `--bulk-lambda-mult 0.25`
- `--bulk-lambda-mult 0.5`
- `--bulk-lambda-mult 2.0`

Keep unchanged:

- 1-degree grid
- 365 days
- `dt = 3600 s`
- mode splitting and `lax.scan`
- float32
- GM `kappa_gm = 1000`
- FCT/TVD tracer transport
- seasonal 2023 wind
- bathymetry smoothing and minimum-depth handling

The existing baseline corresponds to `--bulk-lambda-mult 1.0`.

## Metrics

For each run:

- A1 zonal-mean SST correlation and RMSE
- A2 SST pattern correlation and RMSE
- max|u|, max|eta|, KE drift
- global heat and salt drift
- error grouped by the same polar/boundary/ coast / deep-ocean regions
- regression of `model - WOA` against `T_atm - WOA`

## Decision rule

1. If no multiplier improves global A2 RMSE by at least `5%`, stop tuning
   scalar bulk restoring. The next experiment should address spatially varying
   atmospheric forcing or vertical mixing.
2. If a multiplier improves A2 RMSE by at least `5%` without instability or a
   materially worse budget, repeat that setting once and only then consider a
   default change.
3. Do not adopt a setting merely because it reduces coastal error while
   worsening the global score.
4. Do not switch to a full two-dimensional WOA SST restoring target in this
   experiment; that would make A2 circular.

## Why this follows the model survey

MOM6, NEMO, ROMS, HYCOM and MPAS-Ocean are normally driven by spatially and
temporally varying atmospheric states, not by a single zonal profile. The
current solver's target is useful for controlled testing, but it is a severe
idealization. The survey therefore points first to surface forcing, not to
more transport tuning or an immediate vertical-coordinate rewrite.
