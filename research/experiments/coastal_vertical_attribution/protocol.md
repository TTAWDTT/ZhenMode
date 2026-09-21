# Coastal / Vertical Attribution Diagnostic Protocol

Status: locked
Date: 2026-09-21
Primary run: `results/real_air_temp/global_real_air_2m_1deg_365d_repeat.npz`

## Question

After switching to annual-mean real 2m air temperature, where does the
remaining SST error live, and is it more consistent with coastal masking,
coastal upwelling, or vertical mixing?

## No new model run

This is a diagnostic-only experiment. It uses the reproduced annual real-air
365-day output. No transport, forcing, or mask changes are made here.

## Climate fields

Use the last 90 days of `T_top` snapshots as the model SST climatology.
Compare against WOA surface temperature stored in `T_init[:, :, 0]`.

Three error fields are diagnosed:

1. **raw ocean error**: `model_sst - woa_sst`, ocean cells only;
2. **official A2 smoothed error**: the same non-masked 2-degree box smoother
   used by the locked climate benchmark;
3. **mask-aware smoothed error**: a 2-degree ocean-only smoother, used only to
   test whether the benchmark is amplifying coastal mask artifacts.

## Explanatory variables

For each wet cell:

- distance to land in grid cells, periodic in longitude;
- distance to the closed north/south wall;
- smoothed ETOPO depth;
- latitude;
- WOA SST;
- WOA surface-minus-50m stratification;
- annual mean wind-stress magnitude from the exact twelve monthly snapshots
  used by the run;
- annual mean absolute wind-stress curl.

## Metrics

For each group report:

- count;
- mean error;
- RMSE;
- share of global summed squared error for raw and official A2 errors;
- maximum absolute error.

For the official A2 error, also report the remaining global RMSE if that
group's error were exactly zero. This is attribution only, not a proposed
model result.

## Diagnostic questions

1. Does the coast dominate because of per-cell RMSE, or does deep ocean
   dominate because of area?
2. Are the largest errors physically localized cold/warm anomalies, or a broad
   hemispheric bias?
3. Do stronger winds make coastal cells colder, as simple coastal upwelling
   would suggest?
4. Does stronger WOA surface stratification coincide with colder model SST,
   which would be consistent with excessive downward mixing rather than
   insufficient mixing?

## Decision rule

- If the coast has high RMSE and mask-aware smoothing sharply reduces the
  coastal share, investigate mask/smoothing artifacts before physics.
- If coastal cold bias increases with wind, prioritize upwelling/vertical
  exchange.
- If stronger stratification coincides with colder model SST, prioritize a
  mixed-layer / vertical mixing experiment.
- If deep open ocean remains the largest SSE share, design a broad heat-balance
  experiment in addition to any coastal fix.
