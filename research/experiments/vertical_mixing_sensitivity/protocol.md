# Vertical Mixing Sensitivity Protocol

Status: locked
Date: 2026-09-21
Baseline: reproduced annual real-air 365-day run

## Question

Is the remaining cold SST bias caused by excessive vertical mixing?

The annual real-air run has a global mean SST bias of `-1.21 C`.  The cold bias
grows from `-0.55 C` in the weakest WOA surface-to-50 m stratification quintile
to `-1.58 C` in the strongest quintile.  This correlation is suggestive but not
proof.

## Hypothesis

The model mixes surface heat downward too efficiently.  Reducing vertical
diffusivity and/or convective adjustment should warm the surface, reduce the
strong-stratification cold bias, and improve the global SST pattern score.

## Runs

Use the reproduced annual real-air baseline physics.  Only change:

1. `kappa_v = 1e-6` (10x lower background vertical diffusivity)
2. `kappa_conv = 0.01` (5x lower convective adjustment)
3. both together

Keep unchanged:

- annual-mean NCEP R1 2m air forcing
- 365 days
- 1-degree grid
- FCT/TVD transport
- GM `kappa_gm = 1000`
- seasonal 2023 wind
- all bathymetry and mask settings

The existing reproduced baseline is
`results/real_air_temp/global_real_air_2m_1deg_365d_repeat.npz`.

## Metrics

For each run:

- A1 zonal-mean SST correlation and RMSE
- A2 SST pattern correlation and RMSE
- stability verdict
- global heat and salt drift
- mean SST bias
- mean bias and RMSE by WOA surface-to-50m stratification quintile
- regional error shares: coast, deep ocean, near wall

## Decision rules

1. Stability must remain PASS.
2. A candidate must improve global A2 RMSE by at least `2%` relative to the
   reproduced annual real-air baseline (`1.886 C`).
3. If a reduced-mixing run improves the mean bias in the strongest
   stratification quintile by at least `0.2 C`, record that as support for the
   over-mixing hypothesis.
4. If lowering mixing causes instability, do not immediately reject the
   hypothesis. Record the stability failure and consider a more complete
   mixed-layer closure instead of a scalar-only change.
5. Do not change the default physics until a 365d candidate has been analyzed.
