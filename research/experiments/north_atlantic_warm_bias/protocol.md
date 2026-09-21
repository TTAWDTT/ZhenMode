# North Atlantic Warm-Bias Audit Protocol

Status: locked
Date: 2026-09-21
Primary run: reproduced reduced-vertical-mixing candidate

## Question

Why do the largest remaining errors concentrate in `300..360E / 40..60N`, and
are they warm biases associated with coast, deep ocean, boundary, or surface
forcing?

## Existing outputs

- `results/vertical_mixing_sensitivity/global_real_air_kv1e-6_kconv001_repeat.npz`
- `results/real_air_temp/global_real_air_2m_1deg_365d_repeat.npz`

No new model runs are performed.

## Region definition

The audit focuses on wet cells with:

- longitude in `[300, 360)` degrees E
- latitude in `[40, 60]` degrees N

Within that region, classify by:

- distance to land (`<3` cells versus `>=3` cells)
- distance to the closed north/south wall
- depth (`<1000`, `1000..3000`, `>=3000` m)
- longitude sub-sector (`300..320`, `320..340`, `340..360`)

## Metrics

For each group:

- count
- warm/cold cell share
- mean bias
- RMSE
- share of regional SSE
- `T_atm - WOA SST` forcing proxy
- correlation between SST error and `T_atm - WOA SST`

## Decision rule

If the warm bias is mostly coastal/shallow, prioritize mask/bathymetry
diagnostics. If it is mostly deep and open ocean, prioritize surface heat
flux or missing polar/ice physics. If `T_atm - WOA SST` explains a large part
of the SST error, improve the bulk heat-flux formulation before adding a
mixed-layer closure.
