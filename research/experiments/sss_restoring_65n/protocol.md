# Surface Salinity Restoring on the 65N lambda80 GM500 Localized-Convection Candidate

Status: locked
Date: 2026-09-21
Baseline: `65N + lambda80 + GM500 + localized convection + annual real air`

## Question

Does the remaining North Atlantic cold bias come partly from a biased density
field and weak thermohaline circulation? The model currently has no SSS
restoring, and the candidate already shows a regional salinity bias.

## Runs

Keep all candidate settings unchanged. Add WOA surface salinity restoring:

1. `sss30`: `--sss-restore-days 30`
2. `sss90`: `--sss-restore-days 90`

The reference is the preferred diagnostic candidate with no SSS restoring.

## Metrics

- Global A1/A2 RMSE
- North Atlantic `300..360E / 40..60N` bias and RMSE
- near-wall `55..60N` bias and RMSE
- salt drift and stability
- 365d climate metric

## Decision rules

1. If either SSS-restoring run improves the target region, density forcing is
   a relevant lever for the remaining circulation bias.
2. If the stronger 30d restoring helps but 90d does not, the bias is sensitive
   to restoring strength and needs a mechanistic decomposition.
3. If both worsen, the current density field is not the main limiting factor.
