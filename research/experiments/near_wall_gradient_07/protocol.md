# Near-Wall SST Gradient Diagnostic at 0.7 Degree

Status: complete
Date: 2026-09-22
Baseline: `candidate_65n_07_gm0`
Question: Does the remaining North Atlantic cold bias track local SST gradients,
current-gradient alignment, or solver advective cooling?

## Inputs

Use the matched 0.7-degree `gm0_3dterms` and `gm500_3dterms` runs. For each
run, average the saved state fields over the final 90 days (days 300, 330, 360,
and 365). Use WOA surface temperature from the run metadata as the reference.

## Definitions

- `grad_east` and `grad_north`: surface SST derivatives in K per 100 km.
- `alignment`: cosine of the angle between mean surface velocity and the SST
  gradient. Positive means the current points up-gradient.
- `adv_tend`: horizontal proxy `-u * dT/dx - v * dT/dy` in K/day.
- `saved_adv`: the full solver advection tendency from `terms_*.npy` in K/day.
- Regions: North Atlantic `300..360E / 40..60N`, near-wall `55..60N`, and
  interior `40..55N`.

## Outputs

- `region_summary.csv`
- `worst_near_wall_cells.csv`
- `metrics.json`
- `analysis.md`

The worst-cell file ranks the 100 coldest near-wall cells per run and records
their gradients, velocity, alignment, and advection.
