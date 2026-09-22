# Near-Wall Temperature-Gradient / Advection Diagnosis

Status: locked
Date: 2026-09-22
Baseline: `candidate_65n_07_gm0`
Reference runs: `gm0_3dterms`, `gm500_3dterms`

## Question

Why does the GM500 closure increase near-wall advective cooling even though the
large-scale surface velocity field is nearly unchanged?

## Hypothesis

GM500 changes the local SST gradient/orientation more than the current speed.
If so, the relevant lever is the near-wall temperature-gradient structure or
advective orientation, not bulk current speed.

## Metrics

Use final 90-day 3D snapshots at 0.7 degree. For North Atlantic
`300..360E / 40..60N` and near-wall `55..60N`:

- SST gradient magnitude and dominant direction
- surface velocity magnitude and direction
- velocity/temperature-gradient alignment
- advection tendency from the saved 3D stack
- local correlations between these fields and SST error

## Decision rules

1. If cold cells have stronger `|grad(T)|` and stronger velocity-gradient
   alignment, target the near-wall thermal front and advective orientation.
2. If GM500 mainly strengthens the velocity-gradient alignment, retune or taper
   GM rather than disabling it globally.
3. If velocity fields are still nearly identical but temperature gradients
   differ, focus on heat redistribution and vertical mixing, not momentum.
