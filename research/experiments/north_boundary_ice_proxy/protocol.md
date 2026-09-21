# North Boundary / Polar-Cap Sensitivity Protocol

Status: locked
Date: 2026-09-21
Candidate baseline: reduced vertical mixing + annual real-air forcing

## Question

Is the cold SST bias near `300..360E / 40..60N` caused by the closed northern
wall at 60N or by the current polar-cap treatment?

## Hypothesis

The current 60N wall and 2-row polar cap may be forcing cold water to
accumulate near the northern boundary. Extending the domain poleward or
widening the cap should reveal whether this is a boundary artifact.

## Runs

Use the candidate reduced-mixing physics and annual real-air forcing. Change
only boundary geometry:

1. `lat65`: extend the northern boundary to `65N` and use `ny=130` so the
   meridional spacing stays close to the current 1-degree grid.
2. `polarcap4_6`: widen the polar cap to `4` rows with a `6`-row taper.
3. `nocap90d`: turn the polar cap off for a 90-day stability probe.

Keep unchanged:

- 1-degree nominal resolution
- annual-mean NCEP R1 2m air forcing
- FCT/TVD transport
- GM `kappa_gm = 1000`
- seasonal 2023 wind
- `kappa_v = 1e-6`
- `kappa_conv = 0.01`

## Metrics

For each 365d run:

- A1 and A2 SST correlations/RMSEs
- global heat/salt drift
- North Atlantic region `300..360E / 40..60N` bias and RMSE
- near-wall bias and RMSE
- depth-binned regional errors

For the 90d no-cap probe, only stability and boundedness are required.

## Decision rules

1. If the `lat65` run materially reduces the near-wall cold bias, the closed
   60N wall is a major source of the error.
2. If `polarcap4_6` improves more than `lat65`, the polar-cap treatment is the
   main lever.
3. If neither helps, the cold bias is more likely from surface heat flux or
   missing high-latitude physics.
4. If the no-cap probe is unstable, keep the cap and diagnose with the
   widened-cap run instead.
