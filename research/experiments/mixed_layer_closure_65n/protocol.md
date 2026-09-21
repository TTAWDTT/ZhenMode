# Mixed-Layer / Convective-Closure Sensitivity on the 65N lambda80 GM500 Candidate

Status: locked
Date: 2026-09-21
Baseline: `65N + lambda_bulk=80 + kappa_gm=500 + kappa_v=1e-6 + kappa_conv=0.01`

## Question

After reducing GM strength, is the remaining high-latitude cold bias controlled
by convective/mixed-layer mixing?

## Runs

Keep all other settings unchanged. Compare three closures:

1. `localized_conv`: gate convective adjustment per unstable interface using
   `--localize-conv`.
2. `kconv005`: use the original stronger convective diffusivity
   `--kappa-conv 0.05`.
3. `kconv0002`: use weaker convective diffusivity
   `--kappa-conv 0.002`.

The reference is the GM500 run with column-wide convective adjustment and
`kappa_conv=0.01`.

## Metrics

- Global A1/A2 correlations and RMSEs
- North Atlantic `300..360E / 40..60N` bias and RMSE
- near-wall `55..60N` bias and RMSE
- 365d stability, heat/salt drift, KE behavior

## Decision rules

1. If `localized_conv` warms the target region and remains stable, the current
   column-wide convective mask is over-mixing.
2. If `kconv0002` improves more than localization, convective diffusivity
   strength is the primary lever.
3. If `kconv005` improves instead, the candidate mixed layer is too weak after
   the earlier scalar tuning.
4. If none materially improves, inspect 3D heat-tendency snapshots rather than
   continue scalar sweeps.
