# GM0 Attribution at 0.7 Degree

Status: locked
Date: 2026-09-22
Baseline: `candidate_65n_07_gm0`
Question: Is the GM0 benefit physical or an artifact of an over-strong GM closure?

## Runs

Use the locked candidate configuration. Save 30-day 3D state and heat-tendency
snapshots. Run two cases:

1. `gm0_3dterms`: `kappa_gm=0`
2. `gm500_3dterms`: `kappa_gm=500`

Everything else is held fixed.

## Metrics

- Global and North Atlantic A1/A2 RMSE
- Surface and subsurface heat-tendency decomposition
- North Atlantic horizontal velocity statistics
- Zonally integrated meridional heat transport at selected latitudes
- Regional surface/subsurface heat redistribution

## Decision rules

1. If GM0 improves transport structure and does not create physical
   inconsistencies, GM over-strength/taper is the issue.
2. If GM500 suppresses a spurious circulation, GM is still needed but must be
   retuned or tapered differently.
3. If the two runs differ mainly in numerical noise, freeze GM0 cautiously and
   prioritize direct boundary-current diagnostics.
