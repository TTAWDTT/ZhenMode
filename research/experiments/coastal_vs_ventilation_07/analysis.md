# Coastal vs Weak-Ventilation Diagnosis

Date: 2026-09-22
Status: complete
Runs: `gm0_3dterms`, `gm500_3dterms`

The table uses final-90d mean SST and surface-node heat tendencies. Land
distance is in grid cells.

| Run | Group | cells | SST bias (C) | SST RMSE (C) | cold >1C share | surface net (K/day) |
|---|---|---:|---:|---:|---:|---:|
| GM0 | all near wall | 548 | -1.081 | 1.186 | 0.568 | +0.552 |
| GM0 | coastal <=3 cells | 122 | -1.453 | 1.539 | 0.836 | +0.778 |
| GM0 | interior >3 cells | 426 | -0.976 | 1.065 | 0.491 | +0.488 |
| GM0 | 59--60N interior | 56 | -1.197 | 1.246 | 0.679 | +0.471 |
| GM0 | interior coldest 100 | 100 | -1.516 | 1.533 | 1.000 | +0.398 |
| GM500 | all near wall | 548 | -1.146 | 1.246 | 0.637 | +0.417 |
| GM500 | coastal <=3 cells | 122 | -1.447 | 1.537 | 0.852 | +0.558 |
| GM500 | interior >3 cells | 426 | -1.061 | 1.151 | 0.575 | +0.377 |
| GM500 | 59--60N interior | 56 | -1.342 | 1.393 | 0.804 | +0.414 |
| GM500 | interior coldest 100 | 100 | -1.630 | 1.638 | 1.000 | +0.154 |

## Interpretation

1. Coastal cells remain the larger error source: GM0 coastal RMSE is `1.539 C`
   versus `1.065 C` for interior cells.
2. The `59--60N` interior cluster still exists, but it is secondary. GM0 gives
   `1.246 C` RMSE and `0.679` cold-cell share there, after land-adjacent cells
   are excluded.
3. GM500 degrades this interior cluster through much stronger surface advection:
   `-0.157 K/day` versus `-0.092 K/day` in GM0. This is consistent with the
   earlier regional attribution.
4. Even the GM0 interior coldest 100 have positive net surface warming
   (`+0.398 K/day`). So the cold bias is not simply a present-day local cooling
   source in these cells.

## Decision

The next priority is a coastal masking/ventilation diagnostic, not a broad
transport retune. The high-latitude interior signal is real but secondary, so a
narrow high-latitude boundary/ice-proxy experiment should follow only after the
coastal contribution is isolated.
