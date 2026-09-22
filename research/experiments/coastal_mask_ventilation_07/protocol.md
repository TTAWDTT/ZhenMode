# Coastal Mask and Ventilation Attribution Protocol

Status: complete
Date: 2026-09-22
Baseline: `candidate_65n_07_gm0`

## Question

How much of the 0.7-degree near-wall cold bias is caused by land-adjacent or
shallow cells, and how much remains in the deep interior?

## Data

Use the matched final-90d GM0 and GM500 states. Compute:
- distance to land in grid cells,
- bathymetric depth from the reconstructed grid,
- distance to deep water (`depth >= 1000 m`) as a connectivity proxy,
- raw SST error against WOA.

## Groups

Exclusive land-distance groups:
- `0--3` cells,
- `4--7` cells,
- `>=8` cells.

Exclusive depth groups:
- `100--200 m`,
- `200--500 m`,
- `500--1000 m`,
- `>=1000 m`.

Cross groups:
- `coastal_shallow` = land distance `<=3` and depth `<200 m`,
- `coastal_deep` = land distance `<=3` and depth `>=200 m`,
- `interior_shallow` = land distance `>3` and depth `<200 m`,
- `interior_deep` = land distance `>3` and depth `>=200 m`.

## Metrics

- Area-weighted SST bias and RMSE.
- Share of regional SSE.
- Share of cells colder than WOA by more than 1 C.
- Mean land distance, depth, speed, and distance to deep water.
- Counterfactual regional RMSE if the group error were set to zero.
