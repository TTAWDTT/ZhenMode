# Worst Near-Wall Vertical Attribution Protocol

Status: complete
Date: 2026-09-22
Baseline: `candidate_65n_07_gm0`

## Selection

For each matched 0.7-degree run, rank the 100 cells with the coldest final-90d
SST error inside `300..360E / 55..60N`.

## Diagnostics

Use the saved final-90d mean of the 3D heat-tendency snapshots. Report terms by
surface node, 0--50 m, 50--200 m, and 200--1000 m. Add the annual real-air bulk
heat tendency at the surface node. Record land distance and mean surface speed
for each coldest cell.

## Decision rules

1. If the coldest cells have strongly negative saved advection, treat local
   transport as the cause.
2. If they have positive net surface warming, look for geometry, ventilation, or
   history rather than a current local cooling source.
3. If surface convection warms while 50--200 m cools materially, revisit
   vertical redistribution; otherwise deprioritize it.
