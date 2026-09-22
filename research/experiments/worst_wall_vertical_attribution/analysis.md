# Worst Near-Wall Cells: Vertical Heat Attribution

Date: 2026-09-22
Status: complete
Runs: `gm0_3dterms`, `gm500_3dterms`

For each run, rank the 100 coldest cells in `55..60N / 300..360E` by final-90d
SST error and compare their saved heat tendencies. The surface-node values below
are from `z=0`; the 50--200 m bin averages depth nodes between 50 and 200 m.

## Surface-node tendency (K/day)

| Run | Group | advection | convection | surface bulk flux | GM |
|---|---|---:|---:|---:|---:|
| GM0 | all near wall | -0.0377 | +0.4551 | +0.1343 | 0 |
| GM0 | error <= -1 C | -0.0493 | +0.3695 | +0.1547 | 0 |
| GM0 | 100 coldest | -0.0578 | +0.6124 | +0.0961 | 0 |
| GM500 | all near wall | -0.0752 | +0.2806 | +0.1562 | +0.0552 |
| GM500 | error <= -1 C | -0.1109 | +0.1764 | +0.2022 | +0.0490 |
| GM500 | 100 coldest | -0.1872 | +0.2505 | +0.2221 | +0.0422 |

In the 50--200 m layer, convection is a small cooling term. For the 100 coldest
cells it is `-0.00556 K/day` in GM0 and `-0.00233 K/day` in GM500. Diffusion
remains negligible.

## Geometry

The 100 coldest GM0 cells have mean distance-to-land of `3.9` grid cells and
mean surface speed of `0.038 m/s`. They include both coastal cells (many at
1--3 cells from land) and near-wall cells farther from land. Their mean surface
convection is positive, but their bulk heat input is smaller than in the full
near-wall group.

## Interpretation

1. GM500 produces much stronger local advective cooling in the coldest cells
   (`-0.187` versus `-0.058 K/day`). This explains the regional improvement
   from disabling GM.
2. Even in the GM0 coldest cells, the mean surface-node total tendency is
   warming, not cooling. The remaining cold bias is therefore not caused by a
   simple instantaneous cooling term in these cells.
3. The worst cells are a mixed population: many touch or nearly touch land,
   while others sit in the weakly ventilated `59--60N` sector. This supports
   local geometry/ventilation as the next target rather than another global
   transport retune.

## Next

Map these cells against bathymetry and land geometry, then test a narrowly
scoped high-latitude boundary/ice-proxy experiment only if the same clusters
persist after excluding land-adjacent cells.
