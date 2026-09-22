# Coastal vs High-Latitude Ventilation Protocol

Status: complete
Date: 2026-09-22
Baseline: `candidate_65n_07_gm0`

## Question

After excluding land-adjacent cells, does the `59..60N` weak-ventilation
cluster still explain the near-wall cold bias?

## Group definitions

- `near_wall_all`: `300..360E / 55..60N`
- `coastal_dist_le3`: within 3 grid cells of land
- `interior_dist_gt3`: farther than 3 grid cells from land
- `highlat_interior_59_60`: interior cells with latitude `>=59`
- `interior_coldest_100`: the 100 coldest interior near-wall cells

Land distance is in grid cells from the ocean mask, with periodic longitude.

## Metrics

- Area-weighted SST bias and RMSE
- Share of cells colder than WOA by more than 1 C
- Mean land distance and surface speed
- Surface-node advection, convection, bulk flux, GM, and net tendency
- Upper 50--200 m advection and convection

## Decision rule

If the coastal group remains much worse than the interior group, prioritize a
coastal masking/ventilation diagnostic. If `59..60N` interior still remains
materially cold after excluding land-adjacent cells, then design a narrowly
scoped high-latitude boundary/ice-proxy experiment.
