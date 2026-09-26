# Stage-I minimal closed-loop diagnostic

Date: 2026-09-26  
Status: 30d diagnostic complete  
Purpose: close the requested minimal ice/mixed-layer loop on the existing
monthly dynamic-ice probe and record the required budget diagnostics.

## Definition

The closed loop is:

```text
atmospheric forcing
  -> mixed-layer heat capacity
  -> freezing / melting
  -> ocean heat + salt flux
  -> SST / mixed-layer response
```

It is **not** a full sea-ice model.  The current closure has stateful thickness,
latent growth/melt, an insulation proxy, and brine-rejection salt flux.

## Source run

- `results/dynamic_ice_050/global_dynamic_ice_monthly_nofloor_strat_mld_050_30d_fixedsalt.npz`
- 0.5 degree, 65N domain, 30d monthly cold-air forcing
- stratification-derived MLD, dynamic ice enabled, no ice-air floor
- standardized benchmark:
  `benchmark_30d_dynamic_ice_monthly_nofloor_strat.json`

## Manifest

- `stage_i_minimal_closed_loop_manifest.json`

It records:

1. ice thickness/extent;
2. growth/melt volume and latent heat;
3. total brine salt flux;
4. effective mixed-layer depth;
5. surface heat-budget residual;
6. stability verdict;
7. same-grid climate score.

## Interpretation

The 30d monthly run is stable and forms explicit ice, so it is useful as a
minimal closed-loop diagnostic.  It is not a production default and not an
annual climate claim.  The annual no-floor run still forms no explicit ice and
is retained as a negative control.


## Annual diagnostic

A 365d monthly-air dynamic-ice run also closed the same loop:

- `results/mixed_layer_ice_050/global_dynamic_ice_monthly_air_strat_365d.npz`
- standardized benchmark: `benchmark_365d_dynamic_ice_monthly_air.json`
- manifest: `stage_i_annual_dynamic_ice_manifest.json`

It forms explicit ice (77 cells, max thickness 4.77 m, extent 1.32e11 m2),
has finite growth/melt and brine salt flux, and remains stable.  It is
therefore the first **annual minimal closed-loop diagnostic**, but not a
production default:

- global A2 RMSE 1.362 C vs the 0.5-degree ice-floor control 1.128 C;
- NA RMSE 1.190 C vs control 0.924 C;
- near-wall bias improves to -0.256 C vs control -0.921 C.

The next tuning target is therefore not simply more ice, but a regional
heat-budget/MLD formulation that keeps the near-wall gain without hurting
North Atlantic RMSE.