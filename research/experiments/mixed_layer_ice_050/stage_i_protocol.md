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
