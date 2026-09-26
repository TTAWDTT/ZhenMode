# Stage-F 30d dynamic-bulk slice

Date: 2026-09-26  
Status: ocean_solver-side probe complete  
Purpose: move from prescribed restoring to the first realistic bulk heat-flux
slice without sea ice.

## Contract

- Shared 0.5-degree global grid, bathymetry, initial WOA T/S.
- 2023 monthly NCEP wind stress.
- 2023 monthly NCEP R1 2m air temperature.
- Dynamic Haney/Bulk form `q = lambda*(T_air - SST)` with `lambda=80 W/m2/K`.
- No sea ice, no SST/SSS restoring, no idealized meridional heat flux.
- 30d integration; final 10d scoring window.
- `score_external_model.py` or `benchmark_metrics.py` only.

## Ocean_solver run

```bash
DAYS=30 bash scripts/run_industrial_comparison_050_stage_f_30d.sh
```

Result:

- verdict `PASS`;
- global A2 RMSE 1.692 C;
- NA 40--60N RMSE 2.088 C;
- near-wall bias -1.872 C;
- heat drift -0.184 percent.

## Interpretation

This is **not** a climate claim.  It is a diagnostic bulk-flux control.  On the
same monthly-air/no-ice setup, the Stage-I dynamic-ice 30d probe improves to
global A2 0.956 C and NA RMSE 0.874 C.  Thus sea-ice/mixed-layer coupling is a
major lever for this forcing, not a cosmetic add-on.

## Limitation

The prepared MOM6 Stage-F slice currently uses the prescribed
`lambda*(air-WOA SST)` sensible proxy, not an instantaneous dynamic bulk
formulation.  A direct dynamic-bulk comparison therefore needs that limitation
recorded as `not_comparable` until both models share the same bulk closure.
