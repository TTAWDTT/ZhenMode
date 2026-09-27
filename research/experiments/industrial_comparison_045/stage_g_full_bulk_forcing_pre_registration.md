# Stage-G full bulk forcing pre-registration

Date: 2026-09-27
Status: pre_registered, implementation ready, not launched

## Purpose

Close the largest physical-process gap versus industrial ocean models: move from
wind plus a live-SST bulk heat proxy to a full surface flux contract.

## Confirmed NCEP R1 sources

The first Stage-G loaders are now available:

- 2m specific humidity: `shum.2m.mon.mean.nc`, g/kg -> kg/kg;
- downward longwave radiation: `dlwrf.sfc.mon.mean.nc`, W/m2;
- downward shortwave radiation: `dswrf.sfc.mon.mean.nc`, W/m2;
- precipitation rate: `prate.sfc.mon.mean.nc`, kg/m2/s.

They are read on the native NCEP grid, cached locally, bilinearly remapped to
the solver grid, and unit-converted before entering the flux code.

## Implementation status

The solver path and Stage-G run script are ready. The 2023 0.5-degree NCEP
forcing artifact contains monthly air temperature, specific humidity, downward
longwave, downward shortwave, precipitation rate, and 10-m wind speed. It is
not launched until the MOM6 annual v12 gate is scored.

## Required forcing fields

1. 2m air temperature (already used);
2. 2m specific humidity or relative humidity;
3. downward shortwave and longwave radiation;
4. precipitation and evaporation;
5. runoff only if both models can consume the same file.

All fields must be on the shared 0.5-degree grid or remapped with a documented
conservative method. Units, source year, and missing-value policy must be
recorded in the manifest.

## Candidate runs

1. 30d smoke: stability and flux-sign sanity only.
2. 365d annual: same grid, bathymetry, initial state, wind, and reference as the
   existing annual no-ice Stage-F control.

## Promotion gates

1. both runs PASS the stability watchdog;
2. heat and salt drift bounded;
3. annual global A2 not worse than the pre-registered no-ice Stage-F control;
4. NA 40--60N surface RMSE not worse;
5. near-wall signed bias not worse;
6. global 3D temperature RMSE not worse;
7. NA 40--60N 3D RMSE not worse;
8. flux diagnostics report global mean shortwave, longwave, latent, sensible,
   precipitation-minus-evaporation, and runoff.

## Shared-grid provenance

The Stage-G launcher now auto-generates a 0.1-degree ETOPO twin from the
MOM6 0.5-degree `topog_050.nc`, and a second helper converts the shared
MOM6 WOA T/S file into `data/stage_g/stage_g_init_2023_050.npz`. Together
they remove the dependence on prior Stage-F artifacts and reproduce the
shared 67.9% ocean grid without extra smoothing.

## Provenance

The manifest must record every forcing file, source variable, unit conversion,
temporal interpolation, and the exact solver commit. If a field is disabled in
both models, the manifest must explicitly say `not_used_in_both_runs`; it may
not be silently dropped.

## Blocking gate

Do not launch until the annual MOM6 Stage-F v12 final-90d 3D/MLD gate is scored.
