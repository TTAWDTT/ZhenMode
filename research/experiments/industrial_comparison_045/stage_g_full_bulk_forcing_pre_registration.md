# Stage-G full bulk forcing pre-registration

Date: 2026-09-27
Status: pre_registered, not launched

## Purpose

Close the largest physical-process gap versus industrial ocean models: move from
wind plus a live-SST bulk heat proxy to a full surface flux contract.

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

## Provenance

The manifest must record every forcing file, source variable, unit conversion,
temporal interpolation, and the exact solver commit. If a field is disabled in
both models, the manifest must explicitly say `not_used_in_both_runs`; it may
not be silently dropped.

## Blocking gate

Do not launch until the annual MOM6 Stage-F v12 final-90d 3D/MLD gate is scored.
