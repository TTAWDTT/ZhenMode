# Horizontal-Viscosity Sensitivity on the 65N lambda80 GM500 Localized-Convection Candidate

Status: locked
Date: 2026-09-21
Baseline: `65N + lambda80 + GM500 + localized convection + annual real air`

## Question

Is the remaining North Atlantic cold bias limited by overly dissipative
horizontal momentum viscosity, which may weaken boundary currents and reduce
horizontal heat transport?

## Runs

Keep all candidate settings unchanged. Vary only the Laplacian horizontal
viscosity:

1. `nu_h2.5e6`: half the production value
2. `nu_h1e6`: one-fifth the production value

The reference is the preferred diagnostic candidate with `nu_h=5e6`.

## Metrics

- Global A1/A2 RMSE
- North Atlantic `300..360E / 40..60N` bias and RMSE
- near-wall `55..60N` bias and RMSE
- KE and max|u| behavior
- 365d stability

## Decision rules

1. If lower `nu_h` improves the target region and remains stable, the model is
   over-dissipative and boundary-current sharpening is a useful lever.
2. If lower `nu_h` destabilizes, retain production viscosity and test a
   resolution or advection-scheme change instead.
3. If lower `nu_h` does not improve the cold bias, the remaining error is more
   likely limited by resolution or missing boundary-current physics, not by a
   simple viscosity scalar.
