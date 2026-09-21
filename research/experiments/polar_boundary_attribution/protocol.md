# Polar / Boundary Attribution Protocol

Status: locked
Date: 2026-09-21
Primary data: existing 1-degree, 365-day centered / monotone / FCT runs

## Question

Why does the current A2 SST climate score fail even though tracer-transport
choice changes it almost not at all?

The working hypothesis from the previous experiment was that the dominant
error is concentrated near the polar cap, northern boundary, coast, or
shallow bathymetry. This experiment tests that attribution before adding new
physics or running more long integrations.

## Existing outputs

- `results/fct_transport/clim_centered/climatology_compare_g.npz`
- `results/fct_transport/clim_monotone/climatology_compare_g.npz`
- `results/fct_transport/clim_fct/climatology_compare_g.npz`
- `results/diagnostics_budget/global_fct_1deg_365d_budget.npz`

The FCT run is the primary case; centered and monotone are controls.

## Attribution variables

For every wet ocean cell, classify by:

1. **Latitude band**
   - 60S--40S, 40S--20S, 20S--0, 0--20N, 20N--40N, 40N--60N
2. **Distance to land**
   - in grid cells, periodic in longitude
   - bins: 1--3, 3--6, 6--11, >=11 cells
3. **Distance to closed meridional wall**
   - bins: 0--1, 1--3, 3--6, 6--11, >=11 cells
4. **Smoothed ETOPO depth**
   - 0--200, 200--1000, 1000--3000, >=3000 m
5. **Longitude sector**
   - 60-degree bins

## Metrics

For a cell mask `m`, report:

- count
- mean error
- RMSE within group
- maximum absolute error
- unweighted share of global A2 SSE: `sum(error[m]^2) / sum(error[wet]^2)`
- hypothetical global RMSE if that group's error were zero

The hypothetical global RMSE is diagnostic only. It says which group controls
the score; it is not a substitute for a model improvement.

## Pre-registered decision rule

After computing the tables:

1. If coastal/shallow cells account for a majority of SSE, prioritize coast
   and bathymetry sensitivity runs.
2. If polar or northern-boundary rows account for a majority of SSE, run
   polar-cap and boundary sensitivity runs.
3. If deep, open-ocean cells account for the majority of SSE, do **not**
   continue boundary-only fixes. Move to surface forcing, vertical mixing,
   or a masked climatological-error decomposition.
4. Transport should not be retested unless the grouped result differs strongly
   between centered, monotone, and FCT.

## Transferable lesson from mature OGCMs

MOM6, NEMO, ROMS, FESOM2, ICON-Ocean, MPAS-Ocean and HYCOM all emphasize
diagnosing errors against masked, area-consistent climate fields rather than
tuning one maximum error location. Before adding a closure or changing the
grid, the error budget should identify where the squared error actually lives.

## Non-goals

- No new 365-day run in the first pass.
- No transport-scheme changes.
- No neural-PDE extension.
- No unstructured-mesh rewrite.
