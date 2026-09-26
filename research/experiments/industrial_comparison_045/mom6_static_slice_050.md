# MOM6 0.5-degree static slice: initialization only

Date: 2026-09-26  
Status: grid/topography/initial state readable; dynamic smoke blocked by NaN; not comparable

## What was prepared

Using the same ocean_solver 0.5-degree grid (720x260x14, lat 65N cap,
80 bathymetry smoothing passes, 500m depth floor):

- external directory: `C:/Users/zhen.luo/external_models/mom6_slice_050`
- `INPUT/topog_050.nc` from the same ETOPO 2022 bathymetry and mask
- `INPUT/woa_ts_050.nc` from the same WOA T/S initial fields
- a `MOM_input` with `GRID_CONFIG=spherical`, 720x260, NK=14, flat file topography

MOM6 successfully read this bathymetry and initial state and printed global
mass/salt/heat consistent with initialization.

## Current blocker

The one-day smoke run reaches the first step but aborts on NaN in the net-input
EFP sum under `WIND_CONFIG=zero` and `BUOY_CONFIG=zero/const`.  This is a
configuration/forcing issue, not a grid-format blocker.

## Why this is not a benchmark

There is still no matched forcing or sea-ice treatment.  Do not use this as an
MOM6 score.  The next step is a shared forcing conversion and matching
boundary-condition contract.
