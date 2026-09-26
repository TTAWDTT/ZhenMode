# MOM6 0.5° global ALE dynamic smoke

Date: 2026-09-26  
Status: dynamic smoke passed; matched 30d comparison running  
Run directory: `C:/Users/zhen.luo/external_models/mom6_slice_050/p0_shared_ale`

## Fixed relative to the first `p0_shared` attempt

1. `domains_stack_size` was raised from 810000 to 2000000.
2. The unnecessary `SAVE_INITIAL_CONDS` write was disabled.
3. The grid was corrected to the shared ocean_solver 0.5° global slice:
   `720 x 260`, `LAT -65..65`, `LON 0..360`, `EQUATOR_REFERENCE=False`.
4. MOM6 now uses a 14-level Z* ALE coordinate file, not 75 benchmark layers.
5. Initial T/S now reads the shared WOA file `woa_ts_050.nc`.
6. `BUOY_CONFIG=NONE` disables heat/salt forcing in this first dynamic control.
7. Shared 2023 monthly NCEP wind stress is exported to an A-grid MOM6 file.

## Export tooling

`src/export_mom6_wind_forcing.py` builds the same monthly NCEP wind stress
used by ocean_solver and writes `STRESS_X` / `STRESS_Y` on the A grid. The
file uses an unlimited time dimension and 31-day records.

## One-day smoke result

- MOM6 commit: `f49a00096` (recorded in earlier build note)
- grid: `720 x 260 x 14`
- vertical coordinate: Z* ALE, interfaces `0,5,15,30,50,75,100,150,200,300,500,1000,2000,3000,4000 m`
- bathymetry: shared ETOPO-derived `topog_050.nc`
- initial T/S: shared WOA `woa_ts_050.nc`
- forcing: 2023 monthly NCEP wind; no buoyancy forcing
- MPI: 1 rank
- one-day wall time: 332 s
- MaxCFL: 0.0172
- heat/salt drift over 1d: effectively zero
- result: completed, no NaN, no instability

A one-day 4-rank run reproduced the same state and stayed stable.

## 30-day comparison

- ocean_solver wind-only 30d: PASS in 3.4 min
  - global A2 `0.8658 C`
  - raw global bias/RMSE `-0.0580 / 0.5660 C`
  - NA 40--60N RMSE `0.6751 C`
  - near-wall 55--60N bias `-0.1442 C`
  - heat drift `-0.0262%`
- MOM6 30d: running with 4 MPI ranks
- status: `not comparable` until the MOM6 run finishes and both models are scored
  with the same mask and reference.

## Comparability caveats

This is a wind-forced dynamic-control slice, not a full climate benchmark.
It shares grid, bathymetry, initial T/S, wind, duration, and masks, but omits
air-sea heat/salt fluxes. Therefore it must not be called a climate-state
comparison. The next ladder is:

1. score MOM6 SST on the shared grid,
2. produce a same-grid comparison table,
3. then add the matched bulk heat/salt or prescribed-restoring protocol.
