# MOM6 0.5° global ALE dynamic smoke

Date: 2026-09-26  
Status: matched 30d wind-only comparison completed  
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

Both runs use the same 720x260 grid, WOA T/S, ETOPO-derived bathymetry,
2023 monthly NCEP wind, 30d duration, and no heat/salt forcing.

| model | days | verdict | global A2 RMSE | global raw bias/RMSE | NA 40--60N RMSE | near-wall 55--60N bias/RMSE | wall time |
|---|---:|---|---:|---:|---:|---:|---:|
| ocean_solver | 30 | PASS | 0.8658 C | -0.0580 / 0.5660 C | 0.6751 C | -0.1442 / 0.2106 C | 3.4 min |
| MOM6 | 30 | PASS | 1.0961 C | -0.4070 / 1.1827 C | 0.6814 C | +0.0567 / 0.2434 C | 49.6 min (4 ranks) |

MOM6 files:

- `research/experiments/industrial_comparison_045/mom6_wind_only_30d_benchmark.json`
- `research/experiments/industrial_comparison_045/wind_only_30d_comparison_table.md`

## Comparability caveats

This is a wind-forced dynamic-control slice, not a full climate benchmark.
It shares grid, bathymetry, initial T/S, wind, duration, and masks, but omits
air-sea heat/salt fluxes. Therefore it must not be called a climate-state
comparison. The next ladder is:

1. score MOM6 SST on the shared grid,
2. produce a same-grid comparison table,
3. then add the matched bulk heat/salt or prescribed-restoring protocol.
