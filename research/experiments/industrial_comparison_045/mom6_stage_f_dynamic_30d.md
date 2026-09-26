# MOM6 Stage-F 30d exact dynamic-bulk slice

Date: 2026-09-26  
Run directory: `/root/external_models/mom6_slice_050/p0_stage_f_dynamic_30d` (WSL-local; `/mnt/c` is full)  
Status: 1d smoke passed; 30d exact-bulk run launched

## Why this is the matched forcing

The previous MOM6 "sensible proxy" was a prescribed monthly file and therefore not
comparable to ocean_solver's live bulk term.  This run uses MOM6 restoring machinery
as a bulk closure:

- target = 2023 monthly NCEP R1 2m air temperature;
- heat coefficient = `1035 kg/m3 * 3925 J/kg/K * 1.7010 m/day = 80.0 W/m2/K`;
- salt coefficient = 0;
- no prescribed sensible heat, longwave, shortwave, evaporation, precipitation, or runoff;
- same shared grid, bathymetry, initial state, and wind as protocol v2.

## Contract

- Grid: 720x260, lat -65..65, lon 0..360; 14-level Z* ALE.
- Initial state: shared WOA T/S.
- Bathymetry: shared ETOPO-derived 500 m-floor bathymetry.
- Wind: 2023 monthly NCEP stress.
- Heat: `q = lambda*(air - live SST)`, lambda 80 W/m2/K.
- Salt: none.
- Duration: 30d; score final 10d with `score_external_model.py`.

## Smoke result

- 1d, 4 MPI ranks, local WSL filesystem: passed.
- Wall time: 210.4 s.
- No truncation or NetCDF I/O error; `prog.nc` and restart were written.

## Note on the previous proxy run

The prescribed-proxy run reached day 30 with a stable logged mean temperature,
but its final spatial NetCDF/restart write failed because `/mnt/c` was full.  Thus
it has no valid spatial benchmark.  It remains a stability-only warning, not a
comparison table entry.
