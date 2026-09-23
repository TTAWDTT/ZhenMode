# 0.45-Degree Ice-Floor Candidate

Date: 2026-09-24  
Status: consolidated

## New baseline

`candidate_65n_045_icefloor` is now the production-like candidate:

- 0.45-degree grid, `dt=1800s`, `nu_h=2e6`
- lambda80, GM0, localized convection, FCT transport
- annual real 2m air forcing with a `-1.8 C` ice-air floor
- global A2 RMSE: `1.1126 C`
- North Atlantic A2 RMSE: `1.0096 C`
- near-wall raw bias: `-0.9121 C`
- sub-freezing SST cells: `0`
- runner: `scripts/run_candidate_baseline_045_icefloor.sh`

The 365d repeat matches to displayed precision.

## Why it is physical but simple

The old 0.45-degree candidate had 3030 cells below the nominal freezing point.
WOA does not. The proxy stops the bulk target from forcing open water below
freezing. It does not represent ice transport, thickness, brine rejection, or
albedo, so it is not a full sea-ice model.

## Fallbacks

- no-proxy 0.45 candidate: `candidate_65n_045_gm0`
- reproduced 0.50-degree candidate: `candidate_65n_05_gm0`
- older 0.70-degree candidate: `candidate_65n_07_gm0`

## Next

Use the ice-floor baseline as the fixed A/B reference and re-score remaining
error bands. Do not add another closure until the next error attribution is
complete.
