# MOM6 Stage-F exact dynamic-bulk 365d v2

Date: 2026-09-27  
Run directory: `/root/external_models/mom6_slice_050/p0_stage_f_dynamic_365d_v2`  
Status: running on local Linux storage.

## Purpose

Provide the annual direct counterpart to the completed ocean_solver Stage-F
365d 3D control without changing the Stage-F scientific contract.

## Why v2

The first 365d launch exited during startup. The v2 run keeps the validated
30d bulk configuration and changes only:

- `ocean_solo_nml:days = 365`
- diagnostic output interval: 10 days
- `MAXCPU = 2.88e5 s` (diagnostic runtime guard)

It deliberately retains the tested `DAYMAX = 3.0` setting from the successful
30d control; `ocean_solo_nml:days` controls the actual annual segment length.
This is an operational workaround, not evidence that the first configuration
was scientifically wrong.

## Scoring

When complete, score the final 90d window with:

```bash
python src/score_external_3d.py \
  --input /root/external_models/mom6_slice_050/p0_stage_f_dynamic_365d_v2/prog.nc \
  --variable temp \
  --geometry /root/external_models/mom6_slice_050/p0_stage_f_dynamic_365d_v2/ocean_geometry.nc \
  --wet-var wet \
  --lat-var lath \
  --lon-var lonh \
  --depth-var D \
  --reference-npz /root/external_models/results/industrial_comparison_045/global_global_industrial_comparison_050_stage_f_365d_3d.npz \
  --steady-days 90 \
  --out research/experiments/industrial_comparison_045/mom6_stage_f_dynamic_365d_3d_benchmark.json
```
