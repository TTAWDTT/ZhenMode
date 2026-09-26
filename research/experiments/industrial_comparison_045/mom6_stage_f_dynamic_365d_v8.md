# MOM6 Stage-F exact dynamic-bulk 365d v8

Date: 2026-09-27  
Run directory: `/root/external_models/mom6_slice_050/p0_stage_f_dynamic_365d_v8`  
Status: running on local Linux storage via `tmux`.

## Contract

Same as the 30d exact dynamic-bulk control, but for 365d and with daily
diagnostics. This is the annual counterpart to the completed ocean_solver
Stage-F 3D control.

## Ops note

The earlier annual launches failed because the WSL virtual disk was full. After
cleaning failed test runs and pip/uv caches inside WSL, the annual run started
normally from the validated 30d configuration with only `days = 365`.

## Scoring

When complete, score the final 90d window with:

```bash
python src/score_external_3d.py \
  --input /root/external_models/mom6_slice_050/p0_stage_f_dynamic_365d_v8/prog.nc \
  --variable temp \
  --geometry /root/external_models/mom6_slice_050/p0_stage_f_dynamic_365d_v8/ocean_geometry.nc \
  --wet-var wet \
  --lat-var lath \
  --lon-var lonh \
  --depth-var D \
  --reference-npz /root/external_models/results/industrial_comparison_045/global_global_industrial_comparison_050_stage_f_365d_3d.npz \
  --steady-days 90 \
  --out research/experiments/industrial_comparison_045/mom6_stage_f_dynamic_365d_3d_benchmark.json
```

