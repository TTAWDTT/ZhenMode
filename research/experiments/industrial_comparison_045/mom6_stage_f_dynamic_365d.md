# MOM6 Stage-F exact dynamic-bulk 365d run

Date: 2026-09-26  
Run directory: `/root/external_models/mom6_slice_050/p0_stage_f_dynamic_365d`  
Status: running on local Linux storage

## Contract

Same as the 30d exact dynamic-bulk slice, but for 365d and with daily
diagnostics. The configuration is the direct annual counterpart to:

- `results/industrial_comparison_045/global_industrial_comparison_050_stage_f_365d.npz`
- `research/experiments/industrial_comparison_045/ocean_solver_stage_f_365d_benchmark.json` (after scoring)

## Scoring command

```bash
python src/score_external_model.py \
  --input <run_dir>/prog.nc \
  --variable temp \
  --reference-npz results/industrial_comparison_045/global_industrial_comparison_050_stage_f_365d.npz \
  --geometry <run_dir>/ocean_geometry.nc \
  --wet-var wet --lat-var geolat --lon-var geolon \
  --level 0 --steady-days 90 \
  --model MOM6 --run-id mom6_stage_f_dynamic_365d \
  --stats <run_dir>/ocean.stats.nc \
  --out research/experiments/industrial_comparison_045/mom6_stage_f_dynamic_365d_benchmark.json
```
