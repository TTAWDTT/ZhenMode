# MOM6 Stage-F exact dynamic-bulk 365d v12

Date: 2026-09-27  
Run directory: /root/external_models/mom6_slice_050/p0_stage_f_dynamic_365d_v12  
Status: launched in tmux session mom6_v12 after v11 was killed before completion.

## Contract

Same 365d Stage-F dynamic-bulk run as v11, relaunched from a clean copy of the
v11 input directory. This is the direct industrial annual 3D comparison target.

## Scoring

When complete, use the standard final-90d window with:

    python src/score_external_3d.py \
      --input /root/external_models/mom6_slice_050/p0_stage_f_dynamic_365d_v12/prog.nc \
      --variable temp \
      --geometry /root/external_models/mom6_slice_050/p0_stage_f_dynamic_365d_v12/ocean_geometry.nc \
      --wet-var wet \
      --lat-var lath \
      --lon-var lonh \
      --depth-var D \
      --salt-variable salt \
      --reference-npz /root/external_models/results/industrial_comparison_045/global_global_industrial_comparison_050_stage_f_365d_3d.npz \
      --steady-days 90 \
      --out research/experiments/industrial_comparison_045/mom6_stage_f_dynamic_365d_v12_3d_benchmark.json

Do not claim industrial superiority before this annual score exists.


## Manifest

A standardized contract manifest is now recorded at:

research/experiments/industrial_comparison_045/mom6_stage_f_dynamic_365d_v12_manifest.json

The grid, bathymetry, initial state, wind, bulk heat/salt forcing, sea-ice treatment,
duration, reference field, scoring window, and masks are marked comparable. The MOM6
source commit and MPI version were not recorded, so exact code provenance is marked
not_comparable in the manifest. This does not block the numerical comparison, but it
does mean the result is not yet a fully reproducible source-level benchmark.
