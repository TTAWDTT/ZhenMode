# Standardized external-model benchmark protocol v2

Date: 2026-09-26  
Status: active  
Purpose: turn one-off external runs into reproducible, gate-controlled comparison
ladder that can eventually support the claim “competitive on a defined slice”,
without overclaiming global superiority.

## Fixed benchmark contract

Every external comparison run must record these fields:

| Item | Required value |
|---|---|
| grid | shared 720x260 0.5° global lat-lon slice, `lat=-65..65`, `lon=0..360` |
| vertical grid | documented per model; ocean_solver uses 14 z-levels, MOM6 uses 14-level Z* ALE |
| bathymetry | ETOPO2022-derived shared topography, 80 smoothing passes, 500 m floor |
| initial state | WOA T/S |
| wind | 2023 monthly NCEP R1 10m wind converted to stress |
| heat/salt forcing | stage-specific, see ladder |
| sea ice | stage-specific; report exact treatment |
| duration | 30d stability, then 365d climate if gates pass |
| reference | WOA top-level SST on the same grid |
| scoring window | final 10d for 30d runs; final 90d for 365d runs |
| masks | `wet_mask` intersection and exact grid equality check |
| reproducibility | command, commit, wall time, hardware, repeat policy |

A missing contract item is reported as `not_comparable`; it must not be silently
dropped.

## Runtime provenance contract

For every new ocean_solver run, the NPZ must carry the exact runtime
environment in its config payload:

- git_commit
- git_dirty
- python_version
- 
umpy_version
- jax_version
- jax_devices

enchmark_manifest.py may derive commit from this field when the caller
does not pass --commit. A manifest is still allowed to be

ot_comparable, but it must say so explicitly; provenance is not silently
dropped.

## Stage ladder

### Stage W — wind-only dynamic control

Purpose: isolate dynamical cores with no thermodynamic feedback.

- heat/salt flux: none
- sea ice: none
- completed for 30d
- output: `wind_only_30d_manifest.json`

This is the minimum external-model sanity check.  It is **not** a climate
benchmark.

### Stage R — prescribed SST/SSS restoring

Goal: make a climate-like boundary condition reproducible in both models without
yet adding full bulk flux complexity.

Required inputs:

1. same wind stress;
2. same WOA top-level SST restore target;
3. same restoring timescale, initially 30 days;
4. same bathymetry, mask and initial state;
5. no sea-ice closure unless both models use the same closure;
6. 30d and then 365d.

Gate:

- both models `PASS`;
- heat/salt drift bounded;
- both scored by `score_external_model.py` or `benchmark_metrics.py`.

The 30d prescribed-restore run is completed and recorded in
`restore_30d_manifest.json`. It validates this stage on the matched 30d slice,
but a 365d reproducible run is still required before any climate-state claim.

### Stage F — matched bulk heat/salt forcing

Goal: move from prescribed restoring to the realistic air-sea flux contract.

Required forcing fields, in priority order:

1. 2m air temperature and humidity;
2. shortwave/longwave radiation;
3. precipitation/evaporation;
4. wind stress;
5. runoff only if both models can consume the same file.

No winner table is allowed until all of these are either matched or explicitly
disabled in both models.

### Stage I — sea-ice / mixed-layer minimal closed loop

The minimum loop is:

```text
atmospheric forcing
  -> mixed-layer heat capacity
  -> freezing / melting
  -> ocean heat + salt flux
  -> SST/MLD response
```

Current solver implementation has:

- stateful ice thickness;
- latent growth/melt;
- ice conductivity insulation proxy;
- brine salt flux;
- mixed-layer heat capacity;
- optional stratification-derived MLD.

A run may be called a **minimum closed loop** only when all of these are
recorded in one manifest:

1. ice thickness/extent;
2. growth/melt heat flux;
3. brine salt flux;
4. effective mixed-layer depth;
5. surface heat-budget residual;
6. stability verdict;
7. same-grid climate score.

It may not be called a full sea-ice model.  It is a coupled thermodynamic proxy
for the eventual industrial comparison.

## Stage-F exact dynamic-bulk MOM6 configuration

The prescribed `lambda*(air-WOA SST)` file is a proxy and cannot be used for a
climate comparison because it lacks the live SST negative feedback.  For the
matched Stage-F 30d/365d bulk control, MOM6 must use the live SST form:

- `RESTOREBUOY = True`;
- `VARIABLE_BUOYFORCE = True`;
- target SST field = 2023 monthly NCEP R1 2m air temperature;
- `FLUXCONST_T = 1.7010 m/day` with `RESTORE_FLUX_RHO=1035`, `C_p=3925`,
  giving 80 W/m2/K;
- `FLUXCONST_S = 0`;
- prescribed sensible/latent/longwave/shortwave/precip/runoff = zero.

Keep the run on local Linux storage.  The C: filesystem is full and caused the
earlier proxy run to lose its final spatial output.

## Standardized 3D latitude-depth and MLD metrics

Any 3D temperature comparison should also report the standardized
latitude-band x depth matrix produced by the shared scorer. The default bands
are `60--40S`, `40--20S`, `20--0`, `0--20N`, `20--40N`, and `40--60N`. This
matrix is the primary diagnostic for separating wall-region errors from broad
subtropical upper-ocean errors.

When both temperature and salinity are available, the scorer must also report
mixed-layer-depth bias/RMSE using the same density-threshold definition and the
same WOA reference field. The 3D MLD metric is diagnostic, not a substitute for
SST skill; it exists to tell us whether a closure changes ventilation.

For ocean_solver snapshots, the same report must also include
`mld_latitude_bands` using the standardized latitude bands above. This is
required because a global MLD number can hide opposite high-latitude and
subtropical errors.


## 3D scoring requirement

Any 3D profile comparison must apply the shared bathymetric vertical mask:

- solver snapshots: use `--depth-file <ocean_geometry.nc> --depth-var D`, or
  store/read `wet_mask_z` in the run NPZ;
- MOM6/external NetCDF: use `--depth-var D`;
- never score a level just because the horizontal column is wet.

The initial bug counted below-seafloor ghost layers and falsely implied a
4000m warm bias of +6.35 C. The scorer now reports `depth_mask_applied`; without the shared bathymetric mask it returns `FAIL` instead of silently producing a misleading PASS.
For a multi-snapshot 3D window, use:

```bash
python src/score_solver_3d.py \
  --npz <run.npz> \
  --snap-dir <run>_3d \
  --snap-days 10 --steady-days 90 \
  --out <solver_3d_benchmark.json>
```
## Scoring commands

Internal run:

```bash
python src/benchmark_metrics.py \
  --npz results/industrial_comparison_045/global_<run>.npz \
  --out research/experiments/industrial_comparison_045/<run>_benchmark.json \
  --steady-days 10
```

External structured NetCDF:

```bash
python src/score_external_model.py \
  --input <model_output.nc> \
  --variable <surface_temp> \
  --geometry <grid_or_mask_file.nc> \
  --wet-var <wet_mask> \
  --lat-var lat --lon-var lon \
  --reference-npz results/industrial_comparison_045/global_<reference_run>.npz \
  --steady-days 10 \
  --model <MODEL> \
  --run-id <run> \
  --stats <conservation_or_stats_file> \
  --out research/experiments/industrial_comparison_045/<run>_benchmark.json
```

Comparison:

```bash
python src/benchmark_table.py \
  <ocean_solver_benchmark.json> <external_benchmark.json> \
  --label ocean_solver=<ocean_solver.json> \
  --label <MODEL>=<external.json> \
  --out <comparison>.md
```

## Promotion rules

A result may be promoted only when all of the following are true:

1. matched grid, bathymetry, initial state, wind, forcing, duration, reference;
2. both runs `PASS` the stability watchdog;
3. heat and salt drift are bounded;
4. global A2 is not worse than the pre-registered control;
5. NA 40--60N RMSE is not worse;
6. near-wall raw bias is not worse;
7. the run is reproduced to numerical precision;
8. ice/MLD diagnostics are reported and not retrofitted.

A 30d result can promote a **protocol**, not a climate claim.

## What may be claimed after Stage W

The current 30d wind-only comparison supports only these statements:

- both runs completed;
- on this wind-only dynamic slice, ocean_solver has lower global A2 and similar
  North Atlantic RMSE;
- MOM6 is slower on the chosen local 4-rank CPU configuration.

It does **not** establish superiority as a global climate model, sea-ice model,
or industrial production system.




## Annual 3D direct comparison

For the annual Stage-F direct comparison, use the same final-90d window on both
models.

ocean_solver:

```bash
python src/score_solver_3d.py \
  --npz results/industrial_comparison_045/global_global_industrial_comparison_050_stage_f_365d_3d.npz \
  --snap-dir /root/external_models/results/industrial_comparison_045/global_global_industrial_comparison_050_stage_f_365d_3d_3d \
  --snap-days 30 --steady-days 90 \
  --out research/experiments/industrial_comparison_045/ocean_solver_stage_f_365d_3d_benchmark.json
```

MOM6:

```bash
python src/score_external_3d.py \
  --input <run_dir>/prog.nc \
  --variable temp \
  --geometry <run_dir>/ocean_geometry.nc \
  --wet-var wet \
  --lat-var lath \
  --lon-var lonh \
  --depth-var D \
  --salt-variable salt \
  --reference-npz /root/external_models/results/industrial_comparison_045/global_global_industrial_comparison_050_stage_f_365d_3d.npz \
  --steady-days 90 \
  --out research/experiments/industrial_comparison_045/mom6_stage_f_dynamic_365d_3d_benchmark.json
```

A 30d result may diagnose a scoring bug or short-term behavior, but only the
annual final-90d window is the pre-registered annual climate gate.




## Contract validation

Before scoring, run the contract validator on every standardized manifest:

python src/benchmark_contract.py manifest.json

contract_pass=true means every required section is present and every section has
an explicit comparability status. A section may still be not_comparable; this is
recorded rather than silently dropped from the comparison.


## Annual candidate gate helper

For annual candidate acceptance, call the gate with the paired 3D/MLD JSON as
well as the paired surface JSON. The optional --control-3d and
--experiment-3d inputs add global/NA/near-wall 3D RMSE and MLD bias/RMSE
checks. If only one 3D file is supplied, the gate is intentionally incomplete.


Both the contract validator and the annual gate CLI now exit nonzero on a failed
check, so they can be used directly as CI or shell-script gates.


For annual 3D/MLD gates, pass --require-3d so the gate cannot accidentally
pass on surface metrics alone.
