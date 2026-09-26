# MOM6 direct-slice runbook

Status: build/smoke complete; slice not configured  
Target: `research/experiments/industrial_comparison_045/slice.yaml`

## Principle

MOM6 is not vendored into this repository.  Clone and build it in a separate
working directory, then copy only the scored metrics, manifest, and configuration
back into this project.

## Setup checklist

1. Record the exact MOM6 commit and build compiler/library versions.
2. Build a 0.5-degree global ocean-only configuration.
3. Use the same ETOPO 2022 bathymetry and WOA T/S initial state as ocean_solver.
4. Use the same NCEP R1 annual 2m air and seasonal 2023 NCEP wind as the current
   ocean_solver slice, or rerun both models with the same OMIP forcing.
5. Use the same wet-mask and minimum-depth rule.
6. Integrate 365d.
7. Regrid the MOM6 SST output to the ocean_solver 0.5-degree reference grid.
8. Score with `src/benchmark_metrics.py` or the external-model equivalent.
9. Write a MOM6 benchmark manifest with `--model MOM6`.
10. Compare using `src/benchmark_table.py`.

## Required output

For MOM6, save:

- `mom6_config.json`
- `mom6_sst_365d.npz`
- `mom6_benchmark_365d.json`
- `mom6_manifest_365d.json`

Only these artifacts should enter this repository.  Large raw MOM6 outputs stay
outside.

## Gate

Do not rank ocean_solver against MOM6 until both runs have the same initial
state, forcing, bathymetry, mask, duration, and reference.  Any missing field
must be recorded as `not_comparable`, not silently omitted.
