# MOM6 build status

Date: 2026-09-26  
Status: built and smoke-passed; same-grid slice not yet configured

## Local build

External working directories (not vendored into this repository):

- MOM6: `C:/Users/zhen.luo/external_models/MOM6`
- MOM6 commit: `f49a00096df607b48354603e2398e14e189fd62e`
- FMS: `2023.03`, commit `527a42f3ff36d75aac68b65757b35f73a74146bb`
- MOM6-examples reference: commit `a5ebafec64c12f308d9f740ce40be1c1bfddd25a`
- Build: WSL, `gfortran 15.2.0`, OpenMPI 5.0.10, NetCDF-C/Fortran
- Executable: `C:/Users/zhen.luo/external_models/MOM6/build/MOM6`

## Smoke test

A 32x32 Cartesian benchmark configuration ran one MPI rank for 1 hour without
fatal errors.  This proves the executable links and initializes/advances, but it
is not a climate comparison.

## Next slice work

1. Create a shared 0.5-degree wet mask, bathymetry, WOA T/S initial state, and forcing NetCDF files.
2. Run ocean_solver and MOM6 on that same slice for 365d.
3. Remap MOM6 SST to the ocean_solver grid.
4. Score both with `src/benchmark_metrics.py` and record manifests.
5. Only then apply the pre-registered comparison table; do not rank before then.
