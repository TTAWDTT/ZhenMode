# Stage-F 365d band-gated dynamic ice + fixed MLD

Updated: 2026-09-27 14:35 local.

**Important:** the first annual run (`v1`) is **invalid**, not a physical
rejection.  Its log used a different bathymetry with `90.8%` ocean, while the
30d probe and annual no-ice control both use `67.9%` ocean.  We therefore
removed the incompatible run and relaunched the candidate as `v2` with the
matching bathymetry.  Do not use the v1 numbers as a climate gate.

| run | ocean fraction | status |
|---|---:|---|
| no-ice control | 67.9% | valid baseline |
| 30d band-ice + fixed-MLD probe | 67.9% | valid 30d gate PASS |
| 365d band-ice + fixed-MLD v1 | 90.8% | invalid bathymetry mismatch |
| 365d band-ice + fixed-MLD v2 | 67.9% | running; annual gate pending |

When v2 completes, score the protocol final-90d window for global A2, NA RMSE,
near-wall RMSE, global/NA 3D temperature, MLD bias/RMSE, and salt drift.

Sources:
- `ocean_solver_stage_f_365d_3d_surface_benchmark.json`
- `ocean_solver_stage_f_365d_3d_benchmark.json`
- `ocean_solver_stage_f_dyn_ice_band40_65_mld100_lat40_60_probe_30d_benchmark.json`
- `ocean_solver_stage_f_dyn_ice_band40_65_mld100_lat40_60_probe_30d_3d_benchmark.json`
- `stage_f_30d_band_mld_ice_gate.json`
- v1 diagnostic JSONs are retained only as an invalid-run audit trail.
