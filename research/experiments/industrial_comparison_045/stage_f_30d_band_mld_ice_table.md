# Stage-F 30d band-gated dynamic ice + fixed MLD

Updated: 2026-09-27 13:48 local. All 3D values use the protocol final-10d
window on the same 0.5-degree Stage-F dynamic-bulk slice.

| run | global A2 | NA RMSE | near-wall RMSE | global 3D | NA 3D | near-wall 3D | MLD bias | MLD RMSE | NA MLD bias | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| no-ice control | 1.691 C | 2.088 C | 1.958 C | 0.861 C | 1.175 C | 1.125 C | +36.7 m | 271.2 m | +260.6 m | PASS |
| full dynamic ice | 1.689 C | 2.088 C | 1.958 C | 0.858 C | 1.175 C | 1.123 C | +37.0 m | 274.1 m | +264.8 m | PASS; not promoted |
| dynamic ice, 40--65N only | 1.689 C | 2.088 C | 1.958 C | 0.858 C | 1.175 C | 1.123 C | +37.0 m | 274.1 m | +264.8 m | PASS; not promoted |
| dynamic ice 40--65N + fixed 100m MLD 40--60N | 1.562 C | 0.848 C | 0.534 C | 0.777 C | 0.580 C | 0.311 C | +16.3 m | 151.5 m | +17.0 m | PASS; annualized then rejected |

The combined probe passes all pre-registered 30d gates: stability, global A2,
North Atlantic, near-wall, heat drift, and salt drift. It also improves the
3D temperature field and the standardized MLD diagnostic. The 30d result was not enough: the matching 365d final-90d check
failed badly and the closure is rejected. See
`stage_f_365d_band_mld_ice_table.md`.

Sources:
- `ocean_solver_stage_f_30d_benchmark.json`
- `ocean_solver_stage_f_30d_3d_benchmark.json`
- `ocean_solver_stage_f_dyn_ice_band40_65_mld100_lat40_60_probe_30d_benchmark.json`
- `ocean_solver_stage_f_dyn_ice_band40_65_mld100_lat40_60_probe_30d_3d_benchmark.json`
- `stage_f_30d_band_mld_ice_gate.json`



