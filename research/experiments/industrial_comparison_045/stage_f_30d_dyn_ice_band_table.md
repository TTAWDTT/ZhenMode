# Stage-F 30d band-limited dynamic-ice gate

Same 0.5-degree Stage-F dynamic-bulk slice, 30d window. The new run gates the
minimal dynamic-ice closure to `40--65N`; it does **not** change mixed-layer
depth or any other closure parameter. This is a diagnostic ablation, not a
promoted baseline.

| run | global A2 | NA RMSE | near-wall RMSE | global 3D RMSE | NA 3D RMSE | near-wall 3D RMSE | mean MLD | MLD RMSE | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Stage-F no-ice control | 1.691 C | 2.088 C | 1.958 C | 0.861 C | 1.175 C | 1.125 C | 68.2 m | 271.2 m | PASS |
| Stage-F full dynamic ice | 1.689 C | 2.088 C | 1.958 C | 0.707 C | 0.961 C | 0.911 C | 54.2 m | 194.0 m | PASS; not annualized as production default |
| Stage-F dynamic ice, 40--65N only | 1.689 C | 2.088 C | 1.958 C | 0.858 C | 1.175 C | 1.123 C | 68.6 m | 274.1 m | PASS; not promoted |

Interpretation:
- On the 30d slice, limiting dynamic ice to 40--65N removes most of the
  3D/MLD gain seen in the full dynamic-ice diagnostic, so the southern
  high-latitude response is not redundant.
- The 365d full dynamic-ice run still fails the annual 3D/MLD gate, so neither
  full nor north-only dynamic ice is promoted.
- The useful next experiment is not another latitude cut; it is a band/ice-state
  dependent ventilation closure, with separate northern and southern behavior.

Sources:
- `ocean_solver_stage_f_30d_3d_benchmark.json`
- `ocean_solver_stage_f_dyn_ice_mld_diag_30d_3d_benchmark.json`
- `ocean_solver_stage_f_dyn_ice_lat40_65_probe_30d_benchmark.json`
- `ocean_solver_stage_f_dyn_ice_lat40_65_probe_30d_3d_benchmark.json`
