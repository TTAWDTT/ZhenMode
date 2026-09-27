# Stage-F 30d band-limited dynamic-ice gate

Same 0.5-degree Stage-F dynamic-bulk slice. All 3D numbers below use the
protocol's final-10d window; the older full-ice 30d file that used the whole
30d average is not comparable and is kept only as a diagnostic.

| run | global A2 | NA RMSE | near-wall RMSE | global 3D RMSE | NA 3D RMSE | near-wall 3D RMSE | mean MLD | MLD RMSE | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Stage-F no-ice control | 1.691 C | 2.088 C | 1.958 C | 0.861 C | 1.175 C | 1.125 C | 68.2 m | 271.2 m | PASS |
| Stage-F full dynamic ice, final 10d | 1.689 C | 2.088 C | 1.958 C | 0.858 C | 1.175 C | 1.123 C | 68.6 m | 274.1 m | PASS; not promoted |
| Stage-F dynamic ice, 40--65N only | 1.689 C | 2.088 C | 1.958 C | 0.858 C | 1.175 C | 1.123 C | 68.6 m | 274.1 m | PASS; not promoted |
| Stage-F full dynamic ice, whole-30d diagnostic | n/a | n/a | n/a | 0.707 C | n/a | n/a | 54.2 m | 194.0 m | not protocol-comparable |

Interpretation:
- On the protocol's final-10d window, north-only dynamic ice is essentially
  equivalent to full dynamic ice; neither improves the no-ice control.
- The apparently stronger full-ice result came from averaging the whole 30d
  window, not from the final-10d climate gate.
- This removes the earlier "north-only loses the 3D/MLD gain" claim. The next
  closure still needs band/ice-state dependence, but not because of this probe.

Sources:
- `ocean_solver_stage_f_30d_3d_benchmark.json`
- `ocean_solver_stage_f_dyn_ice_mld_diag_30d_final10d_3d_benchmark.json`
- `ocean_solver_stage_f_dyn_ice_lat40_65_probe_30d_benchmark.json`
- `ocean_solver_stage_f_dyn_ice_lat40_65_probe_30d_3d_benchmark.json`
