# Stage-F 365d band-gated dynamic ice + fixed MLD

Updated: 2026-09-27 15:15 local.

The first annual run (`v1`) was invalid because it used a different bathymetry.
`v2` uses the same `67.9%` ocean bathymetry as the annual no-ice control and is
therefore the valid annual gate.

| run | global A2 | NA RMSE | near-wall RMSE | global 3D | NA 3D | near-wall 3D | MLD bias | MLD RMSE | NA MLD bias | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| no-ice control | 1.204 C | 0.975 C | 1.055 C | 1.547 C | 1.276 C | 1.124 C | +139.3 m | 655.7 m | +224.6 m | PASS |
| band ice 40--65N + fixed 100m MLD 40--60N | 1.246 C | 1.495 C | 1.260 C | 1.559 C | 1.626 C | 1.387 C | +121.2 m | 618.2 m | +17.2 m | FAIL |

Interpretation:
- stability passes, salt drift is bounded, and MLD improves substantially;
- but global A2 and North Atlantic RMSE get worse;
- the annual 3D field is slightly worse globally and clearly worse in
  North Atlantic / near-wall regions;
- this is a useful diagnostic, but not a promoted baseline.

Conclusion:
- do **not** promote the fixed-depth band-ice closure;
- keep the 365d Stage-F no-ice control as the internal baseline;
- keep the 30d result as a short-window diagnostic only;
- wait for MOM6 v11 before choosing the next physical closure.

Sources:
- `ocean_solver_stage_f_365d_3d_surface_benchmark.json`
- `ocean_solver_stage_f_365d_3d_benchmark.json`
- `ocean_solver_stage_f_band_mld_ice_365d_v2_benchmark.json`
- `ocean_solver_stage_f_band_mld_ice_365d_v2_3d_benchmark.json`
- `stage_f_365d_band_mld_ice_v2_gate.json`
