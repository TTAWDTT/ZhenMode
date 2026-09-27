# Stage-F 365d band-gated dynamic ice + fixed MLD

Updated: 2026-09-27 14:25 local. All values use the protocol final-90d window
on the same 0.5-degree Stage-F dynamic-bulk slice.

| run | global A2 | NA RMSE | near-wall RMSE | global 3D | NA 3D | near-wall 3D | MLD bias | MLD RMSE | salt drift | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| no-ice control | 1.204 C | 0.975 C | 1.055 C | 1.547 C | 1.276 C | 1.124 C | +139.3 m | 655.7 m | -0.00038% | PASS |
| dynamic ice 40--65N + fixed 100m MLD 40--60N | 4.759 C | 8.434 C | 11.949 C | 4.042 C | 6.860 C | 9.697 C | +81.2 m | 464.6 m | -0.1497% | FAIL |

The annual run is stable (`PASS` in the solver watchdog) but fails the climate
gate on global A2, North Atlantic RMSE, 3D temperature, and salt drift.  The
improved MLD bias is not useful because the temperature field is grossly too
warm in the North Atlantic and near-wall regions.  This rejects annualizing the
otherwise attractive 30d candidate.

Conclusion:
- do **not** promote the band-ice + fixed-depth closure;
- do not use the 30d gain as evidence for a seasonal climate skill;
- keep the annual no-ice Stage-F control as the current internal baseline;
- wait for the MOM6 annual 3D score before choosing the next closure.

Sources:
- `ocean_solver_stage_f_365d_3d_surface_benchmark.json`
- `ocean_solver_stage_f_365d_3d_benchmark.json`
- `ocean_solver_stage_f_band_mld_ice_365d_benchmark.json`
- `ocean_solver_stage_f_band_mld_ice_365d_3d_benchmark.json`
- `stage_f_365d_band_mld_ice_gate.json`
