# Stage-F annual 3D error layer breakdown

Same 0.5-degree slice, shared WOA reference and bathymetric mask.

| depth | 30d ocean_solver RMSE | 365d ocean_solver RMSE |
|---|---:|---:|
| 0 m | 1.703 C | 1.142 C |
| 5 m | 1.448 C | 1.690 C |
| 15 m | 1.296 C | 2.075 C |
| 30 m | 1.007 C | 2.191 C |
| 50 m | 0.668 C | 2.023 C |
| 75 m | 0.617 C | 1.796 C |
| 100 m | 0.647 C | 1.744 C |
| 150 m | 0.610 C | 1.652 C |
| 200 m | 0.520 C | 1.614 C |
| 300 m | 0.418 C | 1.378 C |
| 500 m | 0.372 C | 1.021 C |
| 1000 m | 0.206 C | 0.384 C |
| 2000 m | 0.103 C | 0.191 C |
| 4000 m | 0.070 C | 0.124 C |

Interpretation: the annual skill loss is not a deep-ocean artifact. It is a
broad subsurface warming/cooling signal that grows through the upper 500 m.
This points to annual mixed-layer/ventilation and seasonal heat storage, not
just surface forcing.

Source:
- 30d: `ocean_solver_stage_f_30d_3d_benchmark.json`
- 365d: `ocean_solver_stage_f_365d_3d_benchmark.json`
