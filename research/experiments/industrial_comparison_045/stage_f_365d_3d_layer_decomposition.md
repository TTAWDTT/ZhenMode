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

## Latitude-band decomposition (final-90d annual 3D)

| band | RMSE | bias |
|---|---:|---:|
| 60--40S | 0.919 C | -0.296 C |
| 40--20S | 1.743 C | -1.146 C |
| 20--0S | 1.633 C | -0.625 C |
| 0--20N | 1.680 C | -0.475 C |
| 20--40N | 1.978 C | -1.230 C |
| 40--60N | 1.370 C | -0.170 C |

The largest annual 3D errors are in the southern subtropics and northern
subtropics/extratropics, not only the high-latitude wall. This reinforces the
upper-ocean/ventilation interpretation and argues against a high-latitude-only
closure.
