# Stage-F 30d direct 3D temperature comparison

Same 0.5-degree slice, final 10-day window, shared WOA reference, and shared
bathymetric vertical mask. The bathymetric mask is required: without it, ghost
layers below the seafloor were counted and created a false deep-ocean signal.

| model | global 3D RMSE | global 3D bias | 40-60N 3D RMSE | 55-60N 3D RMSE | surface RMSE | 4000m RMSE |
|---|---:|---:|---:|---:|---:|---:|
| MOM6 | 1.036 C | -0.193 C | 1.123 C | 1.095 C | 1.884 C | 0.035 C |
| ocean_solver | 0.861 C | -0.177 C | 1.175 C | 1.166 C | 1.703 C | 0.070 C |

Interpretation: with the corrected wet-column mask, ocean_solver has better
global and surface temperature, while MOM6 is modestly better in 40--60N and
near the 55--60N wall. There is no 4000m warm-bias signal once below-seafloor
ghost layers are excluded.
