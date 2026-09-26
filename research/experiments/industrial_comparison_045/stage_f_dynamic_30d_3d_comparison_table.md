# Stage-F 30d direct 3D temperature comparison

Same 0.5-degree slice, final 10-day window, and shared WOA reference.

| model | global 3D RMSE | global 3D bias | 40-60N 3D RMSE | 55-60N 3D RMSE | surface RMSE | 4000m bias | 4000m RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| MOM6 | 1.129 C | -0.132 C | 1.310 C | 1.333 C | 1.884 C | 0.422 C | 1.389 C |
| ocean_solver | 2.842 C | 0.384 C | 3.599 C | 4.032 C | 1.703 C | 6.347 C | 9.288 C |

Interpretation: ocean_solver currently beats MOM6 on this slice's surface
temperature, but loses decisively below 1000 m because of a warm deep-ocean
bias. This makes vertical mixing, 3D advection, and water-mass treatment the
main physics gap, not SST tuning.
