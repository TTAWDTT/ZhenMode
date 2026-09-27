# Stage-F annual 3D MLD band diagnostic

Same 0.5-degree slice, final 90d annual 3D window, shared WOA reference, and
bathymetric mask. MLD uses the density-threshold definition on the annual 3D
snapshot and the WOA initial T/S on the same grid.

| band | model mean MLD | reference mean MLD | bias | RMSE |
|---|---:|---:|---:|---:|
| 60--40S | 224.6 m | 48.8 m | +175.8 m | 546.1 m |
| 40--20S | 17.5 m | 28.5 m | -11.1 m | 24.9 m |
| 20--0S | 18.7 m | 29.0 m | -10.3 m | 16.5 m |
| 0--20N | 22.4 m | 23.3 m | -0.8 m | 12.9 m |
| 20--40N | 26.0 m | 22.3 m | +3.7 m | 29.6 m |
| 40--60N | 247.1 m | 22.5 m | +224.6 m | 751.8 m |

Interpretation: the excessive mixed-layer depth is concentrated in the high
latitude bands, while the subtropical bands are too shallow. This supports
using a better high-latitude sea-ice / convective closure rather than another
global scalar mixing tune.

Source: `ocean_solver_stage_f_365d_3d_mld_benchmark.json`
