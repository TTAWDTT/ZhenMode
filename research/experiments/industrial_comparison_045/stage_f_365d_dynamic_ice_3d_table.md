# Stage-F 365d dynamic-ice 3D gate

Same 0.5-degree slice, final 90d window, shared WOA reference and bathymetric
mask. MLD uses the density-threshold definition on the same 3D snapshots.

| run | global 3D RMSE | global 3D bias | 40--60N 3D RMSE | 55--60N 3D RMSE | surface RMSE | mean MLD | MLD bias | MLD RMSE | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Stage-F no-ice control | 1.547 C | -0.670 C | 1.276 C | 1.124 C | 1.142 C | 170.8 m | +139.3 m | 655.7 m | PASS; current control |
| dynamic ice, no MLD | 1.543 C | -0.669 C | 1.278 C | 1.140 C | 1.180 C | 288.7 m | +257.2 m | 952.7 m | PASS stability; not promoted |

Interpretation: the annual dynamic-ice run does not improve the annual 3D
temperature field and substantially worsens the high-latitude MLD diagnostic.
Do not promote it as the annual baseline; the remaining industrial comparison
target is the MOM6 annual 3D score.
