# Stage-F 365d direct 3D temperature comparison

Same 0.5-degree slice, final 90-day window, shared WOA reference, and shared
bathymetric vertical mask. The annual ocean_solver control is complete. The
MOM6 annual exact dynamic-bulk rerun is in progress on local Linux storage.

| model | global 3D RMSE | global 3D bias | 40-60N 3D RMSE | 55-60N 3D RMSE | surface RMSE | MLD bias | MLD RMSE | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ocean_solver | 1.547 C | -0.670 C | 1.276 C | 1.124 C | 1.142 C | +139.3 m | 655.7 m | PASS |
| MOM6 | pending | pending | pending | pending | pending | pending | pending | running |

Interpretation: the annual 3D skill degrades relative to the corrected 30d
control (ocean_solver global 3D RMSE `0.861 C` -> `1.547 C`), mainly through
upper-ocean cold bias. This is now the annual target that MOM6 must match under
the same Stage-F contract.
