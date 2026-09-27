# Direct external comparison

Current direct external model is MOM6 on the matched 0.5-degree slice.

## Surface / integrated diagnostics

| run | verdict | days | global A2 | NA RMSE | near-wall RMSE | global bias | heat drift | salt drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wind_30d_ocean_solver | PASS | 30 | 0.8658 | 0.6751 | 0.2106 | -0.05795 | -0.02618 | -0.000203 |
| wind_30d_MOM6 | PASS | 30 | 1.096 | 0.6814 | 0.2434 | -0.407 | 1.886e-14 | 0 |
| restore_30d_ocean_solver | PASS | 30 | 0.8645 | 0.6661 | 0.2154 | -0.05159 | -0.02618 | -0.0001377 |
| restore_30d_MOM6 | PASS | 30 | 1.115 | 0.5794 | 0.2179 | -0.395 | 0.008733 | 0 |
| stage_f_30d_ocean_solver | PASS | 30 | 1.692 | 2.088 | 1.958 | -0.2762 | -0.1838 | -0.0002087 |
| stage_f_30d_MOM6 | PASS | 30 | 1.782 | 2.207 | 2.095 | -0.4771 | -0.1074 | 0 |
| stage_f_365d_ocean_solver | PASS | 365 | 1.297 | 0.7992 | 0.788 | -0.6496 | -0.962 | -0.0006209 |
| stage_f_365d_MOM6 | pending | 365 | - | - | - | - | - | - |

## 3D temperature diagnostics

| run | global 3D RMSE | 40--60N 3D RMSE | surface RMSE | verdict |
|---|---:|---:|---:|---|
| stage_f_30d_ocean_solver | 0.861 C | 1.175 C | 1.703 C | PASS |
| stage_f_30d_MOM6 | 1.036 C | 1.123 C | 1.884 C | PASS |
| stage_f_365d_ocean_solver | 1.547 C | 1.276 C | 1.142 C | PASS |
| stage_f_365d_MOM6 | pending | pending | pending | running |



## Contract status

The paired annual Stage-F manifests now pass the standardized contract validator:

- ocean_solver: ocean_solver_stage_f_365d_3d_manifest.json
- MOM6: mom6_stage_f_dynamic_365d_v12_manifest.json

The MOM6 provenance is explicitly marked not_comparable because the source commit
was not recorded, so the numerical comparison may proceed but exact code
reproducibility remains incomplete.
