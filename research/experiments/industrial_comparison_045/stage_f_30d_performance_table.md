# Stage-F matched performance

Direct wall-time on the same WSL host is diagnostic only: ocean_solver uses the
JAX runtime while MOM6 uses four MPI ranks. This is not a scaling benchmark.

## 30d

| model | 30d wall time | runtime mode | surface A2 | corrected global 3D RMSE |
|---|---:|---|---:|---:|
| ocean_solver | 3.5 min | JAX, 0.5-degree slice | 1.692 C | 0.861 C |
| MOM6 | 74.5 min | 4 MPI ranks | 1.782 C | 1.036 C |

## 365d

| model | 365d wall time | runtime mode | status |
|---|---:|---|---|
| ocean_solver annual 3D control | 30.4 min | JAX, 0.5-degree slice | completed, global 3D RMSE 1.547 C |
| MOM6 annual v8 | running | 4 MPI ranks | annual final-90d 3D score pending |
