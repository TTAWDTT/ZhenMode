# Stage-F 30d matched performance

Direct wall-time on the same WSL host is diagnostic only: ocean_solver uses the
JAX runtime while MOM6 uses four MPI ranks. This is not a scaling benchmark.

| model | 30d wall time | runtime mode | surface A2 | global 3D RMSE |
|---|---:|---|---:|---:|
| ocean_solver | 3.5 min | JAX, 0.5-degree slice | 1.692 C | 2.842 C |
| MOM6 | 74.5 min | 4 MPI ranks | 1.782 C | 1.129 C |

The performance result means the prototype is much faster in this short run,
but the industrial model is much better at representing 3D temperature. Skill
and speed must not be conflated.
