# Near-Wall Gradient and Advective-Cooling Diagnosis

Date: 2026-09-22
Status: complete
Runs: `gm0_3dterms`, `gm500_3dterms`

Both runs use the locked 0.7-degree candidate physics and are compared over the
final 90 days. The table below uses the final-90d mean SST and the full solver
advection term from the saved 3D heat-tendency snapshots.

| Run | Region | SST bias (C) | SST RMSE (C) | east grad. | north grad. | speed (m/s) | alignment | saved advection (K/d) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GM0 | 40--60N | -0.832 | 1.052 | +0.212 | -0.529 | 0.0814 | 0.601 | -0.0572 |
| GM500 | 40--60N | -0.911 | 1.138 | +0.208 | -0.526 | 0.0817 | 0.587 | -0.0720 |
| GM0 | 55--60N wall | -1.081 | 1.186 | +0.296 | -0.282 | 0.0416 | 0.355 | -0.0377 |
| GM500 | 55--60N wall | -1.146 | 1.246 | +0.276 | -0.285 | 0.0418 | 0.300 | -0.0752 |
| GM0 | 40--55N interior | -0.762 | 1.012 | +0.188 | -0.599 | 0.0927 | 0.670 | -0.0627 |
| GM500 | 40--55N interior | -0.844 | 1.106 | +0.188 | -0.595 | 0.0930 | 0.668 | -0.0711 |

Gradients are in K per 100 km. The solver advection is the full saved term, not
just the horizontal proxy.

## Findings

1. The near-wall sector is colder and has a smaller current speed than the
   broader 40--60N sector, but its local SST gradient is not anomalously large.
2. GM0 has slightly larger near-wall eastward gradient, yet weaker full solver
   advective cooling (`-0.0377` versus `-0.0752` K/day in GM500).
3. Solver advection correlates positively with SST error: colder cells tend to
   have stronger advective cooling. The area-weighted correlation over the full
   North Atlantic is `0.271` for GM0 and `0.385` for GM500.
4. The 100 coldest near-wall cells cluster near `300--302E` at `57--59N` and
   around `313--315E / 59.45N`. Several of these cells have very weak mean
   current (`0.006--0.010 m/s`), so they are not simply downstream of an
   over-strong boundary current.

## Interpretation

The remaining error is localized rather than a broad regional transport failure.
The full solver advection still explains part of the cold-error gradient, but
the near-wall correlation is weak (`0.128` for GM0). This means the coldest
cells are controlled more by local wall/geometry and vertical redistribution
than by a simple large-scale current-temperature misalignment.

GM500 increases near-wall advective cooling by about `0.038 K/day` relative to
GM0, which is consistent with the earlier regional attribution. However, the
large-scale speed and gradient fields remain almost unchanged. Therefore, GM
over-cooling is a local interaction, not a change in the boundary-current speed.

## Next

1. Run a vertical heat-tendency attribution for the top near-wall cold cells.
2. Check whether these cells are isolated by bathymetry/land masks or connected
   to the open North Atlantic.
3. Only after that decide whether the correct next lever is a north-boundary
   heat/salt condition, a high-latitude surface-physics proxy, or another
   transport-geometry change.
