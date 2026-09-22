# Near-Land Structure Diagnosis

Date: 2026-09-22
Status: complete
Runs: `lambda80_min500_3dterms`, `lambda160_min500_3dterms`

We compared the final-90d 3D state and heat-tendency snapshots of the
global-A2-favored candidate (`lambda_bulk=80`, `min-depth=500`) with the
regional-bias-favored candidate (`lambda_bulk=160`, `min-depth=500`).

## Near-land bands

| Region | Group | lambda80 SST bias / RMSE | lambda160 SST bias / RMSE | lambda80 speed | lambda160 speed |
|---|---|---:|---:|---:|---:|
| global | land 0--3 | -1.048 / 1.503 C | -0.905 / 1.410 C | 0.162 m/s | 0.162 m/s |
| global | land 4--7 | -0.712 / 1.012 C | -0.626 / 0.929 C | 0.175 m/s | 0.175 m/s |
| global | land >=8 | -0.579 / 0.878 C | -0.618 / 0.936 C | 0.218 m/s | 0.218 m/s |
| NA 40--60N | land 0--3 | -0.672 / 0.937 C | -0.438 / 0.875 C | 0.070 m/s | 0.069 m/s |
| NA 40--60N | land 4--7 | -1.094 / 1.232 C | -0.762 / 0.950 C | 0.070 m/s | 0.069 m/s |
| NA 40--60N | land >=8 | -0.791 / 1.011 C | -0.617 / 0.812 C | 0.092 m/s | 0.092 m/s |
| near wall 55--60N | land 0--3 | -1.308 / 1.405 C | -1.292 / 1.448 C | 0.040 m/s | 0.036 m/s |
| near wall 55--60N | land 4--7 | -1.161 / 1.197 C | -0.908 / 0.959 C | 0.036 m/s | 0.035 m/s |
| near wall 55--60N | land >=8 | -0.684 / 0.769 C | -0.593 / 0.662 C | 0.049 m/s | 0.051 m/s |

## Interpretation

1. Lambda 160 improves the `land 4--7` band in both the broader North Atlantic
   and the near-wall sector. This is where much of the cold bias lives.
2. Surface current speed is almost unchanged in the coastal bands. Thus the
   regional improvement is not caused by a major change in boundary-current
   structure.
3. In the `land 0--3` band the model is still very cold, especially in
   `55..60N`. Convection is very large there, while surface bulk flux is almost
   zero. This suggests that the remaining issue is not simply weak surface
   heat exchange.
4. The upper `50--200 m` convection term is small, so vertical export is not the
   dominant cause of the surface cold bias.
5. The lambda 160 benefit is mostly a stronger surface restoring / bulk-flux
   effect, not a boundary-current-structure fix.

## Decision

The remaining near-wall cold bias is still localized in the immediate coastal
band, but lambda tuning has not changed the current structure much. The next
step should therefore test coastal geometry (bathymetry smoothing / mask shape)
rather than continue tuning `lambda_bulk`.
''',encoding='utf-8')
