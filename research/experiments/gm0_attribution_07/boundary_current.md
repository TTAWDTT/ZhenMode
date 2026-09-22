# Boundary Current / Heat Transport Diagnosis

Date: 2026-09-22
Status: complete
Runs: `gm0_3dterms`, `gm500_3dterms`

## Surface velocity and heat-transport metrics

| Metric | GM0 NA | GM500 NA | GM0 near wall | GM500 near wall |
|---|---:|---:|---:|---:|
| mean u (m/s) | 0.0149 | 0.0151 | 0.0089 | 0.0088 |
| mean v (m/s) | 0.0043 | 0.0046 | 0.0070 | 0.0076 |
| rms speed (m/s) | 0.0371 | 0.0377 | 0.0275 | 0.0282 |
| max speed (m/s) | 0.228 | 0.228 | 0.172 | 0.173 |
| mean `v*T` | 0.0394 | 0.0401 | 0.0500 | 0.0517 |

The large-scale surface velocities and `v*T` heat-transport proxy are almost
unchanged between GM0 and GM500.

## Interpretation

The GM500 regional degradation is therefore not caused by a gross change in
boundary-current speed. It comes from a local temperature-gradient/advection
interaction that strengthens surface advective cooling near the wall.

## Next

Continue with a finer diagnostic of the near-wall temperature gradient,
advective orientation, and local heat convergence before deciding whether GM
should be retuned or simply disabled at this resolution.
