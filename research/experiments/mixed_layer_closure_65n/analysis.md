# Mixed-Layer / Convective-Closure Sensitivity on the 65N lambda80 GM500 Candidate

Date: 2026-09-21
Status: complete
Reference: `results/gm_transport_65n/global_real_air_kv1e-6_kconv001_lat65_lambda80_gm500.npz`

## Results

| Run | Verdict | A1 RMSE | A2 RMSE | NA 40--60N A2 bias | NA 40--60N A2 RMSE |
|---|---:|---:|---:|---:|---:|
| GM500, column conv, kconv=0.01 | PASS | 1.029 C | 1.449 C | -0.883 C | 1.417 C |
| localized convective gate | PASS | 0.964 C | 1.410 C | -0.713 C | 1.306 C |
| localized conv repeat | PASS | 0.964 C | 1.410 C | -0.713 C | 1.306 C |
| GM500, column conv, kconv=0.05 | PASS | 1.025 C | 1.446 C | -0.888 C | 1.422 C |
| GM500, column conv, kconv=0.002 | PASS | 1.037 C | 1.455 C | -0.867 C | 1.402 C |

Relative to the GM500 column-convective baseline, the localized convective
gate improves global A2 RMSE by `2.71%` and the North Atlantic regional A2
RMSE by `7.84%`. In the common `55..60N` band, the cold bias falls from
`-1.213 C` to `-1.081 C`.

Changing scalar `kappa_conv` by factors of 5 in either direction has only a
small effect. Thus the improvement comes from the closure structure, not from
another scalar mixing tune.

## Decision

1. Treat `65N + lambda80 + GM500 + localized convective adjustment` as the
   preferred diagnostic candidate.
2. Keep it opt-in for now; a production-default change should wait for a
   longer baseline and 3D heat-budget validation.
3. The next diagnostic should use 3D heat-tendency snapshots to decompose the
   remaining error into advection, convection, GM, diffusion and surface-flux
   contributions.
