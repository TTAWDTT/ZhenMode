# Surface Salinity Restoring Sensitivity

Date: 2026-09-22
Status: complete
Reference: `results/heat_tendency_decomposition/global_real_air_lambda80_gm500_localconv_3dterms.npz`

## Results

| Run | Verdict | A1 RMSE | A2 RMSE | NA 40--60N A2 bias | NA 40--60N A2 RMSE |
|---|---:|---:|---:|---:|---:|
| no SSS restoring | PASS | 0.966 C | 1.412 C | -0.721 C | 1.313 C |
| SSS restoring 30d | PASS | 0.959 C | 1.409 C | -0.712 C | 1.308 C |
| SSS restoring 90d | PASS | 0.961 C | 1.409 C | -0.712 C | 1.305 C |

Both SSS-restoring runs were stable. They improved global A2 RMSE by about
`0.19%` and the North Atlantic regional RMSE by `0.38--0.58%`. This is real but
small and well below any decisive improvement threshold.

## Decision

1. SSS restoring gives a very small, reproducible improvement.
2. It is not the main lever for the remaining cold bias.
3. Keep it as an optional diagnostic refinement, but do not treat it as the
   next structural fix.
