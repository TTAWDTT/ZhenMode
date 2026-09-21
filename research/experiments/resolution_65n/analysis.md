# Horizontal Resolution Sensitivity on the Preferred 65N Candidate

Date: 2026-09-22
Status: complete
Reference: `results/heat_tendency_decomposition/global_real_air_lambda80_gm500_localconv_3dterms.npz`

## Results

| Run | Verdict | A1 RMSE | A2 RMSE | NA 40--60N A2 bias | NA 40--60N A2 RMSE |
|---|---:|---:|---:|---:|---:|
| 1.0 degree | PASS | 0.966 C | 1.412 C | -0.721 C | 1.313 C |
| 0.8 degree | PASS | 0.995 C | 1.360 C | -0.779 C | 1.252 C |

The 0.8-degree run improves global A2 RMSE by `3.64%` and the North Atlantic
regional RMSE by `4.65%`. It also reduces the common near-wall `55..60N`
regional RMSE from `1.666 C` to `1.421 C`.

The 0.5-degree probe with the same candidate physics was unstable and produced
NaNs by day 10, so it is not yet a usable production rung.

## Decision

1. Horizontal resolution is a real lever for the remaining North Atlantic cold
   bias.
2. Treat 0.8 degree as the new preferred diagnostic resolution.
3. Before moving to 0.5 degree, stabilize that rung or identify which coastal
   cell or closure is responsible for the blow-up.

## Follow-up stability probes

- 0.5 degree: FAIL_BLOWUP, NaN by day 10.
- 0.6 degree: FAIL_BLOWUP, NaN by day 10.
- 0.8 degree: PASS for both 30d and 365d.

Thus 0.8 degree is currently the finest usable resolution for this candidate.
