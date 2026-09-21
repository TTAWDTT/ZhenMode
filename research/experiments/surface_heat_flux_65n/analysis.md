# Surface Heat-Flux Sensitivity on the 65N Candidate

Date: 2026-09-21
Status: complete
Baseline: `results/north_boundary_ice_proxy/global_real_air_kv1e-6_kconv001_lat65_repeat.npz`
Bulk coefficient in baseline: `lambda_bulk = 40 W/m^2/K`

## Purpose

Separate the surface heat-exchange timescale from ocean heat transport before
implementing a sea-ice proxy. The only changed parameter was the bulk
heat-transfer coefficient.

## Results

| Run | Verdict | A1 RMSE | A2 RMSE | NA 40--60N A2 bias | NA 40--60N A2 RMSE |
|---|---:|---:|---:|---:|---:|
| lambda20 | FAIL | 2.189 C | 2.362 C | -2.583 C | 3.128 C |
| lambda40 baseline | PASS | 1.463 C | 1.773 C | -1.700 C | 2.175 C |
| lambda80 | PASS | 1.130 C | 1.530 C | -1.052 C | 1.572 C |

Relative to `lambda40`:

- `lambda20` worsens global A2 RMSE by `33.2%` and the North Atlantic region by
  `43.8%`.
- `lambda80` improves global A2 RMSE by `13.7%` and the North Atlantic region
  by `27.7%`.
- In the common `55..60N / 300..360E` band, the cold bias falls from
  `-1.67 C` to `-1.31 C`.

Both runs were stable over 365d. The lambda80 run has lower heat drift
(`0.275%`) and no sub-freezing North Atlantic cells in the 40--60N region.

## Interpretation

The response is monotonic and strong: the surface heat-exchange timescale is a
leading control on the remaining cold bias. However, this should be treated as
a **candidate scalar restoring strength**, not a mechanistic sea-ice closure or
a production default change.

Even at lambda80, the residual North Atlantic error is ordered by imposed bulk
flux. The strongest warming quintile remains the coldest relative to WOA
(mean bias `-2.25 C`, RMSE `2.37 C`). Thus scalar surface restoring reduces the
symptom, but ocean heat transport or high-latitude water-mass structure remains
the underlying issue.

## Decision

1. Keep `lambda_bulk=80` as the preferred 65N diagnostic candidate.
2. Do not immediately make it the production default; the original `40` value
   is the physically calibrated default and lambda80 changes the restoring
   timescale.
3. Deprioritize a sea-ice flux cap for this sector because the lambda80
   regional ocean has no cells at/below freezing.
4. Next diagnose ocean heat transport and mixed-layer closure at fixed
   `lambda_bulk=80`; GM/bolus transport is the first targeted sensitivity.
