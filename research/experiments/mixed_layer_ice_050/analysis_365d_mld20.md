# 20m Mixed-layer / Sea-Ice Annual Probe

Date: 2026-09-26  
Status: failed the pre-registered internal gates

## Same-resolution control

`candidate_65n_050_icefloor_365d` is the fair 0.5-degree annual control:

- global A2 RMSE `1.128 C`
- NA 40--60N raw RMSE `0.924 C`
- near-wall raw bias `-0.921 C`

## 20m mixed-layer/ice annual run

| metric | control | mixed-layer/ice 20m |
|---|---:|---:|
| global A2 RMSE | 1.128 C | 1.273 C |
| NA 40--60N raw RMSE | 0.924 C | 1.007 C |
| near-wall raw bias | -0.921 C | -0.469 C |
| heat drift | -0.252% | -0.113% |

## Decision

The 20m parameter pair improves the near-wall bias but still worsens global A2
and North Atlantic RMSE.  It is not a production candidate.

## Next

1. Keep the 0.5-degree ice-floor control as the fair same-resolution reference.
2. Do not scan mixed-layer depth as a pure tuning knob.
3. Next useful step is to diagnose the mixed-layer heat budget and understand
   why the regional improvement is bought by a global error.
