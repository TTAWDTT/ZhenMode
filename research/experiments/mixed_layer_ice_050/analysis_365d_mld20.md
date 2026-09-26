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

## Error decomposition

A 10-degree latitude-band decomposition shows that the 20m mixed-layer/ice run
does not merely trade coastal skill for global skill.  It worsens the 10S..40N
band and the southern subtropics, while improving the polar/near-wall band.

| band | control RMSE | 20m mixed-layer RMSE |
|---|---:|---:|
| -40..-30S | 0.870 | 1.052 |
| -20..-10S | 0.843 | 0.953 |
| 0..10N | 0.729 | 0.954 |
| 20..30N | 1.106 | 1.134 |
| 30..40N | 1.334 | 1.506 |
| 50..60N | 1.353 | 1.337 |
| 60..70N | 2.397 | 2.201 |

So the next physical experiment should not simply deepen or weaken the uniform
slab; it should test a latitude/stratification-aware mixed-layer mask.
