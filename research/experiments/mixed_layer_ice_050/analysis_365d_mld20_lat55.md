# Annual 20m mixed-layer/ice closure: latitude masks

Date: 2026-09-26  
Status: 55--65N passes the pre-registered annual gate

## Annual runs against the 0.5-degree ice-floor control

| run | global A2 | NA raw RMSE | near-wall raw bias | verdict |
|---|---:|---:|---:|---:|
| 0.5° ice-floor control | 1.128 | 0.924 | -0.921 | PASS |
| global 20m mixed-layer/ice | 1.273 | 1.007 | -0.469 | gate FAIL |
| 20m mixed-layer/ice, 50--65N | 1.130 | 0.987 | -0.477 | gate FAIL |
| 20m mixed-layer/ice, 55--65N | 1.127 | 0.872 | -0.604 | gate PASS |

The narrow 55--65N mask preserves global A2, improves the North Atlantic RMSE
by 5.6%, and improves near-wall bias by 0.317 C without breaking stability or
heat/salt drift.  It is a diagnostic-only candidate until reproduced.

## Reproducibility

The 365d repeat reproduces the candidate to ~4 decimal places:

- global A2 RMSE `1.12725 C`
- NA 40--60N raw RMSE `0.87224 C`
- near-wall raw bias `-0.60383 C`

This is a validated diagnostic closure at 0.5 degree. It is not a full CICE/SI3
sea-ice model and should not be described as production sea-ice.
