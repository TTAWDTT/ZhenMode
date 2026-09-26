# Annual 20m mixed-layer/ice closure: latitude masks

Date: 2026-09-26  
Status: 50--65N fails annual NA gate; 55--65N annual check running

## Annual runs against the 0.5-degree ice-floor control

| run | global A2 | NA raw RMSE | near-wall raw bias | verdict |
|---|---:|---:|---:|---:|
| 0.5° ice-floor control | 1.128 | 0.924 | -0.921 | PASS |
| global 20m mixed-layer/ice | 1.273 | 1.007 | -0.469 | gate FAIL |
| 20m mixed-layer/ice, 50--65N | 1.130 | 0.987 | -0.477 | gate FAIL |
| 20m mixed-layer/ice, 55--65N | pending | pending | pending | pending |

The 50--65N mask removes most of the uniform-depth global penalty, but the NA
gate still fails.  The 55--65N probe is the next, narrower test.
