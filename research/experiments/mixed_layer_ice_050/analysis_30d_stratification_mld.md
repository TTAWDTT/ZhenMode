# Stratification-aware mixed-layer depth: 30d probe

Date: 2026-09-26  
Status: 30d PASS, annual FAIL

## Change

Replaces the fixed 55--65N latitude mask with a 2D mixed-layer depth derived
from the WOA density threshold (`0.03 kg/m3` below the 10m reference).  The
field is clipped to 10--100m.  It is an initial-state diagnostic, not yet an
evolving MLD state.

Observed MLD field on the 0.5-degree grid:

- mean `31.5 m`
- p10/p50/p90: `15 / 30 / 50 m`

## 30d result

Against the same-protocol 30d `55--65N, 20m` diagnostic:

| metric | 55--65N 20m | stratification MLD | gate |
|---|---:|---:|---|
| global A2 RMSE | 0.825 C | **0.756 C** | PASS |
| NA 40--60N RMSE | 0.489 C | **0.420 C** | PASS |
| near-wall bias | -0.320 C | **-0.286 C** | PASS |

It also passes against the uniform global 20m probe (`global A2 0.758 C`,
`NA RMSE 0.438 C`, near-wall bias `-0.311 C`).  Heat/salt drift are bounded and
stability is PASS.  Because this is only 30d, the decision remains annual.


## Annual rejection

The same field was run for 365d.  It improved near-wall bias but failed both
main annual climate gates:

| metric | 55--65N 20m | stratification MLD | gate |
|---|---:|---:|---|
| global A2 RMSE | 1.127 C | 1.279 C | FAIL |
| NA 40--60N RMSE | 0.872 C | 1.051 C | FAIL |
| near-wall bias | -0.604 C | **-0.405 C** | PASS |

Heat drift remains bounded, but the fixed-latitude 55--65N 20m closure remains
the better validated diagnostic.  The stratification field should not be
promoted.  A possible follow-up is to restrict it regionally or combine it with
evolving MLD, not to keep accepting short probes.
