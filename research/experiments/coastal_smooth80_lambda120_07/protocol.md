# Coastal Geometry x Bulk-Lambda Protocol: lambda120 + smooth80

Status: complete
Date: 2026-09-22
Baseline: `lambda80 + min-depth500 + smooth80`
Question: Can lambda120 improve the North Atlantic / near-wall cold bias while
keeping most of the global-A2 benefit from smooth80?

## Change

Only `lambda_bulk` changes from 80 to 120. All other physics and coastal-mask
settings remain fixed at the smooth80 candidate.

## Runs

1. 30d stability probe: PASS.
2. 365d comparison: PASS.

## Result

| metric | value |
|---|---:|
| global A2 RMSE | 1.281 C |
| NA 40--60N A2 RMSE | 1.044 C |
| near-wall 55--60N A2 RMSE | 1.063 C |

## Decision

Lambda120 + smooth80 is a useful compromise but does not beat lambda160 +
smooth80 on regional skill. Keep it as a compromise diagnostic candidate.
