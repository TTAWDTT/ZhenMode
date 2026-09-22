# Coastal Mask Depth Protocol: min-depth 1000

Status: complete
Date: 2026-09-22
Baseline: `lambda160 + min-depth500 + smooth80`
Question: Does raising the minimum ocean depth to 1000 m remove the remaining
0--3-cell coastal cold bias?

## Change

Only `--min-depth` changes from 500 to 1000 m. All other physics are fixed at
the lambda160 + smooth80 candidate.

## Runs

1. 30d stability probe: PASS.
2. 365d comparison: PASS.

## Result

| metric | value |
|---|---:|
| global A2 RMSE | 1.276 C |
| NA 40--60N A2 RMSE | 1.026 C |
| near-wall 55--60N A2 RMSE | 1.095 C |

## Decision

Min-depth1000 improves global A2 but degrades the regional metrics, so keep
min-depth500 as the all-around candidate.
