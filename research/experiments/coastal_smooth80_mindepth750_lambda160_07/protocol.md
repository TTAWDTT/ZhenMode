# Coastal Mask Depth Protocol: min-depth 750

Status: complete
Date: 2026-09-22
Baseline: `lambda160 + min-depth500 + smooth80`
Question: Does a 750 m minimum depth improve the coastal-band bias without
losing the regional candidate?

## Change

Only `--min-depth` changes from 500 to 750 m. All other physics are fixed at
the lambda160 + smooth80 candidate.

## Runs

1. 30d stability probe: PASS.
2. 365d comparison: PASS.

## Result

| metric | value |
|---|---:|
| global A2 RMSE | 1.282 C |
| NA 40--60N A2 RMSE | 1.019 C |
| near-wall 55--60N A2 RMSE | 1.061 C |

## Decision

Min-depth500 remains the all-around candidate. Stop the mask-floor scan.
