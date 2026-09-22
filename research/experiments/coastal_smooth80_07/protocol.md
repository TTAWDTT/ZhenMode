# Coastal Geometry Sensitivity Protocol: smooth-passes 80

Status: complete
Date: 2026-09-22
Baseline: `lambda160 + min-depth 500`
Question: Does stronger bathymetry smoothing improve the remaining land-adjacent
cold bias without degrading the regional candidate?

## Change

Only `--smooth-passes` changes, from 30 to 80. All other candidate physics are
fixed at the lambda 160 / min-depth 500 configuration.

## Runs

1. 30d stability probe: PASS.
2. 365d comparison: PASS, global A2 `1.294 C`.
3. 365d repeat: PASS, global A2 `1.294 C`.

## Decision

Promote `lambda160 + min-depth500 + smooth80` as the all-around diagnostic
candidate. Keep `lambda80 + smooth80` as the pure global-A2 candidate. The
near-wall metric is slightly worse than smooth30, so the next experiment should
isolate the immediate-land 0--3-cell band rather than continue scanning lambda.
