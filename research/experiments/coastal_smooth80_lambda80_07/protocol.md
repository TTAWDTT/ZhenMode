# Coastal Geometry Sensitivity Protocol: lambda80 + smooth-passes 80

Status: planned
Date: 2026-09-22
Baseline: `lambda80 + min-depth 500`
Question: Does the smoother 80-pass bathymetry also improve the global-A2
candidate without hurting the regional cold bias?

## Change

Only `--smooth-passes` changes from 30 to 80 on the lambda80 + min-depth 500
candidate. All other physics are unchanged.

## Runs

1. 30d stability probe.
2. If stable, one 365d comparison run.
