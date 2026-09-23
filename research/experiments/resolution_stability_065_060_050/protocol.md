# Finer-Resolution Stability Protocol

Status: active
Date: 2026-09-23
Question: Does stabilizing finer horizontal resolution improve the remaining
coastal/North Atlantic error?

Primary stabilization: `--dt 1800`, `--nu-h 2e6`. Keep the 0.7-degree all-around
candidate physics otherwise unchanged. Use area remapping for resolutions that
are not multiples of 0.1 degree.

Metrics:
- stability: 10d, 30d, and 365d verdicts
- climate: global A2, North Atlantic 40..60N A2, and near-wall 55..60N A2 RMSE

Decision rules:
1. Promote a finer resolution only after a PASS 365d run and scored metrics.
2. Reject a finer rung if it fails or materially worsens the all-around metrics.
