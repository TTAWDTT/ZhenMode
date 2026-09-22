# Upper Bulk-Lambda Sensitivity on the Min-Depth 500 Candidate

Status: complete
Date: 2026-09-22
Baseline: `candidate_65n_07_gm0_min500`
Question: Does continuing the earlier upward bulk-lambda sweep beyond 80 improve
the regional cold bias without over-constraining the global SST pattern?

## Change

Only `--lambda-bulk` changes. The current candidate physics, including
`--min-depth 500`, GM0, localized convection, FCT/TVD transport, and annual
real-air forcing, are otherwise fixed.

## Runs

- 30d stability probes at lambda 120, 160, and 240.
- 365d comparison runs at lambda 120, 160, and 240.

## Decision rules

1. Promote a higher lambda only if global A2 improves or remains materially
   unchanged while the North Atlantic / near-wall metric improves.
2. Do not set it as a production default if the gain is mainly from stronger
   restoring to the atmospheric state.
3. Prefer the best balance between global A2, regional A2, and near-wall A2.
