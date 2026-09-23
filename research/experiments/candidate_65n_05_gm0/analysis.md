# Candidate Baseline at 0.5 Degree

Status: validated
Date: 2026-09-23
Question: Does the stabilized 0.5-degree grid improve the locked 0.7-degree
`lambda80` candidate?

## Change

Same candidate physics as `candidate_65n_07_gm0`, but at 0.5 degree with:
- `dt=1800s`
- `nu_h=2.0e6 m2/s`
- `min_depth=500`
- `smooth_passes=80`

## Runs

1. 30d stability probe: PASS.
2. 365d validation: PASS, 40.4 min wall time, max|u| peak 2.404.

## 365d metrics

| metric | 0.7 candidate | 0.5 candidate |
|---|---:|---:|
| global A2 RMSE | 1.336 C | 1.171 C |
| North Atlantic 40--60N A2 RMSE | 1.131 C | 1.025 C |
| North Atlantic raw bias | -0.832 C | -0.695 C |
| near-wall raw bias | -1.082 C | -0.923 C |

The North Atlantic A2 number here uses the current standard 90d steady-window
smoothing metric. The older locked-baseline note recorded `1.043 C` from an
earlier scoring pass, but the raw bias and near-wall values match exactly.

## Decision

Promote `candidate_65n_05_gm0` as the finer-resolution production-like
candidate. It is more expensive but improves the global and North Atlantic
metrics without using the coastal SST constraint. Keep the 0.7-degree run as a
fallback/legacy baseline.
