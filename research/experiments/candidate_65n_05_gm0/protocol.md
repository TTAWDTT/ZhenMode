# Candidate 0.5-Degree Baseline Protocol

Status: validated
Date: 2026-09-23
Question: Does the 0.5-degree grid improve the locked 0.7-degree `lambda80`
candidate without changing other physics?

Runner: `scripts/run_candidate_baseline_050.sh`
Physics: 65N, `lambda_bulk=80`, `kappa_v=1e-6`, `kappa_conv=0.01`, `kappa_gm=0`,
localized convection, FCT transport, projected advective velocity,
`min_depth=500`, `smooth_passes=80`.

Metrics: global A2 RMSE, North Atlantic 40--60N A2 RMSE, and near-wall raw bias.
Decision rule: promote only after a stable 30d probe and a PASS 365d run with
improved all-around metrics.
