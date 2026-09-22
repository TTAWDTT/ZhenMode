# Coastal Mask Sensitivity Protocol: min-depth 500 m

Status: complete
Date: 2026-09-22
Baseline: `candidate_65n_07_gm0`
Question: Does raising the minimum ocean depth to 500 m improve the near-wall
cold bias by changing the land-adjacent/transitional mask?

## Change

Only `--min-depth` changes from 100 m to 500 m. All other locked candidate
physics remain unchanged.

## Runs

1. 30d stability probe.
2. If stable, one 365d comparison run.

## Decision rules

1. If the 30d probe is stable, run the 365d experiment.
2. If global A2 RMSE improves by at least 2% and the near-wall or coastal
   error improves, promote it to a diagnostic candidate. [MET 2026-09-22]
3. If global A2 improves but the target sector worsens, keep it as diagnostic
   information only.
4. If the run becomes unstable, record the failure and try a gentler mask
   change only if justified.
