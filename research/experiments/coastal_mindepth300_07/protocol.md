# Coastal Mask Sensitivity Protocol: min-depth 300 m

Status: complete
Date: 2026-09-22
Baseline: `candidate_65n_07_gm0_min500`
Question: Does a gentler 300 m mask floor preserve the coastal improvement
while reducing the broader North Atlantic degradation seen at 500 m?

## Change

Only `--min-depth` changes, from 500 m to 300 m relative to the preferred
min-depth 500 candidate. All other physics remain unchanged.

## Runs

1. 30d stability probe.
2. If stable, one 365d comparison run.

## Decision rules

1. If the 30d probe is stable, run the 365d experiment.
2. If global A2 improves and the North Atlantic or near-wall metric improves
   relative to min-depth 500, promote it as the preferred diagnostic candidate. [NOT MET 2026-09-22]
3. If it improves global but worsens the target sector, keep it as diagnostic
   information only.
