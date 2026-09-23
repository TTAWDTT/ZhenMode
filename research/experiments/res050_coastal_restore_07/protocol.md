# 0.5-Degree Plus Coastal Restore Protocol

Status: complete
Date: 2026-09-23
Question: Does the 0.50-degree stabilized resolution subsume the coastal SST
constraint?

Control: 0.50-degree stabilized run with no coastal restore.
Experiment: same run plus hard coastal restore, `tau=0.5d`, cells<=7.
Timescales: 30d probe followed by a 365d validation run.

Decision rule: run 365d only if the 30d probe improves global, North Atlantic,
and near-wall metrics over the no-restore control.

## Gentler-timescale follow-up

After validating tau=0.5d, run 30d probes at tau=3d and tau=1d, then validate tau=1d for 365d. Decision rule: choose the gentlest timescale that retains most of the all-around gain while still reporting tau=0.5d as the upper bound.

