# Gentle Restore on the 0.5-Degree Candidate

Status: validated
Date: 2026-09-23
Question: Does a gentle `tau=1d` coastal SST restore retain most of the gain on
the 0.5-degree lambda80 candidate?

Control: `candidate_65n_05_gm0`, no coastal restore.
Experiment: same run plus hard coastal SST restore, `tau=1d`, cells<=7.
Decision rule: run 365d only if the 30d probe improves global, North Atlantic,
and near-wall metrics.
