# Min-Depth 300 m Sensitivity Result

Date: 2026-09-22
Status: complete
Runs: `coastal_mindepth300_30d`, `coastal_mindepth300_365d`

Only `--min-depth` changes from 500 m to 300 m relative to the preferred
min-depth 500 candidate. The 30d probe and 365d run were stable.

## Climate metrics

| Metric | baseline | min-depth 300 | min-depth 500 |
|---|---:|---:|---:|
| Global A2 RMSE | 1.336 C | 1.299 C | 1.282 C |
| North Atlantic 40--60N A2 RMSE | 1.131 C | 1.137 C | 1.158 C |
| Near-wall 55--60N A2 RMSE | 1.153 C | 1.136 C | 1.148 C |

The 300 m floor is better than the baseline globally, but it is not better than
the 500 m candidate. It slightly improves the near-wall metric relative to the
baseline, but the broader North Atlantic metric is still slightly worse.

## Decision

Keep `min-depth 500` as the preferred diagnostic candidate. A 300 m floor is a
useful sensitivity point but does not dominate it. Next coastal experiments
should avoid further mask-floor tuning and focus on near-land ventilation or
current structure.
