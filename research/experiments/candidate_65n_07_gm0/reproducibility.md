# Candidate Baseline Reproducibility

Date: 2026-09-22
Status: complete
Runner: `scripts/run_candidate_baseline.sh`
Tag: `candidate_65n_07_gm0_repeat`

## Result

The locked candidate reproduced the original 0.7-degree GM0 run.

| Metric | Original | Repeat |
|---|---:|---:|
| A1 RMSE | 0.973 C | 0.973 C |
| Official A2 RMSE | 1.336 C | 1.336 C |
| North Atlantic 40--60N A2 RMSE | 1.043 C | 1.043 C |
| North Atlantic raw bias | -0.832 C | -0.832 C |
| Near-wall raw bias | -1.082 C | -1.082 C |

Both runs were stable over 365d. The repeated run passed A1 and A2.

## Note

This confirms `candidate_65n_07_gm0` is a reproducible diagnostic baseline.
The next step is to attribute the GM0 benefit against GM500 at the same
resolution using velocity, heat transport, and heat-tendency diagnostics.
