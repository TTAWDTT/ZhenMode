# Gentle Coastal Restore on the 0.5-Degree Candidate

Status: validated
Date: 2026-09-23
Baseline: `candidate_65n_05_gm0` (`lambda80`, 0.5 degree, no coastal restore)

## 30d probe

Adding a hard coastal SST restore with `tau=1d`, cells<=7 improves the 30d
global A2 RMSE from `0.836 C` to `0.781 C`.

## 365d metrics

| run | global A2 | NA 40--60N | NA raw bias | near-wall raw bias |
|---|---:|---:|---:|---:|
| 0.7 candidate | 1.336 C | 1.131 C | -0.832 C | -1.082 C |
| 0.5 no restore | 1.171 C | 1.025 C | -0.695 C | -0.923 C |
| 0.5 + tau=1d | 1.020 C | 0.911 C | -0.520 C | -0.508 C |

The 365d `tau=1d` run passes with 40.8 min wall time and no drift flags.

## Decision

`lambda80 + 0.5 degree + tau=1d coastal restore` is the best all-around
diagnostic so far, but remains assimilation-like and is not a production
default. It is a useful compromise: stronger than the unmodified 0.5-degree
candidate while less aggressive than `tau=0.5d`.

## Next

1. Physicalize the coastal constraint with a boundary/lateral transport closure.
2. Continue using the 0.5-degree no-restore run as the production-like
   candidate until such a closure is available.
