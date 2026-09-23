# Candidate 0.5-Degree Reproducibility

Date: 2026-09-23  
Status: complete  
Runner: `scripts/run_candidate_baseline_050.sh`  
Tag: `candidate_65n_05_gm0_repeat`

## Result

The 365d repeat passes and reproduces the first 0.5-degree candidate run.

| Metric | First run | Repeat |
|---|---:|---:|
| verdict | PASS | PASS |
| global A2 RMSE | 1.171 C | 1.171 C |
| North Atlantic 40--60N A2 RMSE | 1.025 C | 1.025 C |
| North Atlantic raw bias | -0.695 C | -0.695 C |
| near-wall raw bias | -0.923 C | -0.923 C |
| wall time | 40.4 min | 41.0 min |
| heat drift | -0.3914% | -0.3911% |
| salt drift | -0.00019% | -0.00048% |

The tiny differences are consistent with float32 restart-free numerical noise,
not a physics or configuration change. This is sufficient to lock
`candidate_65n_05_gm0` as the current production-like candidate.

## Decision

Lock `candidate_65n_05_gm0` as the current production-like baseline. Keep
`candidate_65n_07_gm0` as the legacy fallback baseline.
