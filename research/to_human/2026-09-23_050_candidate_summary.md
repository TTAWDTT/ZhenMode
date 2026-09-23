# 0.5-Degree Candidate and Gentle Restore Summary

Date: 2026-09-23  
Status: consolidated

## Where we are

1. **Stabilized finer resolution**
   - The old 0.65/0.60/0.55/0.50-degree failures were not a fundamental limit.
   - Using `dt=1800s` and `nu_h=2.0e6 m2/s` makes these rungs stable.
   - The 0.50-degree 365d run passes and improves the climate metrics.

2. **New production-like candidate**
   - `candidate_65n_05_gm0`
   - `lambda_bulk=80`, `kappa_gm=0`, localized convection, FCT transport,
     `min_depth=500`, `smooth_passes=80`.
   - Compared with the 0.70-degree baseline:
     - global A2 RMSE: `1.336 C -> 1.171 C`
     - North Atlantic 40--60N A2 RMSE: `1.131 C -> 1.025 C`
     - North Atlantic raw bias: `-0.832 C -> -0.695 C`
     - near-wall raw bias: `-1.082 C -> -0.923 C`
   - Runner: `scripts/run_candidate_baseline_050.sh`

3. **Best diagnostic compromise**
   - `lambda80 + 0.50 degree + tau=1d coastal SST restore`
   - global A2 RMSE: `1.020 C`
   - North Atlantic A2 RMSE: `0.911 C`
   - near-wall raw bias: `-0.508 C`
   - This is the best all-around result so far, but it is assimilation-like and
     is **not** a production default.

## What we learned

- Resolution and coastal SST restore are complementary.
- The finer grid reduces the coastal/North Atlantic error but does **not** remove
  the boundary-value problem.
- The heat-budget diagnostic showed that the coastal constraint acts like a
  mixed-layer/boundary control, not as a missing scalar diffusion or bulk-flux
  term.
- Cosine tapering of the restore band is worse than the hard band.

## Current file map

- Locked 0.70-degree baseline: `research/experiments/candidate_65n_07_gm0/`
- Validated 0.50-degree candidate: `research/experiments/candidate_65n_05_gm0/`
- Gentle restore diagnostic:
  `research/experiments/lambda80_res050_restore_07/`
- Resolution ladder:
  `research/experiments/resolution_stability_065_060_050/`
- Coastal budget diagnostic: `research/experiments/coastal_budget_07/`
- Running state: `research/research-state.yaml`

## Next priorities

1. Keep `candidate_65n_05_gm0` as the current production-like candidate.
2. Reproduce the 0.50-degree candidate once more before locking it.
3. Replace the diagnostic coastal restore with a physical boundary/lateral
   heat-transport closure.
4. Only then consider coupling the closure to the 0.50-degree baseline.
