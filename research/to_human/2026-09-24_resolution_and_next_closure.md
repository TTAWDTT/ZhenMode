# Resolution Ladder and Next Closure

Date: 2026-09-24  
Status: autonomous continuation

## Current candidate

- The 0.50-degree candidate is reproduced and locked:
  - global A2 RMSE: `1.171 C`
  - North Atlantic A2 RMSE: `1.025 C`
  - near-wall raw bias: `-0.923 C`
  - runner: `scripts/run_candidate_baseline_050.sh`

## Resolution decision

The 365d ladder shows diminishing returns beyond 0.45 degree:

| resolution | global A2 | North Atlantic A2 | near-wall bias | wall time |
|---:|---:|---:|---:|---:|
| 0.50 | 1.171 C | 1.025 C | -0.923 C | 41 min |
| 0.45 | 1.149 C | 1.011 C | -0.917 C | 54 min |
| 0.40 | 1.142 C | 1.005 C | -0.937 C | 64 min |
| 0.35 | 1.114 C | 1.017 C | -0.965 C | 88 min |

0.45 is the best all-around rung. The repeat matches: global/NA A2 
`1.1488/1.0105 C`, near-wall bias `-0.9175 C`. `candidate_65n_045_gm0` is now 
locked and 0.50 is the fallback.

## What was rejected

Enhanced local coastal diffusion is not the answer:

- coastal vertical diffusivity `1e-5` and `1e-4`: no gain or worse;
- coastal horizontal diffusivity `1e3` and `1e4`: no gain or worse;
- coastal SST restore remains a useful diagnostic constraint, but is not a
  production default.

## Next physical test

Use a mature-model-style lateral eddy-transport closure first. The registered
A/B test compares the 0.45-degree control with:

- `kappa_gm=500 m2/s`
- `kappa_gm=1000 m2/s`

Run 30d first. Promote to 365d only if the global and regional metrics improve
without a stability cost. Runner:
`scripts/run_res045_gm_probe.sh`.


