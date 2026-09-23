# 0.45-Degree Candidate Reproducibility

Status: locked
Date: 2026-09-24

## Configuration

The first 365d run and repeat both used:

- `lat_max=65`, `resolution=0.45`, `resolution-remap=area`
- `lambda_bulk=80`, `kappa_v=1e-6`, `kappa_conv=0.01`, `kappa_gm=0`
- localized convection, FCT transport, projected advective velocity
- `min_depth=500`, `smooth_passes=80`
- `dt=1800s`, `nu_h=2e6`
- seasonal 2023 NCEP wind and annual 2023 NCEP R1 2m air temperature

## Reproducibility

| run | verdict | global A2 | NA 40--60N A2 | NA raw bias | near-wall raw bias |
|---|---|---:|---:|---:|---:|
| first | PASS | 1.1488 C | 1.0105 C | -0.6888 C | -0.9174 C |
| repeat | PASS | 1.1488 C | 1.0105 C | -0.6888 C | -0.9175 C |

The repeat matches to displayed precision. Promote
`candidate_65n_045_gm0` to the current production-like baseline; retain the
0.50-degree candidate as fallback.

## Reference files

- first run: `results/resolution_stability_07/global_res045_nuh2e6_dt1800_365d.npz`
- repeat: `results/candidate_65n_045_gm0/global_candidate_65n_045_gm0_repeat.npz`
- scored metrics: `metrics_repeat.json`
- runner: `scripts/run_candidate_baseline_045.sh`
