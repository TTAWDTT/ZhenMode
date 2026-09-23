# 0.45-Degree Ice-Floor Candidate Reproducibility

Status: locked
Date: 2026-09-24

## Configuration

Both annual runs used the 0.45-degree lambda80 candidate plus the opt-in
freezing-point proxy:

- `dt=1800s`, `nu_h=2e6`, `area` remapping
- `lambda_bulk=80`, `kappa_gm=0`, localized convection, FCT transport
- `min_depth=500`, `smooth_passes=80`
- seasonal 2023 NCEP wind and annual 2023 NCEP R1 2m air
- `--ice-air-floor --ice-air-floor-temp -1.8`

## Reproducibility

| run | verdict | global A2 | NA 40--60N A2 | near-wall raw bias | below-freezing cells |
|---|---|---:|---:|---:|---:|
| first | PASS | 1.11258 C | 1.00964 C | -0.91208 C | 0 |
| repeat | PASS | 1.11257 C | 1.00962 C | -0.91208 C | 0 |

The repeat matches to displayed precision. Promote the ice floor to the current
production-like candidate. Keep `candidate_65n_045_gm0` as the no-proxy
fallback.

## Reference files

- first: `results/ice_proxy_045/global_res045_icefloor_365d.npz`
- repeat: `results/ice_proxy_045/global_res045_icefloor_365d_repeat.npz`
- scored metrics: `metrics_365d_repeat.json`
- runner: `scripts/run_candidate_baseline_045_icefloor.sh`
