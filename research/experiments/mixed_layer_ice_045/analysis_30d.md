# Mixed-layer / Sea-Ice Closure 30d A/B

Date: 2026-09-26  
Status: 30d A/B passed; 365d check pending

## Runs

| Run | Flags | Verdict | Wall |
|---|---|---:|---:|
| control | `candidate_65n_045_icefloor`, 30d | PASS | ~5 min |
| mixed-layer/ice | `--mixed-layer-depth 50 --ice-salt-flux 1e-7`, 30d | PASS | 4.9 min |

Both use 0.45-degree, 65N, lambda80, GM0, localized convection, FCT,
annual real air forcing, seasonal wind, ice-air floor.

## Metrics

| Metric | Control 30d | Mixed-layer/ice 30d | Change |
|---|---:|---:|---:|
| global A2 RMSE | 0.8023 C | 0.7165 C | -10.7% |
| global A1 RMSE | 0.4058 C | 0.2101 C | -48.2% |
| global raw bias | -0.3024 C | -0.1443 C | +52.3% |
| global raw RMSE | 0.4828 C | 0.3262 C | -32.4% |
| NA 40--60N raw bias | -0.4926 C | -0.2948 C | +40.2% |
| NA 40--60N raw RMSE | 0.5493 C | 0.4282 C | -22.0% |
| near-wall 55--60N raw bias | -0.5569 C | -0.2045 C | +63.3% |
| near-wall 55--60N raw RMSE | 0.5909 C | 0.2195 C | -62.9% |
| heat drift | -0.1063% | -0.0441% | better |
| salt drift | -0.000189% | -0.000183% | similar |
| below-freezing cells | 0 | 0 | same |
| mean MLD | 31.5 m | 31.5 m | same |
| A2 corr | 0.9973 | 0.9974 | slightly better |

## Decision

1. The 30d A/B passes all four pre-registered gates.
2. The mixed-layer/ice closure gives a broad improvement, not a single-metric
   tradeoff.
3. Promote the closure to the 365d check.
4. Do not promote it as production default until that 365d check repeats.

## Reference files

- `results/mixed_layer_ice_045/global_mixed_layer_ice_045_30d.npz`
- `research/experiments/mixed_layer_ice_045/benchmark_30d.json`
- `results/ice_proxy_045/global_res045_icefloor_30d.npz`
- `research/experiments/ice_proxy_045/benchmark_30d.json`
- runner: `scripts/run_mixed_layer_ice_045.sh`
