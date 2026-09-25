# Mixed-layer / sea-ice benchmark status

Date: 2026-09-26

## What is new

1. Added a standardized benchmark scorer:
   `src/benchmark_metrics.py`
2. Added the minimal sea-ice / mixed-layer closure:
   `src/mixed_layer_ice.py`
3. Added opt-in solver flags:
   `--mixed-layer-depth`, `--ice-salt-flux`, `--ice-freeze-temp`
4. Added a 30d A/B runner:
   `scripts/run_mixed_layer_ice_045.sh`

## 30d A/B result

Against the 0.45-degree ice-floor control:

| Metric | Control | Mixed-layer/ice | Change |
|---|---:|---:|---:|
| global A2 RMSE | 0.8023 C | 0.7165 C | -10.7% |
| NA 40--60N raw bias | -0.4926 C | -0.2948 C | +40.2% |
| near-wall raw bias | -0.5569 C | -0.2045 C | +63.3% |
| heat drift | -0.1063% | -0.0441% | better |
| salt drift | -0.000189% | -0.000183% | similar |
| below-freezing cells | 0 | 0 | same |
| mean MLD | 31.5 m | 31.5 m | same |

Both runs pass the stability watchdog.

## Next step

Run a 365d A/B with the same flags. Do not promote the closure as
production default until that annual check reproduces.
