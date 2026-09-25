# Mixed-layer / Sea-Ice 0.5-degree Annual Check

Date: 2026-09-26  
Status: running

## Question

Does the 30d mixed-layer/ice improvement survive a full annual integration at
the 0.5-degree resolution?

## Run

```bash
bash scripts/run_mixed_layer_ice_050.sh mixed_layer_ice_050_365d
```

Flags:

- 0.5 degree, 65N domain, 365d
- `--mixed-layer-depth 50`
- `--ice-salt-flux 1e-7`
- annual real 2m air forcing
- seasonal 2023 wind
- ice-air floor at -1.8 C
- one final snapshot to limit I/O

## Control

Use the existing 0.45-degree annual control as the internal reference:

- `results/ice_proxy_045/global_res045_icefloor_365d_repeat.npz`
- `research/experiments/ice_proxy_045/benchmark_365d_repeat.json`

This is not a same-resolution control.  If the 0.5-degree annual run passes,
the next step is a same-resolution 0.5-degree control/repeat ladder.

## Gate

1. Verdict `PASS`.
2. Heat and salt drifts bounded.
3. Global A2 not worse than the current internal annual control.
4. NA 40--60N RMSE not worse.
5. Near-wall bias not worse.
