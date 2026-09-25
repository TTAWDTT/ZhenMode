# Mixed-layer / Sea-Ice Closure Benchmark Ladder

Status: smoke run passed; 30d A/B pending  
Date: 2026-09-26

## Question

Does the minimal mixed-layer / sea-ice closure improve or at least preserve
the current `candidate_65n_045_icefloor` metrics without destabilizing the
solver?

## Control

Use the locked 365d run:

```text
results/ice_proxy_045/global_res045_icefloor_365d_repeat.npz
```

Key metrics are already recorded in:

```text
research/experiments/ice_proxy_045/benchmark_365d_repeat.json
```

## Experimental flags

1. `--mixed-layer-depth 50`
   - spreads surface heat flux over a 50 m well-mixed slab instead of the
     top 5 m surface node.
2. `--ice-salt-flux 1e-7`
   - adds a brine-rejection salt flux only where live SST <= `-1.8 C`.

No other physics flags may change in the A/B comparison.

## Ladder

1. **1d 1-degree smoke run** — `PASS`
2. **30d 0.45-degree A/B probe**
3. **365d 0.45-degree A/B check** only if the 30d probe is stable and not
   worse than the baseline.

## Decision rules

1. A 30d run must be stable (`PASS`), not worsen global A2, NA 40--60N A2,
   near-wall bias, or salt drift.
2. If the 30d run improves those metrics, run a 365d A/B.
3. Do not promote the mixed-layer/ice closure as production default until the
   365d check reproduces.
4. If the closure only helps one metric and worsens another, reject it.

## Smoke gate

`results/mixed_layer_ice_smoke/global_mixed_layer_ice_smoke.npz`

- 1-day 1-degree run
- verdict `PASS`
- mean MLD `31.7 m`
- zero sub-freezing cells
- benchmark file:
  `research/experiments/mixed_layer_ice_045/benchmark_smoke_1d.json`
