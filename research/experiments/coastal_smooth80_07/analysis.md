# Coastal Geometry Sensitivity Result: smooth-passes 80

Status: complete
Date: 2026-09-22
Baseline: `lambda160 + min-depth500 + smooth30`

## Metrics

| lambda | smooth passes | global A2 | NA 40--60N | near-wall 55--60N |
|---:|---:|---:|---:|---:|
| 160 | 30 | 1.304 C | 1.009 C | 1.010 C |
| 160 | 80 | 1.294 C | 0.997 C | 1.019 C |
| 120 | 80 | 1.281 C | 1.044 C | 1.063 C |
| 80 | 80 | 1.269 C | 1.143 C | 1.148 C |

The lambda160 + smooth80 repeat gives the same metrics:
global `1.294 C`, NA `0.997 C`, near-wall `1.019 C`.

## Interpretation

1. Smooth80 improves the global and North Atlantic metrics for both lambda80
   and lambda160.
2. For lambda160, the near-wall metric is slightly worse than smooth30.
3. The all-around tradeoff still favors lambda160 + smooth80, but the remaining
   immediate-land cold bias is not solved by more bathymetry smoothing alone.

## Decision

Promote `lambda160 + min-depth500 + smooth80` as the all-around diagnostic
candidate. Keep `lambda80 + smooth80` as the pure global-A2 candidate. Stop the
lambda scan and isolate the 0--3-cell coastal band next.
