# Gentle Coastal Constraint on the Ice-Floor Baseline

Status: active
Date: 2026-09-24

## Context

The 0..3-cell coastal band still contains `47.9%` of global A2 SSE in the new
0.45-degree ice-floor candidate. Simple diffusion and marine-air smoothing are
rejected. The remaining hypothesis is unresolved boundary-current/eddy heat
transport.

## Interpretation

A gentle coastal SST constraint is not a forecastable sea-ice or boundary
current model. Treat it as a controlled proxy for the missing lateral boundary
heat transport. The goal is to find the weakest useful strength, not to maximize
a metric.

## First ladder

Use the ice-floor candidate unchanged, plus `cells<=7` coastal SST restore:

1. `tau=30d`
2. `tau=10d`

Run 30d A/B first. Promote only a rung that improves global, North Atlantic,
and near-wall metrics without a stability cost. Then run 365d; if repeated,
consider it a diagnostic parameterization, not an unqualified production
default.

## Metrics

- 0..3 and 4..7 land-distance band raw bias/RMSE
- global A2
- North Atlantic `40..60N` A2
- near-wall `55..60N` raw bias
- max velocity and verdict
