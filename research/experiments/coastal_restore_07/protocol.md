# Coastal T Restoring Diagnostic Protocol

Status: complete
Date: 2026-09-23
Baseline: `lambda160 + min-depth500 + smooth80`
Question: How much of the coastal-band cold bias can be attributed to a local
surface boundary condition?

## Change

Add a diagnostic surface-temperature restoring term only in the land-adjacent
`0..7`-cell band, relaxing SST toward the WOA initial surface temperature.
The production candidate remains unchanged when the flag is off.

## Runs

1. 30d restoring-strength probes with tau=10d, 3d, 1d, 0.5d, and 0.25d.
2. 365d runs with tau=3d, 1d, and 0.5d at cells<=7.

## Result

The 365d tau=0.5d, cells<=7 run improves global A2 from `1.294 C` to
`1.154 C`, NA 40--60N from `0.997 C` to `0.880 C`, and near-wall 55--60N from
`1.019 C` to `0.804 C`. The tau=0.25d probe is stronger but approaches direct
SST assimilation.

## Decision

Keep `tau=0.5d, cells<=7` as the best diagnostic upper bound. It is not a
production default. Next replace it with a defensible coastal boundary-layer
or mixing closure.
