# Coastal T Restoring Diagnostic Protocol

Status: complete
Date: 2026-09-23
Baseline: `lambda160 + min-depth500 + smooth80`
Question: How much of the 0..3-cell coastal cold bias can be attributed to a
local surface boundary condition?

## Change

Add a diagnostic surface-temperature restoring term only in the land-adjacent
`0..7`-cell band, relaxing SST toward the WOA initial surface temperature.
The production candidate remains unchanged when the flag is off.

## Runs

1. 30d probes with tau=30d, 10d, and 3d: all PASS.
2. 30d width probes with cells<=1, 3, 5, 7, 9: all PASS.
3. 365d runs with tau=3d and cells<=3, 7, 9: all PASS.

## Result

The 365d tau=3d, cells<=7 run improves global A2 from `1.294 C` to `1.229 C`,
NA 40--60N from `0.997 C` to `0.933 C`, and near-wall 55--60N from `1.019 C`
to `0.885 C`. The tau=3d, cells<=9 run improves slightly further but covers a
much larger fraction of the ocean.

## Decision

Keep `tau=3d, cells<=7` as the current diagnostic candidate. It is not a
production default and should be replaced later by a more physical coastal
closure.
