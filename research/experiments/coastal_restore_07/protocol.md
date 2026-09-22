# Coastal T Restoring Diagnostic Protocol

Status: complete
Date: 2026-09-23
Baseline: `lambda160 + min-depth500 + smooth80`
Question: How much of the 0..3-cell coastal cold bias can be attributed to a
local surface boundary condition?

## Change

Add a diagnostic surface-temperature restoring term only in the land-adjacent
`0..3`-cell band, relaxing SST toward the WOA initial surface temperature.
The production candidate remains unchanged when the flag is off.

## Runs

1. 30d probes with tau=30d, 10d, and 3d: all PASS.
2. 365d run with tau=10d: PASS.
3. 365d run with tau=3d: PASS.

## Result

The 365d tau=3d run improves global A2 from `1.294 C` to `1.253 C`, NA 40--60N
from `0.997 C` to `0.978 C`, and near-wall 55--60N from `1.019 C` to
`0.979 C`. The tau=10d run is less aggressive and still improves all three.

## Decision

Keep the restoring experiment as a diagnostic branch. It confirms that the
immediate-land band is a first-order boundary-condition error, but it is not yet
a physical closure or a production default.
