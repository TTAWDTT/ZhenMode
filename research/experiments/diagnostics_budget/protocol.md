# Experiment: heat / salt / mass budget diagnostics

## Purpose
Before adding more closures, make the current solver's budgets explicit enough to
tell whether an instability is a numerical transport problem, a mask/coordinate
problem, or a forcing/boundary-condition problem.

## Why
The FCT transport experiment shows that changing tracer transport barely changes
the large-scale climatology. That means the dominant error is probably elsewhere.
Before changing vertical coordinates or adding new physics, we need a cheap,
reliable way to see where heat, salt, mass, and energy are entering or leaking.

## Hypothesis
A standardized budget layer will expose one or more residual patterns that the
current end-state metrics (`max|T|`, `max|u|`, `max|eta|`) miss.

## Protocol
1. Add a small diagnostics module, separate from the solver core.
2. Compute, per snapshot:
   - global mean SST / SSS,
   - global heat content proxy,
   - global salt content,
   - total volume,
   - max / RMS eta,
   - KE,
   - optional energy tendency.
3. Save the time series in the run output, not only final scalars.
4. Add one test that verifies:
   - a zero-forcing, zero-diffusion run has constant global heat and salt content,
   - a closed-boundary run has zero net tracer flux through walls.
5. Rerun the existing 1°/365d baseline with the diagnostics enabled.
6. Compare centered vs FCT:
   - the main question is whether their heat/salt residuals differ,
   - if they are the same, transport is confirmed not to be the bottleneck.

## Decision rule
- If the diagnostics reveal a clear budget leak, fix that leak first.
- If they show clean budgets, move on to vertical-coordinate remap / z-star.
