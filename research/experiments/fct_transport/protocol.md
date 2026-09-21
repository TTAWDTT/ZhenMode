
# Experiment: full 3D flux-corrected-transport tracer option

## Purpose
Test whether a bounded, conservative, 3D FCT transport scheme improves the
robustness of the current 1° global solver without paying too much runtime.

## Why
Mature models such as FESOM2 and NEMO commonly use a bounded/monotone high-order
scheme plus a low-order fallback. FESOM2 applies the limiter to full 3D fluxes,
which is attractive because vertical advection can be the practical bottleneck near
coastlines. `ocean_solver` currently has centered and donor-cell horizontal
advection, but no unified 3D FCT limiter.

## Hypothesis
A full 3D flux-corrected-transport option will reduce pathological tracer pileup
and drift relative to both centered and donor-cell transport, especially in
narrow/complex topography, while keeping the run stable.

## Prediction
At 1°/30d and 1°/365d, FCT should produce:
- lower max |T| than centered,
- similar or better tracer drift than donor-cell,
- similar max |u| and max |eta|,
- wall time not more than ~1.5× donor-cell.

## Protocol
1. Use the existing 1° global configuration with real ETOPO2022 bathymetry.
2. Run three tracer advection modes on the same forcing and initial state:
   - centered (current baseline),
   - monotone donor-cell (current bounded fallback),
   - 3D FCT limiter.
3. Metrics:
   - max |T|,
   - max |u|,
   - max |eta|,
   - global mean T drift,
   - global heat/salt budget residual,
   - wall time.
4. Run lengths:
   - 30 days first,
   - 365 days if 30 days passes.
5. Decision rule:
   - If FCT improves drift and stability without excessive cost, promote it as a
     non-default option first and consider becoming default after longer runs.
   - If it is only marginal, keep it as an experimental option.
   - If it fails, document why and move to vertical-coordinate remap instead.

## Note
This protocol should be committed before the FCT implementation is added.
