# 3D Heat-Tendency Decomposition on the Preferred Diagnostic Candidate

Status: locked
Date: 2026-09-21
Candidate: `65N + lambda80 + GM500 + localized convective adjustment + annual real air`

## Question

Which physical process controls the remaining North Atlantic cold bias after
the scalar and closure improvements?

## Run

Use the preferred diagnostic candidate for one 365d integration. Save a
30-day-interval 3D tracer/state snapshot and its heat-tendency decomposition:

- advection
- horizontal diffusion
- vertical diffusion
- convective adjustment
- GM/bolus skew transport
- Redi diffusion

Surface bulk heat flux is not included in the saved 6-term stack, so it is
diagnosed separately from `lambda*(T_atm-SST)`.

## Metrics

For North Atlantic `300..360E / 40..60N` and near-wall `55..60N`:

- depth-binned mean and RMS heat tendencies for every term
- regional surface bulk heat-flux tendency
- correlation between local model-WOA SST error and each tendency
- late-integration mean over the final 90 days

## Decision rules

1. If advection dominates the residual regional heat budget, prioritize heat
   transport / current-structure improvements.
2. If convection dominates, refine convective closure despite the SST gain.
3. If GM dominates, tune its taper or spatial dependence rather than scalar
   strength.
4. If surface bulk flux dominates, revisit atmospheric forcing or sea-ice proxy.
5. If diffusion dominates, inspect vertical diffusivity and boundary-layer
   treatment.
