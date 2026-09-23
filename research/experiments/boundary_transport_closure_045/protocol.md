# Boundary Transport Closure A/B at 0.45 Degree

Status: registered
Date: 2026-09-24

## Question

After rejecting simple coastal diffusion, can a standard lateral
transport closure improve the North Atlantic/near-wall cold band without
degrading global A2?

## Fixed reference

Use the 0.45-degree lambda80 candidate after its reproducibility run passes.
Keep `dt=1800s`, `nu_h=2e6`, `area` remapping, annual real air forcing, and all
other candidate flags unchanged.

## First A/B ladder

1. Control: `kappa_gm=0`.
2. Gentle GM: `kappa_gm=500 m2/s`.
3. Moderate GM: `kappa_gm=1000 m2/s`.

Run 30d first. If a probe passes and improves both global A2 and
North Atlantic/near-wall metrics, run the same setting for 365d.

## Metrics

- integration verdict and max velocity
- global A2 SST RMSE
- North Atlantic `40..60N` A2 RMSE
- near-wall `55..60N` raw bias and A2 RMSE
- runtime

## Decision rules

- Reject a closure if global A2 worsens by more than 1% or near-wall bias becomes colder.
- Promote to 365d only if the 30d probe improves the all-around comparison.
- Keep any diagnosed coastal restore separate from the production candidate.
- Do not add a second new closure before interpreting the single-lever A/B result.

## Rationale

GM/Redi is the mature-model lateral eddy-transport closure, whereas the earlier
local diffusion and bulk-flux probes were rejected. The first rung should test
the existing closure at the finer grid rather than adding a bespoke scheme.
