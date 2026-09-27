# Stage-G dynamic-ice + fixed-MLD pre-registration

Date: 2026-09-28
Status: pre_registered, diagnostic-only, not launched

## Purpose

Extend the Stage-G full-bulk forcing contract with the existing minimal
sea-ice/mixed-layer closure.  This is the next diagnostic rung after the
no-ice Stage-G control; it is not a claim that the closure is a production
sea-ice model.

## Candidate

- shared 720x260 0.5-degree grid;
- Stage-G monthly NCEP full-bulk forcing;
- dynamic ice in 40--65N;
- fixed 100m mixed layer in 40--60N;
- same WOA initial state and MOM6-derived shared topography;
- 30d stability smoke first, then 365d annual only if the 30d run passes.

## Blocking gate

Do not launch until:

1. the annual MOM6 Stage-F v12 final-90d 3D/MLD gate is scored; and
2. the no-ice Stage-G 30d/365d control has been scored.

The candidate must then beat or match the no-ice Stage-G control on the same
pre-registered gates before any promotion.
