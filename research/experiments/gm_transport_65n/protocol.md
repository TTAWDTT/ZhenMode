# GM/Bolus Transport Sensitivity on the lambda80 65N Candidate

Status: locked
Date: 2026-09-21
Baseline: `65N + lambda_bulk=80 + reduced vertical mixing + annual real air`

## Question

Is the remaining North Atlantic cold bias controlled by unresolved eddy-induced
heat transport? The strongest residual cold-error cells receive the largest
surface warming, so the missing process may be ocean heat transport rather than
surface restoring.

## Runs

Keep all 65N candidate settings unchanged. Vary only GM bolus diffusivity:

1. `gm0`: `--kappa-gm 0`
2. `gm3000`: `--kappa-gm 3000`

The reference is `lambda80` with `kappa_gm=1000`. No Redi diffusion is added,
so the test isolates adiabatic bolus transport.

## Metrics

- Global A1/A2 correlations and RMSEs
- North Atlantic `300..360E / 40..60N` bias and RMSE
- near-wall `55..60N` bias and RMSE
- 365d stability, heat/salt drift, KE behavior

## Decision rules

1. If `gm3000` improves the target region and remains stable, unresolved bolus
   heat transport is likely too weak.
2. If `gm0` improves, the current GM strength is excessive or mis-tapered.
3. If both move global and regional metrics in opposite directions, decompose
   the error into transport-dominated and surface-dominated sectors before any
   further tuning.
4. If neither materially improves, move to mixed-layer/convection closure.

## Follow-up run

After the initial two runs, `kappa_gm=500` was added as a non-zero
intermediate point. This keeps a physical GM closure at 1 degree while testing
whether the original strength was excessive.
