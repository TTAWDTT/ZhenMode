# Coastal T Restoring Diagnostic Result

Status: complete
Date: 2026-09-23
Baseline: `lambda160 + min-depth500 + smooth80`

## 30d probes

| tau | global 0..3 | NA 0..3 | near-wall 0..3 |
|---:|---:|---:|---:|
| baseline | -0.760 / 1.290 C | -0.491 / 0.906 C | -1.237 / 1.380 C |
| 30d | -0.734 / 1.252 C | -0.478 / 0.880 C | -1.203 / 1.342 C |
| 10d | -0.688 / 1.184 C | -0.453 / 0.833 C | -1.140 / 1.273 C |
| 3d | -0.560 / 1.003 C | -0.382 / 0.700 C | -0.959 / 1.076 C |

## 365d metrics

| run | global A2 | NA 40--60N | near-wall 55--60N |
|---|---:|---:|---:|
| baseline | 1.294 C | 0.997 C | 1.019 C |
| coastal-restore tau=10d | 1.277 C | 0.989 C | 1.002 C |
| coastal-restore tau=3d | 1.253 C | 0.978 C | 0.979 C |

## Interpretation

1. The local restoring monotonically improves the `0..3`-cell cold bias.
2. It also improves the global and regional metrics, so the immediate-land band
   is not just a local cosmetic problem.
3. `tau=10d` gives most of the regional gain with a smaller perturbation;
   `tau=3d` is stronger and gives the best metrics, but is more artificial.

## Decision

Keep this as a diagnostic branch, not a production default. The next physical
step is to replace the empirical restoring with a defensible coastal boundary
or mixing closure.
