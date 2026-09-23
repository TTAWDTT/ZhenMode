# Coastal T Restoring Diagnostic Result

Status: complete
Date: 2026-09-23
Baseline: `lambda160 + min-depth500 + smooth80`

## 30d restoring-strength sweep (cells<=7)

| tau | global 0..7 | NA 0..7 | near-wall 0..7 |
|---:|---:|---:|---:|
| baseline | -0.644 / 1.079 C | -0.564 / 0.838 C | -0.950 / 1.064 C |
| 10d | -0.578 / 0.986 C | -0.510 / 0.759 C | -0.863 / 0.969 C |
| 3d | -0.465 / 0.828 C | -0.415 / 0.621 C | -0.707 / 0.799 C |
| 1d | -0.296 / 0.609 C | -0.268 / 0.406 C | -0.458 / 0.524 C |
| 0.5d | -0.193 / 0.490 C | -0.175 / 0.268 C | -0.298 / 0.343 C |
| 0.25d | -0.116 / 0.414 C | -0.103 / 0.161 C | -0.175 / 0.203 C |

## 365d metrics

| run | global A2 | NA 40--60N | near-wall 55--60N |
|---|---:|---:|---:|
| baseline | 1.294 C | 0.997 C | 1.019 C |
| tau=3d, cells<=7 | 1.229 C | 0.933 C | 0.885 C |
| tau=1d, cells<=7 | 1.181 C | 0.894 C | 0.816 C |
| tau=0.5d, cells<=7 | 1.154 C | 0.880 C | 0.804 C |

## Interpretation

1. Stronger coastal SST restoring improves all metrics monotonically.
2. `tau=0.5d, cells<=7` is already a strong upper-bound diagnostic; pushing to
   `tau=0.25d` would be closer to direct SST assimilation than a physical
   closure.
3. The improvement is largest in the immediate coastal and transitional bands,
   but it also improves the global metric substantially.

## Decision

Keep `tau=0.5d, cells<=7` as the best diagnostic upper bound. It is not a
production default. The next step is to replace this SST constraint with a
defensible coastal boundary-layer or mixing closure.
