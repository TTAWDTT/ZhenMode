# Coastal T Restoring Diagnostic Result

Status: complete
Date: 2026-09-23
Baseline: `lambda160 + min-depth500 + smooth80`

## 30d width sweep (tau=3d)

| band | global 0..3 | global 0..7 | NA 0..7 | near-wall 0..7 |
|---|---:|---:|---:|---:|
| baseline | -0.760 / 1.290 C | -0.644 / 1.079 C | -0.564 / 0.838 C | -0.950 / 1.064 C |
| cells<=1 | -0.669 / 1.144 C | -0.602 / 1.001 C | -0.554 / 0.806 C | -0.922 / 1.021 C |
| cells<=3 | -0.560 / 1.003 C | -0.546 / 0.920 C | -0.514 / 0.745 C | -0.847 / 0.928 C |
| cells<=5 | -0.550 / 0.992 C | -0.502 / 0.867 C | -0.456 / 0.674 C | -0.758 / 0.841 C |
| cells<=7 | -0.551 / 0.992 C | -0.465 / 0.828 C | -0.415 / 0.621 C | -0.707 / 0.799 C |
| cells<=9 | -0.550 / 0.991 C | -0.460 / 0.823 C | -0.408 / 0.615 C | -0.698 / 0.793 C |

## 365d metrics

| run | global A2 | NA 40--60N | near-wall 55--60N |
|---|---:|---:|---:|
| baseline | 1.294 C | 0.997 C | 1.019 C |
| coastal-restore tau=3d, cells<=3 | 1.253 C | 0.978 C | 0.979 C |
| coastal-restore tau=3d, cells<=7 | 1.229 C | 0.933 C | 0.885 C |
| coastal-restore tau=3d, cells<=9 | 1.217 C | 0.913 C | 0.866 C |

## Interpretation

1. The restoring band-width sweep shows a clear monotonic improvement from the
   immediate coast into the `0..7`-cell coastal/transitional zone.
2. The gain from `0..7` to `0..9` is smaller, so `0..7` is a reasonable
   diagnostic width.
3. The 365d `cells<=7` run improves global A2 from `1.294 C` to `1.229 C`, NA
   from `0.997 C` to `0.933 C`, and near-wall from `1.019 C` to `0.885 C`.

## Decision

Keep `lambda160 + min-depth500 + smooth80 + coastal T restore tau=3d,
cells<=7` as the current diagnostic candidate. It is not a production default
and should be replaced later by a more physical coastal closure.
