# Marine-Air Target Correction at 0.45 Degree

Status: active
Date: 2026-09-24

## Motivation

For the new ice-floor candidate, the annual 2m air target is colder than WOA
SST by:

| land-distance band | mean target minus WOA |
|---:|---:|
| 0..3 cells | `-0.933 C` |
| 4..7 cells | `-0.432 C` |
| 8..14 cells | `-0.507 C` |
| >=15 cells | `-0.682 C` |

The model's 0..3-cell raw bias is `-1.23 C`, so a substantial part of the
remaining coastal error is consistent with land-contaminated atmospheric
forcing, not only ocean dynamics.

## Test

Add an opt-in wet-cell-only smoother for the bulk air target. It does not
interpolate through land and keeps the large-scale NCEP target. Compare:

1. control: candidate ice floor, no smoothing;
2. `--air-marine-smooth-passes 5`;
3. `--air-marine-smooth-passes 20`.

Run 30d first. Promote only a rung that improves the 0..3-cell/global A2 error
without losing North Atlantic or near-wall skill.

## Decision rules

- If all smoothed targets degrade the global metric, reject this forcing edit.
- If one improves, run 365d with the smallest sufficient smoothing.
- Keep the ice floor on in every run; this is the current physical baseline.
