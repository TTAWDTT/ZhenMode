# Marine-Air Target Correction at 0.45 Degree

Status: rejected
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

## 30d A/B

| marine smooth passes | global A2 | NA 40--60N A2 | near-wall raw bias | verdict |
|---:|---:|---:|---:|---|
| 0 | 0.9041 C | 0.8494 C | -0.7426 C | control |
| 5 | 0.9029 C | 0.8539 C | -0.7467 C | reject |
| 20 | 0.9028 C | 0.8658 C | -0.7571 C | reject |

Although global A2 improves slightly, both smoothing levels make the North
Atlantic and near-wall cells colder. This violates the all-around promotion
rule.

## Decision

Reject wet-cell-only marine-air smoothing. Keep the 0.45-degree ice-floor
candidate unchanged. The remaining coastal issue needs a boundary-current or
lateral heat-transport parameterization, not a broader smoothing of the
atmospheric target.

## Note

The first implementation set land target values to NaN and caused a false
blow-up. This was fixed before scoring: the smoother now updates only wet cells
and leaves land values unchanged.
